import os
import time
import gc

from typing import Optional, List, Tuple

import torch
from torch import nn
from torch.utils.data import DataLoader, random_split

import psutil

from pathlib import Path
from tqdm import tqdm

from .config import TrainerConfig, set_random_seed, TrainHistory
from .logger import LoggerConfig, Logger
from .hooks import HookManager, Hook


class Trainer:
    def __init__(
        self,
        model: nn.Module,
        config: TrainerConfig,
        train_dataloader: DataLoader,
        validation_dataloader: Optional[DataLoader] = None,
        logger_config: Optional[LoggerConfig] = None,
        hooks: Optional[List[Hook]] = None,
        pretrained_path: Optional[str | Path] = None,
    ) -> None:
        # TODO:
        # 1) lr_scheduler with params from user
        # 2) user friendly validation with params
        # 3) Metric evaluator
        # 4) TensorBoard hook on validation
        # 5) Import / Export train params
        # 6) Model to different formats
        # 7) wandb support?
        # 8) OpenMP support?
        # 9) Memory checker (allocated & needed really)
        # 10) pytorch lighting support?
        # 11) Save & log more info
        self.config = config
        self.train_dir = self._create_train_directory()
        self._logger = self._initialize_logger(logger_config)
        self.model = self._initialize_model(model, pretrained_path)
        self.optimizer = self._initialize_optimizer()
        self.use_custom_train_step = hasattr(model, "train_step")
        self.train_dataloader = train_dataloader
        self.use_validation = (
            validation_dataloader is not None or self.config.use_auto_validation
        )
        self.validation_dataloader = self._initialize_validation_dataloader(
            validation_dataloader
        )
        self._use_custom_validation_step = self.use_validation and hasattr(
            model, "validation_step"
        )
        self.hook_manager = self._initialize_hooks(hooks)
        self._set_seed()
        self._train_info_dir = self._create_train_info_directory()
        self.validation_loss = []
        self.epoch_train_loss = []
        self.batch_train_loss = []
        self.train_time_per_epoch = []

        torch.set_float32_matmul_precision("high")
        self.scaler = torch.amp.GradScaler(enabled=self.config.use_amp)
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True

    def save_model(self, epoch: int, best: bool = False) -> None:
        pretrained_path = os.path.join(self.train_dir, "pretrained")
        os.makedirs(pretrained_path, exist_ok=True)
        filename = f"model_epoch_{epoch}.pth" if not best else "best_model.pth"
        filepath = os.path.join(pretrained_path, filename)
        torch.save(self.model.state_dict(), filepath)
        self._logger.info(f"Model saved at {filepath}")

    def load_model(self, model: nn.Module, filepath: str) -> None:
        model.load_state_dict(
            torch.load(filepath, weights_only=True, map_location=self.config.device)
        )
        return model

    def _initialize_model(
        self, model: nn.Module, pretrained_path: Optional[str | Path] = None
    ) -> nn.Module:
        model = model.to(self.config.device)
        if pretrained_path is not None:
            model = self.load_model(model, pretrained_path)
            self._logger.info(f"Model loaded from {pretrained_path}")
        if self.config.do_model_compile:
            model = torch.compile(model)
            self._logger.info(f"Model compiled for {self.config.device.type}")
        return model

    def _initialize_optimizer(self) -> torch.optim.Optimizer:
        return self.config.optimizer_type(
            self.model.parameters(), **self.config.optimizer_params
        )

    def _initialize_logger(
        self, logger_config: Optional[LoggerConfig] = None
    ) -> Logger:
        logger_config = logger_config if logger_config is not None else LoggerConfig()
        if logger_config.file_log:
            logger_config.log_dir = os.path.join(self.train_dir, "logs")
        logger = Logger(logger_config)

        if logger_config.file_log:
            logger.info(f"Training directory created at: {self.train_dir}")
        return logger

    def _initialize_validation_dataloader(
        self, validation_dataloader: Optional[DataLoader]
    ) -> Optional[DataLoader]:
        if validation_dataloader:
            if self.config.use_auto_validation:
                self._logger.warning(
                    "Validation dataloader provided; skipping auto-validation."
                )
            return validation_dataloader
        if self.config.use_auto_validation:
            return self._create_train_val_dataloaders()[1]
        self._logger.warning(
            "No validation dataloader; consider enabling auto-validation."
        )
        return None

    def _initialize_hooks(self, hooks: Optional[List[Hook]]) -> HookManager:
        hook_manager = HookManager()
        if hooks:
            for hook in hooks:
                hook.set_output_dir(self.train_dir)
                hook_manager.add_hook(hook)
        return hook_manager

    def _set_seed(self) -> None:
        if self.config.seed is not None:
            self._logger.info(f"Setting random seed: {self.config.seed}")
            set_random_seed(self.config.seed)

    def _create_train_directory(self) -> str:
        train_dir = os.path.join(self.config.output_dir, self.config.train_name)
        os.makedirs(train_dir, exist_ok=True)
        return train_dir

    def _create_train_info_directory(self) -> str:
        train_info_dir = os.path.join("trainer_info")
        os.makedirs(train_info_dir, exist_ok=True)
        self._logger.info(f"Training info directory created: {train_info_dir}")
        return train_info_dir

    def _create_train_val_dataloaders(
        self, val_split: float = 0.2
    ) -> Tuple[DataLoader, DataLoader]:
        dataset = self.train_dataloader.dataset
        total_len = len(dataset)
        val_len = int(total_len * val_split)
        train_len = total_len - val_len
        train_subset, val_subset = random_split(dataset, [train_len, val_len])
        batch_size = self.train_dataloader.batch_size
        train_loader = DataLoader(
            train_subset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=self.train_dataloader.num_workers,
            drop_last=True,
        )
        val_loader = DataLoader(
            val_subset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=self.train_dataloader.num_workers,
            drop_last=True,
        )
        return train_loader, val_loader

    def __base_train_step(self, batch, batch_idx: int) -> float:
        inputs, targets = batch
        inputs = inputs.to(self.config.device)
        targets = targets.to(self.config.device)
        with torch.autocast(
            device_type=self.config.device.type, enabled=self.config.use_amp
        ):
            outputs = self.model(inputs)
            loss = self.config.criterion(outputs, targets)
        return loss

    def __base_validation_step(self, batch, batch_idx: int) -> float:
        inputs, targets = batch
        inputs = inputs.to(self.config.device)
        targets = targets.to(self.config.device)
        with torch.autocast(
            device_type=self.config.device.type, enabled=self.config.use_amp
        ):
            outputs = self.model(inputs)
            loss = self.config.criterion(outputs, targets)
        return loss

    def validate(self) -> float:
        self.model.eval()
        running_loss = 0.0
        self._logger.info("Start validation loop...")

        with torch.inference_mode():
            with tqdm(
                total=len(self.validation_dataloader), desc="Validation", unit="batch"
            ) as pbar:
                for batch_idx, batch in enumerate(self.validation_dataloader):
                    loss = (
                        self.model.validation_step(
                            batch, self.config.device, self.config.criterion, batch_idx
                        )
                        if self._use_custom_validation_step
                        else self.__base_validation_step(batch, batch_idx)
                    )
                    running_loss += loss.item()
                    pbar.set_postfix({"Batch Loss": loss.item()})
                    pbar.update(1)

        if str(self.config.device) in "cuda":
            torch.cuda.empty_cache()
            self._logger.info("CUDA cache cleared after validation")

        cleared_objects = gc.collect()
        self._logger.info(f"Cleared cache gc: {cleared_objects} objects")

        return running_loss / len(self.validation_dataloader.dataset)

    def train(self) -> None:
        self._logger.info(
            f"Starting training loop for {self.model._get_name()} on {self.config.device}"
        )
        self._logger.info(
            f"Using AMP: {'enabled' if self.config.use_amp else 'disabled'}"
        )
        self.hook_manager.execute("on_train_start", trainer=self)
        try:
            for epoch in range(1, self.config.epochs + 1):
                self._train_epoch(epoch)
        except KeyboardInterrupt:
            self._logger.warning("Train loop interrupted by user...")
            self._logger.warning("Executing hooks for on_train_end action")
            self.hook_manager.execute("on_train_end", trainer=self)
            return
        self._logger.info("Training complete")
        self.hook_manager.execute("on_train_end", trainer=self)

    def _train_epoch(self, epoch: int) -> None:
        self.hook_manager.execute("on_epoch_start", trainer=self, epoch=epoch)
        self.model.train()
        running_loss = 0.0
        start_time = time.time()

        with tqdm(
            total=len(self.train_dataloader),
            desc=f"Epoch [{epoch}/{self.config.epochs}]",
            unit="batch",
        ) as pbar:
            for batch_idx, batch in enumerate(self.train_dataloader):
                loss = self._train_batch(batch, batch_idx, epoch)
                running_loss += loss

                pbar.set_postfix({"Batch Loss": loss})
                pbar.update(1)

        self._finalize_epoch(running_loss, start_time, epoch)

    def _train_batch(self, batch, batch_idx: int, epoch: int) -> float:
        self.hook_manager.execute(
            "on_batch_start",
            trainer=self,
            epoch=epoch,
            batch_idx=batch_idx,
            batch=batch,
        )
        self.optimizer.zero_grad()

        with torch.autocast(
            device_type=self.config.device.type, enabled=self.config.use_amp
        ):
            loss = (
                self.model.train_step(
                    batch, self.config.device, self.config.criterion, batch_idx
                )
                if self.use_custom_train_step
                else self.__base_train_step(batch, batch_idx)
            )

        self.scaler.scale(loss).backward()
        self.scaler.step(self.optimizer)
        self.scaler.update()

        self.batch_train_loss.append(loss.item())
        self.hook_manager.execute(
            "on_batch_end",
            trainer=self,
            epoch=epoch,
            batch_idx=batch_idx,
            loss=loss.item(),
        )
        return loss.item()

    def _finalize_epoch(self, running_loss: float, start_time: float, epoch: int):
        epoch_loss = running_loss / len(self.train_dataloader.dataset)
        self.epoch_train_loss.append(epoch_loss)
        epoch_time = time.time() - start_time
        self.train_time_per_epoch.append(epoch_time)
        self._logger.info(
            f"Epoch [{epoch}/{self.config.epochs}], Loss: {epoch_loss:.4f}, Time: {epoch_time:.2f}s"
        )
        if self.use_validation:
            val_loss = self.validate()
            self.validation_loss.append(val_loss)
            self._logger.info(f"Validation Loss: {val_loss:.4f}")
        self.hook_manager.execute(
            "on_epoch_end", trainer=self, epoch=epoch, epoch_loss=epoch_loss
        )
        if epoch % self.config.save_weights_per_epoch == 0 and epoch > 0:
            self.save_model(epoch, best=False)
        self._log_memory_usage()
        self._log_gradients()

        if str(self.config.device) in "cuda":
            torch.cuda.empty_cache()
            self._logger.info(f"CUDA cache cleared after {epoch} epoch")

        self._logger.info(f"Cleared cache: {gc.collect()} objects")

    def _log_gradients(self) -> None:
        total_norm = 0
        for p in self.model.parameters():
            if p.grad is not None:
                param_norm = p.grad.data.norm(2)
                total_norm += param_norm.item() ** 2
        total_norm = total_norm**0.5
        self._logger.info(f"Gradient Norm: {total_norm:.4f}")

    def _log_memory_usage(self) -> None:
        process = psutil.Process(os.getpid())
        mem = process.memory_info().rss / (1024 * 1024)
        self._logger.info(f"Memory Usage: {mem:.2f} MB")

    def get_history(self) -> TrainHistory:
        return TrainHistory(
            self.epoch_train_loss,
            self.batch_train_loss,
            self.validation_loss,
            self.train_time_per_epoch,
        )
