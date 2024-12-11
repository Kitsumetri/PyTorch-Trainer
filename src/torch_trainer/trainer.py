from typing import Optional, List, Tuple

from tqdm import tqdm
import os
import time

import torch
from torch import nn
from torch.utils.data import DataLoader, random_split
from .config import TrainerConfig, set_random_seed
from .logger import LoggerConfig, Logger
from .hooks import HookManager, Hook
from pathlib import Path
from dataclasses import dataclass, field

@dataclass
class TrainHistory:
    epoch_train_loss: List[float] = field(default_factory=lambda x: [])
    batch_train_loss: List[float] = field(default_factory=lambda x: [])
    validation_loss: List[float] = field(default_factory=lambda x: [])
    train_time_per_epoch: List[int | float] = field(default_factory=lambda x: [])


class Trainer:
    def __init__(
        self,
        model: nn.Module,
        config: TrainerConfig,
        train_dataloader: DataLoader,
        validation_dataloader: Optional[DataLoader] = None,
        logger_config: Optional[LoggerConfig] = None,
        hooks: Optional[List[Hook]] = None,
        pretrained_path: Optional[str | Path] = None
    ) -> None:
        self.config = config
        self.train_dir = self._create_train_directory()
        self._logger = self._initialize_logger(logger_config)
        self.model = self._initialize_model(model, pretrained_path)
        self.optimizer = self._initialize_optimizer()
        self.use_custom_train_step = hasattr(model, 'train_step')
        self.train_dataloader = train_dataloader
        self.use_validation = validation_dataloader is not None or self.config.use_auto_validation
        self.validation_dataloader = self._initialize_validation_dataloader(validation_dataloader)
        self._use_custom_validation_step = self.use_validation and hasattr(model, 'validation_step')
        self.hook_manager = self._initialize_hooks(hooks)
        self._set_seed()
        self._train_info_dir = self._create_train_info_directory()
        self.validation_loss = []
        self.epoch_train_loss = []
        self.batch_train_loss = []
        self.train_time_per_epoch = []
    
    def save_model(self, epoch: int, best: bool = False) -> None:
        pretrained_path = os.path.join(self.train_dir, "pretrained")
        os.makedirs(pretrained_path, exist_ok=True)
        filename = f"model_epoch_{epoch}.pth" if not best else "best_model.pth"
        filepath = os.path.join(pretrained_path, filename)
        torch.save(self.model.state_dict(), filepath)
        self._logger.info(f"Model saved at {filepath}")

    def load_model(self, model: nn.Module, filepath: str) -> None:
        model.load_state_dict(torch.load(filepath, weights_only=True, map_location=self.config.device))
        self._logger.info(f"Model loaded from {filepath}")
        return model

    def _initialize_model(self, model: nn.Module, pretrained_path: Optional[str | Path] = None) -> nn.Module:
        model = model.to(self.config.device)
        if pretrained_path is not None:
            model = self.load_model(model, pretrained_path)
        return model

    def _initialize_optimizer(self):
        return self.config.optimizer_type(self.model.parameters(), **self.config.optimizer_params)

    def _initialize_logger(self, logger_config: Optional[LoggerConfig]) -> Logger:
        if logger_config:
            logger_config.log_dir = os.path.join(self.train_dir, "logs")
        logger = Logger(logger_config)
        logger.info(f"Training directory created at: {self.train_dir}")
        return logger

    def _initialize_validation_dataloader(self, validation_dataloader: Optional[DataLoader]) -> Optional[DataLoader]:
        if validation_dataloader:
            if self.config.use_auto_validation:
                self._logger.warning("Validation dataloader provided; skipping auto-validation.")
            return validation_dataloader
        if self.config.use_auto_validation:
            return self._create_train_val_dataloaders()[1]
        self._logger.warning("No validation dataloader; consider enabling auto-validation.")
        return None

    def _initialize_hooks(self, hooks: Optional[List[Hook]]) -> HookManager:
        hook_manager = HookManager()
        if hooks:
            for hook in hooks:
                hook.set_output_dir(self.train_dir)
                hook_manager.add_hook(hook)
        return hook_manager

    def _set_seed(self):
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

    def _create_train_val_dataloaders(self, val_split: float = 0.2) -> Tuple[DataLoader, DataLoader]:
        dataset = self.train_dataloader.dataset
        total_len = len(dataset)
        val_len = int(total_len * val_split)
        train_len = total_len - val_len
        train_subset, val_subset = random_split(dataset, [train_len, val_len])
        batch_size = self.train_dataloader.batch_size
        train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True, num_workers=self.train_dataloader.num_workers, drop_last=True)
        val_loader = DataLoader(val_subset, batch_size=batch_size, shuffle=False, num_workers=self.train_dataloader.num_workers, drop_last=True)
        return train_loader, val_loader

    def __base_train_step(self, batch, batch_idx):
        inputs, targets = batch
        inputs = inputs.to(self.config.device)
        targets = targets.to(self.config.device)
        outputs = self.model(inputs)
        loss = self.config.criterion(outputs, targets)
        return loss

    def __base_validation_step(self, batch, batch_idx):
        inputs, targets = batch
        inputs = inputs.to(self.config.device)
        targets = targets.to(self.config.device)
        outputs = self.model(inputs)
        loss = self.config.criterion(outputs, targets)
        return loss

    def validate(self) -> float:
        self.model.eval()
        running_loss = 0.0
        with torch.inference_mode():
            for batch_idx, batch in enumerate(self.validation_dataloader):
                loss = self.model.validation_step(batch, self.config.device, self.config.criterion, batch_idx) if self._use_custom_validation_step else self.__base_validation_step(batch, batch_idx)
                running_loss += loss.item()
        return running_loss / len(self.validation_dataloader.dataset)

    def train(self) -> None:
        self._logger.info(f"Starting training loop for {self.model._get_name()}")
        self.hook_manager.execute("on_train_start", trainer=self)
        for epoch in range(1, self.config.epochs + 1):
            self._train_epoch(epoch)
        self._logger.info("Training complete")
        self.hook_manager.execute("on_train_end", trainer=self)

    def _train_epoch(self, epoch: int):
        self.hook_manager.execute("on_epoch_start", trainer=self, epoch=epoch)
        self.model.train()
        running_loss = 0.0
        start_time = time.time()
        with tqdm(total=len(self.train_dataloader), desc=f"Epoch [{epoch}/{self.config.epochs}]", unit="batch") as pbar:
            for batch_idx, batch in enumerate(self.train_dataloader):
                running_loss += self._train_batch(batch, batch_idx, epoch)
                pbar.update(1)
        self._finalize_epoch(running_loss, start_time, epoch)

    def _train_batch(self, batch, batch_idx: int, epoch: int) -> float:
        self.hook_manager.execute("on_batch_start", trainer=self, epoch=epoch, batch_idx=batch_idx, batch=batch)
        self.optimizer.zero_grad()
        loss = self.model.train_step(batch, self.config.device, self.config.criterion, batch_idx) if self.use_custom_train_step else self.__base_train_step(batch, batch_idx)
        loss.backward()
        self.optimizer.step()
        self.batch_train_loss.append(loss.item())
        self.hook_manager.execute("on_batch_end", trainer=self, epoch=epoch, batch_idx=batch_idx, loss=loss.item())
        return loss.item()

    def _finalize_epoch(self, running_loss: float, start_time: float, epoch: int):
        epoch_loss = running_loss / len(self.train_dataloader.dataset)
        self.epoch_train_loss.append(epoch_loss)
        epoch_time = time.time() - start_time
        self.train_time_per_epoch.append(epoch_time)
        self._logger.info(f"Epoch [{epoch}/{self.config.epochs}], Loss: {epoch_loss:.4f}, Time: {epoch_time:.2f}s")
        if self.use_validation:
            val_loss = self.validate()
            self.validation_loss.append(val_loss)
            self._logger.info(f"Validation Loss: {val_loss:.4f}")
        self.hook_manager.execute("on_epoch_end", trainer=self, epoch=epoch, epoch_loss=epoch_loss)
        if epoch % self.config.save_weights_per_epoch == 0 and epoch > 0:
            self.save_model(epoch, best=False)

    def get_history(self) -> TrainHistory:
        return TrainHistory(
            self.epoch_train_loss, 
            self.batch_train_loss, 
            self.validation_loss, 
            self.train_time_per_epoch
        )
