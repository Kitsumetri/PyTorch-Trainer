from typing import Optional, List

from tqdm import tqdm
import os
import time

import torch
from torch import nn
from torch.utils.data import DataLoader
from .config import TrainerConfig, set_random_seed
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
    ) -> None:
        self.config = config
        self.train_dir = self._create_train_directory(config)

        self.train_dataloader = train_dataloader
        self.validation_dataloader = validation_dataloader
        self.use_validation = validation_dataloader is not None

        self.model: nn.Module = model.to(self.config.device)
        self.optimizer = self.config.optimizer_type(
            model.parameters(), **self.config.optimizer_params
        )

        if logger_config:
            logger_config.log_dir = os.path.join(self.train_dir, "logs")

        self._logger = Logger(logger_config)
        self._logger.info(f"Training directory created at: {self.train_dir}")

        self.hook_manager = HookManager()
        if hooks:
            for hook in hooks:
                hook.set_output_dir(self.train_dir)
                self.hook_manager.add_hook(hook)

        if self.config.seed is not None:
            self._logger.info(f"Setting random seed: {self.config.seed}")
            set_random_seed(self.config.seed)

        self._train_info_dir = os.path.join("trainer_info")
        self.model_name = model._get_name()

        os.makedirs(self._train_info_dir, exist_ok=True)
        self._logger.info(f"Training info directory created: {self._train_info_dir}")

    def _create_train_directory(self, config: TrainerConfig) -> str:
        """Create directory for training session files."""
        train_dir = os.path.join(config.output_dir, config.train_name)
        os.makedirs(train_dir, exist_ok=True)
        return train_dir

    @staticmethod
    def evaluate(
        model: nn.Module,
        data_loader: DataLoader,
        criterion: nn.Module,
        device: torch.device,
    ) -> float:
        model.eval()
        running_loss = 0.0
        with torch.inference_mode():
            for inputs, labels in data_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                running_loss += loss.item() * inputs.size(0)
        return running_loss / len(data_loader.dataset)

    def train(self) -> None:
        """Main training loop."""
        self._logger.info(f"Starting training loop for {self.model_name}")
        self.hook_manager.execute("on_train_start", trainer=self)

        for epoch in range(1, self.config.epochs + 1):
            self.hook_manager.execute("on_epoch_start", trainer=self, epoch=epoch)

            self.model.train()
            running_loss = 0.0
            start_time = time.time()

            with tqdm(
                total=len(self.train_dataloader),
                desc=f"Epoch [{epoch}/{self.config.epochs}]",
                unit="batch",
            ) as pbar:
                for batch_idx, (inputs, labels) in enumerate(self.train_dataloader):
                    self.hook_manager.execute(
                        "on_batch_start",
                        trainer=self,
                        epoch=epoch,
                        batch_idx=batch_idx,
                        inputs=inputs,
                        labels=labels,
                    )

                    inputs, labels = (
                        inputs.to(self.config.device),
                        labels.to(self.config.device),
                    )
                    self.optimizer.zero_grad()
                    outputs = self.model(inputs)
                    loss = self.config.criterion(outputs, labels)
                    loss.backward()
                    self.optimizer.step()

                    running_loss += loss.item() * inputs.size(0)
                    pbar.set_postfix({"Loss": loss.item()})
                    pbar.update(1)

                    self.hook_manager.execute(
                        "on_batch_end",
                        trainer=self,
                        epoch=epoch,
                        batch_idx=batch_idx,
                        loss=loss.item(),
                    )

            epoch_loss = running_loss / len(self.train_dataloader.dataset)
            epoch_time = time.time() - start_time
            self._logger.info(
                f"Epoch [{epoch}/{self.config.epochs}], Loss: {epoch_loss:.4f}, Time: {epoch_time:.2f}s"
            )

            if self.use_validation:
                val_loss = self.evaluate(
                    self.model,
                    self.validation_dataloader,
                    self.config.criterion,
                    self.config.device,
                )
                self._logger.info(f"Validation Loss: {val_loss:.4f}")

            self.hook_manager.execute(
                "on_epoch_end",
                trainer=self,
                epoch=epoch,
                epoch_loss=epoch_loss,
            )

        self._logger.info("Training complete")
        self.hook_manager.execute("on_train_end", trainer=self)
