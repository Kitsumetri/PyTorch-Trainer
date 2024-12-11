from typing import Optional, List, Tuple, Callable

from tqdm import tqdm
import os
import time

import torch
from torch import nn
from torch.utils.data import DataLoader, random_split
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

        self.model: nn.Module = model.to(self.config.device)
        self.optimizer = self.config.optimizer_type(
            model.parameters(), **self.config.optimizer_params
        )

        self.use_custom_train_step = True
        if not hasattr(model, 'train_step'):
            self.use_custom_train_step = False
            self._logger.warning("Consider writing trian_step(self, batch, device, criterion, batch_idx) custom method in model class")
            self._logger.warning("Using base train_step method in train loop. It can cause errors...")

        if logger_config:
            logger_config.log_dir = os.path.join(self.train_dir, "logs")

        self._logger = Logger(logger_config)
        self._logger.info(f"Training directory created at: {self.train_dir}")

        self.train_dataloader = train_dataloader
        self.use_validation = True

        if validation_dataloader is not None:
            self.validation_dataloader = validation_dataloader
            if self.config.use_auto_validation:
                self._logger.warning("Found use_auto_validation=True, but validation dataloader was provided. Skip auto validation...")
        elif self.config.use_auto_validation:
            self.train_dataloader, self.validation_dataloader = self._create_train_val_dataloaders(self.train_dataloader)
            self._logger.info('Validation datset was successfully created due to use_auto_validation=True')
        else:
            self.use_validation = False
            self._logger.warning(
                "Validation dataloader wasn't provided & use_auto_validation=False in config. Consider using auto validation.")

        self._use_custom_validation_step = False
        if self.use_validation:
            if not hasattr(model, 'validation_step'):
                self._logger.warning("Consider writing validation_step(self, batch, device, criterion, batch_idx) custom method in model class")
                self._logger.warning("Using base validation_step method. It can cause errors...")
            else:
                self._use_custom_validation_step = True


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

    @staticmethod
    def _create_train_directory(config: TrainerConfig) -> str:
        """Create directory for training session files."""
        train_dir = os.path.join(config.output_dir, config.train_name)
        os.makedirs(train_dir, exist_ok=True)
        return train_dir

    @staticmethod
    def _create_train_val_dataloaders(
        train_dataloader: DataLoader, 
        val_split: float = 0.2) -> Tuple[DataLoader, DataLoader]:

        if not 0 < val_split < 1:
            raise ValueError("val_split must be a float between 0 and 1.")

        dataset = train_dataloader.dataset
        total_len = len(dataset)
        val_len = int(total_len * val_split)
        train_len = total_len - val_len

        train_subset, val_subset = random_split(dataset, [train_len, val_len])

        batch_size = train_dataloader.batch_size
        train_loader = DataLoader(
            train_subset, batch_size=batch_size, shuffle=True, num_workers=train_dataloader.num_workers,
            drop_last=True
        )
        val_loader = DataLoader(
            val_subset, batch_size=batch_size, shuffle=False, num_workers=train_dataloader.num_workers,
            drop_last=True
        )

        return train_loader, val_loader

    def validate(self) -> float:
        self.model.eval()
        running_loss = 0.0
        with torch.inference_mode():
            for batch_idx, batch in enumerate(self.validation_dataloader):
                if self._use_custom_validation_step:
                    loss = self.model.validation_step(batch, self.config.device, self.config.criterion, batch_idx)
                else:
                    loss = self.__base_validation_step(batch, batch_idx)
                running_loss += loss.item() * batch[0].size(0)
        return running_loss / len(self.validation_dataloader.dataset)

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
                for batch_idx, batch in enumerate(self.train_dataloader):
                    self.hook_manager.execute(
                        "on_batch_start",
                        trainer=self,
                        epoch=epoch,
                        batch_idx=batch_idx,
                        batch=batch
                    )
                    self.optimizer.zero_grad()
                    if self.use_custom_train_step:
                        loss = self.model.train_step(batch, self.config.device, self.config.criterion, batch_idx)
                    else:
                        loss = self.__base_train_step(batch, batch_idx)
                    loss.backward()
                    self.optimizer.step()

                    running_loss += loss.item()  # FIXME
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
                val_loss = self.validate()
                self._logger.info(f"Validation Loss: {val_loss:.4f}")

            self.hook_manager.execute(
                "on_epoch_end",
                trainer=self,
                epoch=epoch,
                epoch_loss=epoch_loss,
            )

        self._logger.info("Training complete")
        self.hook_manager.execute("on_train_end", trainer=self)
