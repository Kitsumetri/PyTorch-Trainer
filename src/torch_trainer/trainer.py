from torch.utils.tensorboard import SummaryWriter

from typing import (
    Optional, Any, Union, Literal,
    Dict, List, Tuple, Type
)

from tqdm import tqdm
import logging
import os
import sys
from datetime import datetime
import time

import torch
from torch import nn
from torch import optim
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import *   # FIXME

ROOT_DIR: str = os.getcwd()  # FIXME maybe another module for it

class TrainConfig:
    def __init__(self,
                 epochs: int,
                 criterion: Type[nn.Module],
                 optimizer_type: Type[optim.Optimizer],
                 device: torch.device = torch.device('cpu'),
                 optimizer_params: Dict[str, Any] = None,
                 console_log: bool = True,
                 file_log: bool = False) -> None:
        self.epochs = epochs
        self.criterion = criterion
        self.optimizer_type = optimizer_type
        self.device = device
        self.console_log = console_log
        self.file_log = file_log

        self.optimizer_params = {}  if optimizer_params is None else optimizer_params

class Trainer:
    def __init__(self,
                 model: Type[nn.Module],
                 config: TrainConfig,
                 train_dataloader: DataLoader,
                 validation_dataloader: Optional[DataLoader] = None) -> None:
        
        self.config = config
        self.train_dataloader = train_dataloader

        self.model: nn.Module = model.to(self.config.device) # FIXME add checking module class for forward
        self.optimizer = self.config.optimizer_type(model.parameters(), **self.config.optimizer_params)

        self._train_info_dir = os.path.join(ROOT_DIR, 'trainer_info')

        self._logger: logging.Logger = self._setup_logger()
        self._logger.info('Logger setup was successfully')
        self.model_name = model._get_name()

        self.use_validation = False  # FIXME add auto valid set generation for specifig flag in config
        if validation_dataloader is not None:
            self.validation_dataloader = validation_dataloader
            self.use_validation = True
    
    def _setup_logger(self) -> logging.Logger:
        logger = logging.getLogger(__name__)
        logger.setLevel(logging.INFO)

        logger_formatter = logging.Formatter(
            fmt='>>> [%(asctime)s] %(module)s:%(lineno)d - %(levelname)s - %(message)s',
             datefmt='%Y-%m-%d | %H:%M:%S'
        )

        if self.config.console_log:
            stream_handler = logging.StreamHandler(stream=sys.stdout)
            stream_handler.setFormatter(logger_formatter)
            logger.addHandler(stream_handler)
        
        if self.config.file_log:
            os.makedirs(name=self._train_info_dir, exist_ok=True)

             # FIXME Add model name + date for path
            file_handler = logging.FileHandler(
                filename=os.path.join(self._train_info_dir, f"{__name__}.log"),
                mode='w'
            )
            file_handler.setFormatter(logger_formatter)
            logger.addHandler(file_handler)
        return logger

    @staticmethod
    def evaluate(model: nn.Module, data_loader: DataLoader, criterion: nn.Module, device: torch.device) -> float:
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
        # FIXME save model weights
        # best_model_path = f"{self._train_info_dir}/pretrained/{self.model_name}/best_model.pth"

        writer = SummaryWriter(
            os.path.join(
                self._train_info_dir, 
                f'runs/{self.model_name}_{datetime.now().strftime("%d/%m/%Y_%H:%M:%S")}'
            )
        )

        self._logger.info(f'Starting training loop for {self.model_name}')

        for epoch in range(1, self.config.epochs + 1):
            self.model.train()
            running_loss = 0.0
            start_time = time.time()

            with tqdm(total=len(self.train_dataloader), desc=f"Epoch [{epoch}/{self.config.epochs}]", unit="batch") as pbar:
                for inputs, labels in self.train_dataloader:
                    inputs, labels = inputs.to(self.config.device), labels.to(self.config.device)
                    self.optimizer.zero_grad()

                    outputs = self.model(inputs)
                    loss = self.config.criterion(outputs, labels)
                    loss.backward()
                    self.optimizer.step()

                    running_loss += loss.item() * inputs.size(0)
                    pbar.set_postfix({'Loss': loss.item()})
                    pbar.update(1)

            epoch_loss = running_loss / len(self.train_dataloader.dataset)
            epoch_time = time.time() - start_time

            self._logger.info(f"Epoch [{epoch}/{self.config.epochs}], Loss: {epoch_loss:.4f}, Time: {epoch_time:.2f}s")
            writer.add_scalar('Loss/train', epoch_loss, epoch)

            if self.use_validation:
                val_loss = self.evaluate(
                    self.model, self.validation_dataloader, 
                    self.config.criterion, self.config.device
                )
                self._logger.info(f"Validation Loss: {val_loss:.4f}")
                writer.add_scalar('Loss/val', val_loss, epoch)

        self._logger.info("Training complete")
        writer.close()

