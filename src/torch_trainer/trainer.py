from torch.utils.tensorboard import SummaryWriter

from typing import (
    Optional, Any, Union, Literal,
    Dict, List, Tuple, Type
)

import logging
import os
import sys

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
                 console_log: bool = True,
                 file_log: bool = False) -> None:
        self.epochs = epochs
        self.criterion = criterion
        self.optimizer_type = optimizer_type
        self.device = device
        self.console_log = console_log
        self.file_log = file_log

class Trainer:
    def __init__(self,
                 model: Type[nn.Module],
                 config: TrainConfig,
                 train_dataloader: DataLoader,
                 validation_dataloader: Optional[DataLoader] = None) -> None:
        
        self.config = config
        self.train_dataloader = train_dataloader

        # FIXME add checking module class for forward
        self.model: nn.Module = model.to(self.config.device)

        self.logger: logging.Logger = self._setup_logger()
        self.logger.info('Logger setup was successfully')

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
            train_info_dir = os.path.join(ROOT_DIR, 'trainer_info')
            os.makedirs(name=train_info_dir, exist_ok=True)

            file_handler = logging.FileHandler(
                filename=os.path.join(train_info_dir, f"{__name__}.log"), # FIXME Add model name + date
                mode='w'
            )
            file_handler.setFormatter(logger_formatter)
            logger.addHandler(file_handler)
        return logger

    def train() -> None:
        pass

    def __repr__(self) -> str:
        pass

    def __str__(self) -> str:
        pass

