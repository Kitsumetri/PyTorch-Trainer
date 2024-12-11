import torch
from torch import nn, optim
from typing import Any, Dict, Type, Optional, Literal
import random
import numpy as np
import logging


class TrainerConfig:
    def __init__(
        self,
        train_name: str,
        epochs: int,
        criterion: Type[nn.Module],
        optimizer_type: Type[optim.Optimizer],
        use_auto_validation: bool = False,
        device: Literal['auto', 'cuda', 'mps', 'cpu'] = 'cpu',
        optimizer_params: Optional[Dict[str, Any]] = None,
        console_log: bool = True,
        seed: Optional[int] = None,
        output_dir: str = "./trainer_info",
        save_weights_per_epoch: int = 10
    ) -> None:
        self.train_name = train_name
        self.epochs = epochs
        self.criterion = criterion
        self.use_auto_validation = use_auto_validation
        self.optimizer_type = optimizer_type
        self.device = self._get_device_auto() if device == 'auto' else torch.device(device)
        self.optimizer_params = optimizer_params or {}
        self.console_log = console_log
        self.seed = seed
        self.output_dir = output_dir
        self.save_weights_per_epoch = save_weights_per_epoch
    
    def _get_device_auto(self) -> torch.device:
        device_name = None
        if torch.cuda.is_available():
            device_name = 'cuda'
        elif torch.mps.is_available():
            device_name = 'mps'
        else:
            device_name = 'cpu'
        
        return torch.device(device_name)
            

class LoggerConfig:
    DEFAULT_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"

    def __init__(
        self,
        file_name: str = "training.log",
        level: int = logging.INFO,
        format: Optional[str] = None) -> None:

        if not file_name or not file_name.strip():
            raise ValueError("`file_name` must be a valid non-empty string.")

        self.file_name = file_name
        self.level = level
        self.format = format if format is not None else self.DEFAULT_FORMAT


def set_random_seed(seed: Optional[int]) -> None:
    """
    Set random seed for reproducibility.

    Args:
        seed (Optional[int]): Random seed value. If None, no seed is set.
    """
    if seed is not None:
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        random.seed(seed)
        np.random.seed(seed)
