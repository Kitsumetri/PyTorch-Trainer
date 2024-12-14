import torch
from torch import nn, optim
from typing import Any, Dict, Type, Optional, Literal, List
import random
import numpy as np
import logging
from dataclasses import dataclass, field


class TrainerConfig:
    def __init__(
        self,
        train_name: str,
        epochs: int,
        criterion: Type[nn.Module],
        optimizer_type: Type[optim.Optimizer],
        do_model_compile: bool = True,
        use_amp: bool = True,
        use_auto_validation: bool = False,
        device: Literal["auto", "cuda", "mps", "cpu"] = "auto",
        optimizer_params: Optional[Dict[str, Any]] = None,
        seed: Optional[int] = None,
        output_dir: str = "./trainer_info",
        save_weights_per_epoch: int = 10,
    ) -> None:
        self.do_model_compile = do_model_compile
        self.train_name = train_name
        self.epochs = epochs
        self.criterion = criterion
        self.use_auto_validation = use_auto_validation
        self.optimizer_type = optimizer_type
        self.device = (
            self._get_device_auto() if device == "auto" else torch.device(device)
        )
        self.optimizer_params = optimizer_params or {}
        self.seed = seed
        self.output_dir = output_dir
        self.save_weights_per_epoch = save_weights_per_epoch
        self.use_amp = use_amp

    def _get_device_auto(self) -> torch.device:
        device_name = None
        if torch.cuda.is_available():
            device_name = "cuda"
        elif torch.mps.is_available():
            device_name = "mps"
        else:
            device_name = "cpu"

        return torch.device(device_name)


class LoggerConfig:
    DEFAULT_FORMAT = (
        ">>> [%(asctime)s] %(module)s:%(lineno)d - %(levelname)s - %(message)s"
    )

    def __init__(
        self,
        file_name: Optional[str] = None,
        level: int = logging.INFO,
        console_log: bool = True,
        file_log: bool = True,
        format: Optional[str] = None,
    ) -> None:
        if file_log and (file_name is None or file_name.strip() is None):
            raise ValueError("Prodived file_log=True, while file_name=None")

        if file_name is None:
            file_name = "train.log"

        self.file_name = file_name
        self.console_log = console_log
        self.file_log = file_log
        self.level = level
        self.format = format if format is not None else self.DEFAULT_FORMAT


@dataclass
class TrainHistory:
    epoch_train_loss: List[float] = field(default_factory=lambda x: [])
    batch_train_loss: List[float] = field(default_factory=lambda x: [])
    validation_loss: List[float] = field(default_factory=lambda x: [])
    train_time_per_epoch: List[int | float] = field(default_factory=lambda x: [])


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
