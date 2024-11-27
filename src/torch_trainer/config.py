import torch
from torch import nn, optim
from typing import Any, Dict, Type, Optional
import yaml
import logging
import json


class TrainerConfig:
    """
    Configuration for training process.

    Attributes:
        epochs (int): Number of training epochs.
        criterion (Type[nn.Module]): Loss function class (e.g., nn.CrossEntropyLoss).
        optimizer_type (Type[optim.Optimizer]): Optimizer class (e.g., optim.Adam).
        device (torch.device): Device to use for training (default: 'cpu').
        optimizer_params (Dict[str, Any]): Parameters for optimizer initialization.
        console_log (bool): Enable console logging (default: True).
        file_log (bool): Enable file logging (default: False).
        seed (Optional[int]): Random seed for reproducibility (default: None).
    """

    def __init__(
        self,
        train_name: str,
        epochs: int,
        criterion: Type[nn.Module],
        optimizer_type: Type[optim.Optimizer],
        device: torch.device = torch.device("cpu"),
        optimizer_params: Optional[Dict[str, Any]] = None,
        console_log: bool = True,
        file_log: bool = False,
        seed: Optional[int] = None,
        output_dir: str = "./trainer_info",
    ) -> None:
        self.train_name = train_name
        self.epochs = epochs
        self.criterion = criterion
        self.optimizer_type = optimizer_type
        self.device = device
        self.optimizer_params = optimizer_params or {}
        self.console_log = console_log
        self.file_log = file_log
        self.seed = seed
        self.output_dir = output_dir

    @staticmethod
    def from_dict(config: Dict[str, Any]) -> "TrainerConfig":
        """
        Create a TrainerConfig instance from a dictionary.

        Args:
            config (Dict[str, Any]): Configuration dictionary.

        Returns:
            TrainerConfig: Initialized configuration object.
        """
        try:
            return TrainerConfig(
                epochs=config["epochs"],
                criterion=config["criterion"],
                optimizer_type=config["optimizer_type"],
                device=torch.device(config.get("device", "cpu")),
                optimizer_params=config.get("optimizer_params", {}),
                console_log=config.get("console_log", True),
                file_log=config.get("file_log", False),
                seed=config.get("seed"),
            )
        except KeyError as e:
            raise ValueError(f"Missing required configuration key: {e}")

    @staticmethod
    def from_yaml(file_path: str) -> "TrainerConfig":
        """
        Load TrainerConfig from a YAML file.

        Args:
            file_path (str): Path to the YAML file.

        Returns:
            TrainerConfig: Initialized configuration object.
        """
        try:
            with open(file_path, "r") as f:
                config = yaml.safe_load(f)
            return TrainerConfig.from_dict(config)
        except Exception as e:
            raise ValueError(f"Error loading YAML file {file_path}: {e}")

    @staticmethod
    def from_json(file_path: str) -> "TrainerConfig":
        """
        Load TrainerConfig from a JSON file.

        Args:
            file_path (str): Path to the JSON file.

        Returns:
            TrainerConfig: Initialized configuration object.
        """
        try:
            with open(file_path, "r") as f:
                config = json.load(f)
            return TrainerConfig.from_dict(config)
        except Exception as e:
            raise ValueError(f"Error loading JSON file {file_path}: {e}")

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert TrainerConfig to a dictionary.

        Returns:
            Dict[str, Any]: Configuration as a dictionary.
        """
        return {
            "epochs": self.epochs,
            "criterion": self.criterion,
            "optimizer_type": self.optimizer_type,
            "device": str(self.device),
            "optimizer_params": self.optimizer_params,
            "console_log": self.console_log,
            "file_log": self.file_log,
            "seed": self.seed,
        }

    def to_yaml(self, file_path: str) -> None:
        """
        Save TrainerConfig to a YAML file.

        Args:
            file_path (str): Path to save the YAML file.
        """
        try:
            with open(file_path, "w") as f:
                yaml.dump(self.to_dict(), f)
        except Exception as e:
            raise ValueError(f"Error saving YAML file {file_path}: {e}")

    def to_json(self, file_path: str) -> None:
        """
        Save TrainerConfig to a JSON file.

        Args:
            file_path (str): Path to save the JSON file.
        """
        try:
            with open(file_path, "w") as f:
                json.dump(self.to_dict(), f, indent=4)
        except Exception as e:
            raise ValueError(f"Error saving JSON file {file_path}: {e}")


class LoggerConfig:
    """Configuration for the logging system.

    Args:
        log_dir (str): Directory where logs will be saved.
        file_name (str): Name of the log file.
        level (int): Logging level. Defaults to logging.INFO.
        format (str): Log message format. Defaults to a standard format with timestamps.
    """

    DEFAULT_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"  # Default format for log messages

    def __init__(
        self,
        file_name: str = "training.log",
        level: int = logging.INFO,
        format: Optional[str] = None,
    ) -> None:
        # Validate file name
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
        import random

        random.seed(seed)
        import numpy as np

        np.random.seed(seed)
