from typing import Any, List, Optional
from torch.utils.tensorboard import SummaryWriter
import os


class Hook:
    """Base class for hooks."""

    def __init__(self):
        self.output_dir: Optional[str] = None

    def set_output_dir(self, output_dir: str) -> None:
        """Set the output directory for the hook."""
        self.output_dir = output_dir

    def on_train_start(self, trainer: Any, **kwargs) -> None:
        """Called at the start of training."""
        pass

    def on_epoch_start(self, trainer: Any, epoch: int, **kwargs) -> None:
        """Called at the start of an epoch."""
        pass

    def on_batch_start(
        self, trainer: Any, epoch: int, batch_idx: int, **kwargs
    ) -> None:
        """Called at the start of a batch."""
        pass

    def on_batch_end(self, trainer: Any, epoch: int, batch_idx: int, **kwargs) -> None:
        """Called at the end of a batch."""
        pass

    def on_epoch_end(
        self, trainer: Any, epoch: int, epoch_loss: float, **kwargs
    ) -> None:
        """Called at the end of an epoch."""
        pass

    def on_train_end(self, trainer: Any, **kwargs) -> None:
        """Called at the end of training."""
        pass


class TensorBoardHook(Hook):
    """Hook for logging metrics to TensorBoard."""

    def on_train_start(self, trainer, **kwargs) -> None:
        """Initialize the TensorBoard writer."""
        if self.output_dir is None:
            raise ValueError("Output directory must be set before training starts.")

        log_dir = os.path.join(self.output_dir, "logs")
        os.makedirs(log_dir, exist_ok=True)
        self.writer = SummaryWriter(log_dir=log_dir)
        trainer._logger.info(
            f"TensorBoard initialized. Logs will be saved to {log_dir}"
        )

    def on_epoch_end(self, trainer, epoch: int, epoch_loss: float, **kwargs) -> None:
        """Log loss and other metrics at the end of the epoch."""
        if self.writer:
            self.writer.add_scalar("Loss/train", epoch_loss, epoch)
            trainer._logger.info(
                f"Logged epoch {epoch} loss to TensorBoard: {epoch_loss:.4f}"
            )

    def on_batch_end(
        self, trainer, epoch: int, batch_idx: int, loss: float, **kwargs
    ) -> None:
        """Log batch-level loss."""
        if self.writer:
            global_step = epoch * len(trainer.train_dataloader) + batch_idx
            self.writer.add_scalar("Loss/batch", loss, global_step)

    def on_train_end(self, trainer, **kwargs) -> None:
        """Close the TensorBoard writer."""
        if self.writer:
            self.writer.close()
            trainer._logger.info("TensorBoard writer closed.")


class HookManager:
    """Manages hooks for training events."""

    def __init__(self) -> None:
        self.hooks: List[Hook] = []

    def add_hook(self, hook: Hook) -> None:
        """Add a hook to the manager."""
        if not isinstance(hook, Hook):
            raise TypeError("Only instances of Hook can be added.")
        self.hooks.append(hook)

    def execute(self, method_name: str, **kwargs) -> None:
        """Execute a specific hook method."""
        for hook in self.hooks:
            if hasattr(hook, method_name):
                getattr(hook, method_name)(**kwargs)
