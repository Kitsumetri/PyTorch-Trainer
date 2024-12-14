# PyTorch Trainer

Simple module for comfortable fitting pytorch models. Reference to HF Trainer and PytorchLighting.

## Example
```py
from torch_trainer.trainer import Trainer
from torch_trainer.config import TrainerConfig, LoggerConfig
from torch_trainer.hooks import TensorBoardHook

log_config = LoggerConfig(
    file_name='train.log',
    file_log=True,
    console_log=True
)

train_config = TrainerConfig(
    train_name='cifar_classifier',
    epochs=20,
    criterion=nn.CrossEntropyLoss(),
    optimizer_type=torch.optim.Adam,
    do_model_compile=True,
    use_amp=True,
    use_auto_validation=True,
    optimizer_params={"lr": 1e-3},
    device='auto',
    output_dir="./trainer_info",
    seed=42
)
net = Net()
trainer = Trainer(
    model=net,
    config=train_config,
    logger_config=log_config,
    train_dataloader=trainloader,
    hooks=[TensorBoardHook()]
)

trainer.train()
```

## Documentation

### Trainer
```py
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
)
```
* model - pytorch model (subclass on nn.Module) with forward(x) method.

-----------------------------------------------------------------------------------------------
**Consider writing methods in your model class**:

```train_step(batch: Any, device: str | torch.device, criterion: nn.Module, batch_idx: int) -> pytorch loss from criterion```
Args:
1) ```batch``` - unpacked batch from train dataloader.
2) ```device``` - device for training.
3) ```criterion``` - loss function.
4) ```batch_idx``` - number of batch in train dataloader.

Default train_step if model doesn't have ```train_step()``` method:
```py
def __base_train_step(self, batch, batch_idx: int) -> float:
        inputs, targets = batch
        inputs = inputs.to(self.config.device)
        targets = targets.to(self.config.device)
        with torch.autocast(device_type=self.config.device.type, enabled=self.config.use_amp):
            outputs = self.model(inputs)
            loss = self.config.criterion(outputs, targets)
        return loss
```
-----------------------------------------------------------------------------------------------

```validation_step(batch: Any, device: str | torch.device, criterion: nn.Module, batch_idx: int) -> pytorch loss from criterion```
Args:
1) ```batch``` - unpacked batch from train dataloader.
2) ```device``` - device for training.
3) ```criterion``` - loss function.
4) ```batch_idx``` - number of batch in train dataloader.

Default validation_step if model doesn't have ```validation_step()``` method:
```py
def __base_validation_step(self, batch, batch_idx: int) -> float:
        inputs, targets = batch
        inputs = inputs.to(self.config.device)
        targets = targets.to(self.config.device)
        with torch.autocast(device_type=self.config.device.type, enabled=self.config.use_amp):
            outputs = self.model(inputs)
            loss = self.config.criterion(outputs, targets)
        return loss
```
-----------------------------------------------------------------------------------------------
* ```config``` - TrainConfig object for training, validation and etc (will see below).
* ```train_dataloader``` - train dataloader from pytorch.
* ```validation_dataloader``` - (default=None). validation dataloader from pytorch. Trainer support auto_validation from train_loader (will see below in train config).
If no validation is provided and don't use auto validation flag in config then validation loop will skip.
* ```logger_config``` - (default=None). LoggerConfig object. If None, then default logger will be created.
* ```hooks``` - (default=None). List of Hooks. Hooks will be execute at different checkpoints while training/validation (see more below).
* ```pretrained_path``` - (default=None). Path to pretrained weights of the model. If None, then model won't be loaded  from any checkpoint.
### Trainer Config
```py
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
        device: Literal['auto', 'cuda', 'mps', 'cpu'] = 'auto',
        optimizer_params: Optional[Dict[str, Any]] = None,
        seed: Optional[int] = None,
        output_dir: str = "./trainer_info",
        save_weights_per_epoch: int = 10,
   )
```
* ```train_name``` - name of running training. All training info will be save at: ```cwd/{output_dir}/{train_name}```.
* ```epochs```: number of epochs.
* ```criterion``` - pytorch loss function / sublcass of nn.Module for custom loss function.
* ```optimizer_type``` - pytorch class (not object!) of optimizer / subclass of optim.Optimizer for custom optimizer.
* ```do_model_compile``` - (default=True) bool flag for compiling model before training.
* ```use_amp``` - (default=True). enable torch.amp features while training. This also has an impact on turning on GradScaler().
* ```use_auto_validation```- (default=False). Enable auto validation from train set. If validation loader is provided and use_auto_validation=True, then nothing will happend. Auto validation step will be skipped.
* ```device``` - (default='auto'). Device for training, if auto is provided, then device will choose the most available device. Options: 'cpu', 'mps', 'cuda'.
*  ```optimizer_params``` - (default=None}. Dict with params for optimizer.
*  ```seed``` - (default=None). Set random seed for numpy, random and torch.cuda. If None, then will use default seed.
*  ```output_dir``` - (default='/trainer_info'). Path to main folder with all data and information for all runs.
*  ```save_weights_per_epoch``` - (default=10). Saving model after provided num of epochs, like every 10, 20 and etc finished epochs.
### LoggerConfig
```py
class LoggerConfig:
    DEFAULT_FORMAT = ">>> [%(asctime)s] %(module)s:%(lineno)d - %(levelname)s - %(message)s"

    def __init__(
        self,
        file_name: Optional[str] = None,
        level: int = logging.INFO,
        console_log: bool = True,
        file_log: bool = True,
        format: Optional[str] = None) -> None:
)
```
* ```file_name```- (default=None). ```file_name.log``` for saving logs in file, do not forgot to use ```file_log=True```. Log file will be saved at ```cwd/{output_path}/{train_name}/log/```.
* ```level``` - (default=logging.INFO). Level of logging. Reference to build-in logging module in python.
* ```console_log``` - (default=True). Flag for printing colored logs in stdout stream.
* ```file_log``` - (default=True). Flag for writing logs in file, use with file_name arg.
* ```format``` - (default=None). Custom format for logger. Reference to logging format. If None, then DEFAULT_FORMAT will use.

### Hooks

Work in progress...

#### Example TensorboardHook:
```py
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
```
