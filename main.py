import torch
from torch import nn, optim
from torch.utils.data import DataLoader, TensorDataset

from src.torch_trainer.logger import LoggerConfig
from src.torch_trainer.trainer import Trainer, TrainerConfig
from src.torch_trainer.hooks import TensorBoardHook

def create_dummy_dataset(samples=1000):
    x = torch.rand(samples, 10)
    y = torch.randint(0, 2, (samples,))
    dataset = TensorDataset(x, y)
    return DataLoader(dataset, batch_size=32, shuffle=True)

class SimpleModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(10, 50),
            nn.ReLU(),
            nn.Linear(50, 2)
        )

    def forward(self, x):
        return self.fc(x)
    
    def train_step(self, batch, device, criterion, batch_idx):
        inputs, targets = batch
        inputs = inputs.to(device)
        targets = targets.to(device)

        outputs = self(inputs)
        loss = criterion(outputs, targets)
        return loss
    
    def validation_step(self, batch, device, criterion, batch_idx):
        inputs, targets = batch
        inputs = inputs.to(device)
        targets = targets.to(device)

        outputs = self(inputs)
        loss = criterion(outputs, targets)
        return loss

def main() -> None:
    train_loader = create_dummy_dataset()
    val_loader = create_dummy_dataset(samples=200)

    train_config = TrainerConfig(
        train_name='Net',
        epochs=5,
        criterion=nn.CrossEntropyLoss(),
        optimizer_type=optim.Adam,
        optimizer_params={"lr": 0.001},
        use_auto_validation=True,
        device='auto',
        save_weights_per_epoch=2
    )

    logger_config = LoggerConfig(
        file_name="training.log",
        format=">>> [%(asctime)s] %(module)s:%(lineno)d - [%(levelname)s] - %(message)s",
    )

    model = SimpleModel()
    trainer = Trainer(
        model=model,
        config=train_config,
        train_dataloader=train_loader,
        validation_dataloader=None,
        logger_config=logger_config,
        hooks=[TensorBoardHook()],
        pretrained_path='trainer_info/Net/pretrained/model_epoch_4.pth'
    )
    trainer.train()

    print(trainer.get_history().epoch_train_loss)


if __name__ == '__main__':
    main()
