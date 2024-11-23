from torch_trainer.trainer import TrainConfig, Trainer
import torch
import torch.nn as nn


class FullyConnectedNN(nn.Module):
    def __init__(
        self, 
        input_size: int = 512, 
        num_classes: int = 10) -> None:

        super(FullyConnectedNN, self).__init__()
        self.fc_layers = nn.Sequential(
            nn.Linear(input_size, 256),
            nn.ReLU(),
            nn.Linear(256, num_classes)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc_layers(x)

def main() -> None:
    config = TrainConfig(
        epochs=20,
        criterion=torch.nn.CrossEntropyLoss(),
        optimizer_type=torch.optim.Adam,
        device=torch.device('cpu'),
        console_log=True,
        file_log=True
    )


    trainer = Trainer(FullyConnectedNN(), config=config)

if __name__ == '__main__':
    main()
