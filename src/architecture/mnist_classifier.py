import torch


class Classifier(torch.nn.Module):
    def __init__(self, hidden_dim: int = 128) -> None:
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(28 * 28, hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_dim, 10),
            torch.nn.ReLU()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.view(x.size(0), -1)
        return self.net(x)
