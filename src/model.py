import torch
from torch import nn

class MLP(nn.Module):
    def __init__(self):
        super(MLP, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(3, 512),
            nn.ReLU(),
            nn.BatchNorm1d(512),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.BatchNorm1d(128),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.BatchNorm1d(64),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.BatchNorm1d(32),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.BatchNorm1d(16),
            nn.Linear(16, 8),
            nn.ReLU(),
            nn.BatchNorm1d(8),
            nn.Linear(8, 4),
            nn.ReLU(),
            nn.BatchNorm1d(4),
            nn.Linear(4, 4),
            nn.ReLU(),
            nn.BatchNorm1d(4),
            nn.Linear(4, 4),
            nn.ReLU(),
            nn.BatchNorm1d(4),
            # nn.Linear(2048, 2048),
            # nn.ReLU(),
            # nn.BatchNorm1d(2048),
            # nn.Linear(2048, 2048),
            # nn.ReLU(),
            # nn.BatchNorm1d(2048),
            # nn.Linear(2048, 2048),
            # nn.ReLU(),
            # nn.BatchNorm1d(2048),
            # nn.Linear(2048, 2048),
            # nn.ReLU(),
            # nn.BatchNorm1d(2048),
            # nn.Linear(2048, 2048),
            # nn.ReLU(),
            # nn.BatchNorm1d(2048),
            # nn.BatchNorm1d(2048),
            # nn.Linear(2048, 2048),
            # nn.ReLU(),
            # nn.BatchNorm1d(2048),
            # nn.Linear(2048, 2048),
            # nn.ReLU(),
            # nn.BatchNorm1d(2048),
            # nn.Linear(2048, 2048),
            # nn.ReLU(),
            # nn.BatchNorm1d(2048),
            # nn.Linear(2048, 2048),
            # nn.ReLU(),
            # nn.BatchNorm1d(2048),
            # nn.Linear(2048, 2048),
            # nn.ReLU(),
            # nn.BatchNorm1d(2048),
            # nn.Linear(2048, 2048),
            # nn.ReLU(),
            # nn.BatchNorm1d(2048),
            # nn.Linear(2048, 2048),
            # nn.ReLU(),
            # nn.BatchNorm1d(2048),
            # nn.Linear(2048, 2048),
            # nn.ReLU(),
            # nn.BatchNorm1d(2048),
            # nn.Linear(2048, 2048),
            # nn.ReLU(),
            # nn.BatchNorm1d(2048),
            # nn.Linear(2048, 2048),
            # nn.ReLU(),
            # nn.BatchNorm1d(2048),
            # nn.Linear(2048, 2048),
            # nn.ReLU(),
            # nn.BatchNorm1d(2048),
            # nn.Linear(2048, 2048),
            # nn.ReLU(),
            # nn.BatchNorm1d(2048),
            # nn.Linear(2048, 2048),
            # nn.ReLU(),
            # nn.BatchNorm1d(2048),
            # nn.Linear(2048, 2048),
            # nn.ReLU(),
            # nn.BatchNorm1d(2048),
            # nn.Linear(2048, 2048),
            # nn.ReLU(),
            # nn.BatchNorm1d(2048),
            # nn.Linear(2048, 2048),
            # nn.ReLU(),
            # nn.BatchNorm1d(2048),
            # nn.Linear(2048, 2048),
            # nn.ReLU(),
            # nn.BatchNorm1d(2048),
            # nn.Linear(2048, 2048),
            # nn.ReLU(),
            # nn.BatchNorm1d(2048),
            nn.Linear(4, 1)
            # 最後是一個數字
        )

    def forward(self, x):
        y_hat = self.model(x)
        return y_hat 