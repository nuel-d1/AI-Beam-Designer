from torch import nn


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()

        self.predictor = nn.Sequential(
            nn.Linear(in_features=9, out_features=9),
            nn.Linear(in_features=9, out_features=50),
            nn.Sigmoid(),
            nn.Linear(in_features=50, out_features=30),
            nn.Sigmoid(),
            nn.Linear(in_features=30, out_features=5))
            # nn.Sigmoid())

    def forward(self, x):
        """forward pass"""
        x = self.predictor(x)

        return x
