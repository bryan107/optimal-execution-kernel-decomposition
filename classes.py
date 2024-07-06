import torch
import torch.nn as nn 
from torch.functional import F

class MLP(nn.Module):

    def __init__(self, decomp_dim: int):
        super(MLP, self).__init__()

        self.fc1 = nn.Linear(1, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, decomp_dim)

        self._log_sigma_base = torch.log(torch.FloatTensor([1]))
        # self._log_sigma = self._log_sigma_base
        self._log_sigma = nn.Parameter(data=self._log_sigma_base+torch.log(torch.rand(1)))

        self.price_impact_kappa_base = torch.FloatTensor([1])
        # self.price_impact_kappa = self.price_impact_kappa_base
        self.price_impact_kappa = nn.Parameter(data=self.price_impact_kappa_base*torch.rand(1))

    def forward(self, x):

        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)

        return x

    @property
    def sigma(self):
        return torch.exp(self._log_sigma)
    
    @property
    def kappa(self):
        return self.price_impact_kappa

class MultiTaskLoss(nn.Module):

    def __init__(self, num_losses: int):
        super(MultiTaskLoss, self).__init__()

        self._log_params = nn.Parameter(data=torch.ones(num_losses))

        return None
    
    def forward(self, loss):

        stds = torch.exp(self._log_params)

        total_loss = (1/(1 + stds)) * loss + torch.log(stds)

        return total_loss.mean()