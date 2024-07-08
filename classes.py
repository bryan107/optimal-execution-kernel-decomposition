import torch
import torch.nn as nn 
from torch.functional import F

import numpy as np

class MLP(nn.Module):

    def __init__(self, decomp_dim: int):
        super(MLP, self).__init__()

        self.fc1 = nn.Linear(1, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, decomp_dim)

        self._log_sigma_base = torch.log(torch.FloatTensor([0.1]))
        self._log_sigma = self._log_sigma_base
        # self._log_sigma = nn.Parameter(data=self._log_sigma_base+torch.log(torch.rand(1)))

        self.price_impact_kappa_base = torch.FloatTensor([0.01])
        self.price_impact_kappa = self.price_impact_kappa_base
        # self.price_impact_kappa = nn.Parameter(data=self.price_impact_kappa_base*torch.rand(1))

    def forward(self, x):

        x_one= F.relu(self.fc1(x))
        x_two = F.relu(self.fc2(x_one)) 
        x_three = self.fc3(x_two)

        return x_three
    
    def approx_func(self, s, t):

        s_val = self.forward(s)
        t_val = self.forward(t)

        return (s_val * t_val).sum(axis=1)
    
    def numpy_approx_func(self, s, t):
        
        if type(s) == float:
            s_in_val = np.array([s])
            t_in_val = np.array([t])
        else:
            s_in_val = s
            t_in_val = t

        s = torch.FloatTensor(s_in_val).reshape(-1,1)
        t = torch.FloatTensor(t_in_val).reshape(-1,1)

        s_val = self.forward(s)
        t_val = self.forward(t)

        return (s_val * t_val).sum(axis=-1).detach().numpy()
    
    def permenant_price_impact_func(self, nu):

        return self.kappa * nu
    
    def numpy_permenant_price_impact_func(self, nu):

        return self.kappa.detach().item() * nu

    @property
    def sigma(self):
        return torch.exp(self._log_sigma)
    
    @property
    def kappa(self):
        return self.price_impact_kappa

class MultiTaskLoss(nn.Module):

    def __init__(self, num_losses: int):
        super(MultiTaskLoss, self).__init__()

        self._log_params = nn.Parameter(data=torch.ones(num_losses), requires_grad=True)

        return None
    
    def forward(self, loss):

        stds = torch.exp(self._log_params)**0.5

        total_loss = (1/(1 + stds)) * loss + torch.log(stds)**2

        return total_loss.mean()