import torch
import torch.nn as nn 
from torch.functional import F

import numpy as np

class MLP(nn.Module):

    def __init__(self, decomp_dim: int, 
                 learn_price_impact: bool = False,
                 sigma_start: int = 1,
                 kappa_start: int = 1):
        super(MLP, self).__init__()

        self.learn_price_impact = learn_price_impact

        self.fc1 = nn.Linear(1, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, 64)
        self.fc4 = nn.Linear(64, decomp_dim)

        if learn_price_impact:
            self.fcp1 = nn.Linear(1, 64)
            self.fcp2 = nn.Linear(64, 64)
            self.fcp3 = nn.Linear(64, 64)
            self.fcp4 = nn.Linear(64, 1)
            self._log_sigma_base = torch.log(torch.FloatTensor([sigma_start])+0.01*torch.randn(1))
            self._log_sigma = nn.Parameter(data=self._log_sigma_base)
        else:
            self._log_sigma_base = torch.log(torch.FloatTensor([sigma_start]))
            self._log_sigma = self._log_sigma_base
            # self._log_sigma = nn.Parameter(data=self._log_sigma_base+torch.log(torch.rand(1)))

            self.price_impact_kappa_base = torch.FloatTensor([kappa_start])
            self.price_impact_kappa = self.price_impact_kappa_base
            # self.price_impact_kappa = nn.Parameter(data=self.price_impact_kappa_base*torch.rand(1))

        return None

    def forward(self, x):

        x_one = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x_one))
        x = F.relu(self.fc3(x)) + x_one
        # x = F.softplus(self.fc4(x), beta=0.1)
        x = self.fc4(x)

        return x
    
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

        if self.learn_price_impact:
            x_one = F.relu(self.fcp1(nu))
            x = F.relu(self.fcp2(x_one))
            x = F.relu(self.fcp3(x)) + x_one
            return_val = self.fcp4(x)
            # return_val = (torch.sign(nu).detach()*F.softplus(self.fcp4(x), beta=0.1))
        else:
            return_val = self.kappa * nu

        return return_val
    
    def numpy_permenant_price_impact_func(self, nu):

        if type(nu) == float:
            nu = np.array([nu])
            nu = torch.FloatTensor(nu)
        if torch.is_tensor(nu):
            pass
        if isinstance(nu, np.ndarray):
            nu = torch.FloatTensor(nu)

        if self.learn_price_impact:
            x_one = F.relu(self.fcp1(nu))
            x = F.relu(self.fcp2(x_one))
            x = F.relu(self.fcp3(x)) + x_one
            return_val = self.fcp4(x).detach().numpy()
            # return_val = (torch.sign(nu).detach()*F.softplus(self.fcp4(x), beta=0.1)).detach().numpy()
        else:
            return_val = self.kappa.detach().item() * nu

        return return_val


    @property
    def sigma(self):
        return torch.exp(self._log_sigma)
    
    @property
    def kappa(self):
        return self.price_impact_kappa

class MultiTaskLoss(nn.Module):

    def __init__(self, num_losses: int, lagrangian:bool = False):
        super(MultiTaskLoss, self).__init__()
        
        self.lagrangian = lagrangian
        if lagrangian:
            self._log_params = nn.Parameter(data=torch.ones(num_losses-1), requires_grad=True)
        else:
            self._log_params = nn.Parameter(data=torch.ones(num_losses), requires_grad=True)

        return None
    
    def forward(self, loss):

        stds = torch.exp(self._log_params)**0.5

        if self.lagrangian:
            total_loss = loss[0] + (stds.reshape(-1,1) * loss[1:].reshape(-1, 1)).sum()
        else:
            total_loss = (1/stds) * loss + (1/stds).sum()

        return total_loss.mean()