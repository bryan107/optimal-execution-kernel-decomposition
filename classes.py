import torch
import torch.nn as nn 
from torch.functional import F

import numpy as np

def init_weights(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.xavier_uniform(m.weight)
        m.bias.data.fill_(0.01)

class Kernel(nn.Module):
    def __init__(self, 
                 input_dim: int,
                 decomp_dim: int, 
                 ):
        
        super(Kernel, self).__init__()

        self.fc1 = nn.Linear(input_dim, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, 64)
        self.fc4 = nn.Linear(64, decomp_dim)

        init_weights(self)

        return None

    def forward(self, x, parameters = None):

        if not parameters is None:
            x = torch.hstack([x, parameters])

        x_one = F.relu(self.fc1(x)) 
        x_two = F.relu(self.fc2(x_one)) 
        x = F.relu(self.fc3(x_two)) + x_one
        x = self.fc4(x)

        return x
    
    def approx_func(self, 
                    s, 
                    t,
                    parameters):

        s_val = self.forward(s, parameters)
        t_val = self.forward(t, parameters)

        return (s_val * t_val).sum(axis=1)
    
class PriceImpact(nn.Module):

    def __init__(self,):

        super(PriceImpact, self).__init__()

        self.fc1 = nn.Linear(1, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, 64)
        self.fc4 = nn.Linear(64, 1)

        init_weights(self)

        return None
    
    def forward(self, nu):

        x_one = F.relu(self.fc1(nu))
        x_two = F.relu(self.fc2(x_one))
        x = F.relu(self.fc3(x_two)) + x_one
        return_val = self.fc4(x)

        return return_val
    
class ModelParameters(nn.Module):

    def __init__(self, parameter_start):
        super().__init__()

        if isinstance(parameter_start, torch.FloatTensor):
            start_params = parameter_start.flatten()
        else:
            start_params = torch.FloatTensor(parameter_start.flatten()).flatten()
        start_params[0] = torch.log(start_params[0])

        self._params = nn.Parameter(start_params, 
                                    requires_grad=True)
    
    
    @property
    def params(self):
        return self._params

#####################################################################


class MLP(nn.Module):

    def __init__(self, 
                 decomp_dim: int, 
                 parameter_start = np.array([0.01]),
                 learn_price_impact: bool = False,
                 ):
        super(MLP, self).__init__()

        self.learn_price_impact = learn_price_impact

        self.kernel_func = Kernel(input_dim=len(parameter_start), decomp_dim=decomp_dim)

        if learn_price_impact:
            self.price_impact = PriceImpact()
            self.model_parameters = ModelParameters(parameter_start=parameter_start)
        else:
            self._log_sigma_base = torch.log(torch.FloatTensor([parameter_start[0]]))
            self._log_sigma = self._log_sigma_base
            # self._log_sigma = nn.Parameter(data=self._log_sigma_base+torch.log(torch.rand(1)))

            self.price_impact_kappa_base = torch.FloatTensor([parameter_start[1]])
            self.price_impact_kappa = self.price_impact_kappa_base
            # self.price_impact_kappa = nn.Parameter(data=self.price_impact_kappa_base*torch.rand(1))

        return None

    def forward(self, x, parameters= None):

        if not parameters is None:
            x = self.kernel_func(x,parameters)
        else:   
            x = self.kernel_func(x)

        return x
    
    def approx_func(self, s, t, parameters= None):

        if not parameters is None:
            s_val = self.forward(s, parameters)
            t_val = self.forward(t, parameters)
        else:
            s_val = self.forward(s)
            t_val = self.forward(t)

        return (s_val * t_val).sum(dim=-1)
    
    def numpy_approx_func(self, s, t, parameters= None):
        
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

        return (s_val * t_val).sum(dim=-1).detach().numpy()
    
    def permenant_price_impact_func(self, nu):

        if self.learn_price_impact:
            x = self.price_impact(nu)
        else:
            x = self.kappa * nu

        return x
    
    def numpy_permenant_price_impact_func(self, nu):

        if type(nu) == float:
            nu = np.array([nu])
            nu = torch.FloatTensor(nu)
        if torch.is_tensor(nu):
            pass
        if isinstance(nu, np.ndarray):
            nu = torch.FloatTensor(nu)

        if self.learn_price_impact:
            x = self.price_impact(nu).detach().numpy()
        else:
            x = self.kappa.detach().item() * nu

        return x

    @property
    def sigma(self):
        return torch.exp(self.model_parameters.params[0])
    
    @property
    def params(self):
        return self.model_parameters.params[1:]

#####################################################################


class MultiTaskLoss(nn.Module):

    def __init__(self, num_losses: int, lagrangian:bool = False):
        super(MultiTaskLoss, self).__init__()

        self.num_losses = num_losses
        
        self.lagrangian = lagrangian
        if lagrangian:
            self._log_params = nn.Parameter(data=0.1*torch.ones(num_losses-1), requires_grad=True)
        else:
            self._log_params = nn.Parameter(data=0.1*torch.ones(num_losses), requires_grad=True)

        return None
    
    def forward(self, loss):

        stds = torch.exp(self._log_params)**0.5

        if self.lagrangian:
            total_loss = loss[0] + (stds.reshape(1,self.num_losses-1) * loss[1:].reshape(-1,self.num_losses-1)) 
        else:
            total_loss = stds.reshape(1,self.num_losses) * loss.reshape(-1,self.num_losses) 

        return total_loss.mean() + (1/stds**2).sum()