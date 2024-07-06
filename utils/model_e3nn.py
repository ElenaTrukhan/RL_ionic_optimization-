import torch 
from torch import nn 
import torch.nn.functional as F
from torch_geometric.data import Data
from utils.utils_model import Network 
from torch_cluster import radius_graph
from torch_scatter import scatter
from e3nn import o3
import torch_geometric
from e3nn.nn import FullyConnectedNet, Gate
from e3nn.o3 import TensorProduct, FullyConnectedTensorProduct
from typing import Dict, Union
import torch_scatter
from copy import deepcopy
import e3nn 

torch.set_default_dtype(torch.float64)

class PeriodicNetwork_Pi(Network):
    def __init__(self, em_dim, noise_clip, scaled = False, **kwargs):            
        # override the `reduce_output` keyword to instead perform an averge over atom contributions    

        kwargs['reduce_output'] = False
        self.scaled = scaled        
        
        super().__init__(**kwargs)
        self.em = nn.Linear(1, em_dim)
        self.noise_clip = noise_clip

    def forward(self, data, noise_scale = None) :
        data_copy = data.clone()
        if noise_scale is not None: 
            axis, angle = e3nn.o3.rand_axis_angle(1)
            angle *= 0.5*noise_scale  
            angle = torch.clamp(angle, -self.noise_clip, self.noise_clip)
            rot_matrix = e3nn.o3.axis_angle_to_matrix(axis, angle).to(data_copy.forces_stack.device)
            data_copy.forces_stack = torch.matmul(data_copy.forces_stack, rot_matrix[0])
            
            epsilon = (2*torch.rand(data_copy.forces_norm.shape[0],1)-1)*noise_scale
            epsilon = torch.clamp(epsilon, -self.noise_clip, self.noise_clip).to(data_copy.forces_stack.device)
            data_copy.forces_norm *= (1+epsilon) 
        
        forces_ampl = F.leaky_relu(self.em(data_copy.forces_norm))
        data_copy.x = torch.hstack([data_copy.x, data_copy.forces_stack, forces_ampl])
        output = super().forward(data_copy)
                
        if self.scaled: 
            output = torch.tanh(output)
        return Data(x = output)

class PeriodicNetwork_Q(Network):
    def __init__(self, em_dim, **kwargs):            
        # override the `reduce_output` keyword to instead perform an averge over atom contributions    
        
        kwargs['reduce_output'] = False
        
        super().__init__(**kwargs)
        
        self.em = nn.Linear(1, em_dim)
        self.em_act = nn.Linear(1, em_dim)

    def forward(self, data, actions) -> torch.Tensor:
        
        data_copy = data.clone()
        action = actions.x
        action_norm = action.norm(dim=1) 
        action_norm_cor = action_norm + 1*(action_norm==0)
        action_stack = action/action_norm_cor[:, None]
        action_ampl = F.leaky_relu(self.em_act(action_norm.unsqueeze(1)))
        
        forces_ampl = F.leaky_relu(self.em(data_copy.forces_norm))
        data_copy.x = torch.hstack([data_copy.x, data_copy.forces_stack, forces_ampl, action_stack, action_ampl])
        
        output = super().forward(data_copy)
        # if pool_nodes was set to True, use scatter_mean to aggregate
        output = torch_scatter.scatter_mean(output, data_copy.batch, dim=0)  # take mean over atoms per example
        
        return torch.squeeze(output, -1) 
    
    
