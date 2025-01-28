import torch 
from torch import nn 
import torch.nn.functional as F
from torch_geometric.data import Data
from utils.utils_model import Network 
from e3nn import o3
import torch_geometric
import torch_scatter
import e3nn 
from torch.distributions.normal import Normal

torch.set_default_dtype(torch.float64)

### Networks for TD3 Agent 

class PeriodicNetwork_Pi(Network):
    r"""The class for the Actor, based on Tensor-Field Networks in the e3nn library 

    Parameters
    ----------
    em_dim : int
        Number of neurons in the embedding layer for the force norm.

    noise_clip : float 
        Limit for the absolute value of target policy smoothing noise. 
    
    scaled : bool
        Flag controlling whether to predict actions relative to the unit cell.
    """

    def __init__(self, em_dim, noise_clip, scaled = False, **kwargs):            

        kwargs['reduce_output'] = False
        self.scaled = scaled   
        self.exploration_mode
        
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
    r"""The class for the Critic, based on Tensor-Field Networks in the e3nn library 

    Parameters
    ----------
    em_dim : int
        Number of neurons in the embedding layer for the force norm.

    """
    
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


### Networks for SAC Agent 

LOG_STD_MAX = 1
LOG_STD_MIN = -20

class Gaussian_actor(Network):
    r"""The class for the SAC Actor, based on Tensor-Field Networks in the e3nn library 

    Parameters
    ----------
    irreps_in : `e3nn.o3.Irreps`
        representation of the input node features
    
    irreps_out : `e3nn.o3.Irreps` or None
        representation of the output node features
    
    irreps_node_attr : `e3nn.o3.Irreps`
        representation of the node attributes
    
    em_dim : int
        Number of neurons in the embedding layer for the force norm.

    max_radius : float
        maximum radius for the convolution

    layers : int
        number of gates (non linearities)
    
    mul : int 
        multiplicity of the Irreps in hidden layers

    lmax : int 
        maximum order of spherical harmonics in the convolutional layer 

    number_of_basis : int
        number of basis on which the edge length are projected
    
    radial_layers : int
        number of hidden layers in the radial fully connected network

    radial_neurons : int
        number of neurons in the hidden layers of the radial fully connected network    
    
    num_neighbors : float
        typical number of nodes convolved over

    """
    def __init__(self,  
                irreps_in, 
                irreps_out,
                irreps_node_attr,
                em_dim,
                max_radius ,
                layers = 2,
                mul = 20,
                lmax = 1, 
                number_of_basis=10,
                radial_layers=1,
                radial_neurons=100,
                num_neighbors=1):            
        
        act = { 1: torch.nn.functional.silu,  -1: torch.tanh} 
        act_gates = {  1: torch.sigmoid,  -1: torch.tanh}

        self.irreps_in = o3.Irreps(irreps_in) if irreps_in is not None else None
        self.irreps_hidden = o3.Irreps([(mul, (l, p)) for l in range(lmax + 1) for p in [-1, 1]])
        self.irreps_out = o3.Irreps(irreps_out)
        self.irreps_node_attr = o3.Irreps(irreps_node_attr) if irreps_node_attr is not None else o3.Irreps("0e")
        self.irreps_edge_attr = o3.Irreps.spherical_harmonics(lmax)
        
        super().__init__(irreps_in, lmax, irreps_node_attr, max_radius, number_of_basis)
        
        # Convolutional layer for expectation values vector mu 
        self.layers_out1 = super().get_convolution(self.irreps_in, self.irreps_hidden, self.irreps_edge_attr, self.irreps_node_attr, 
                                                "1x1o", act, act_gates, layers, number_of_basis, radial_layers, radial_neurons, 
                                                num_neighbors)
        # Convolutional layer for std 
        self.layers_out2 = super().get_convolution(self.irreps_in, self.irreps_hidden, self.irreps_edge_attr, self.irreps_node_attr, 
                                                "1x0e", act, act_gates, layers, number_of_basis, radial_layers, radial_neurons, 
                                                num_neighbors)
        self.em1 = nn.Linear(1, em_dim) 
        
    def forward(self, data, deterministic = False, with_logprob = True): 
        
        x, z, edge_src, edge_dst, edge_attr, edge_length_embedded, batch = super().forward(data)
        
        forces_ampl = F.leaky_relu(self.em1(data.forces_norm))
        mu = torch.hstack([x, data.forces_stack, forces_ampl])
        log_std = torch.hstack([x, data.forces_stack, forces_ampl])

        # Predict mu and construct unit direction vectors
        for lay in self.layers_out1: 
            mu = lay(mu, z, edge_src, edge_dst, edge_attr, edge_length_embedded)
        mu_norm = mu.norm(dim=1).unsqueeze(1)
        mu_norm_cor = mu_norm + 1*(mu_norm==0)
        mu_stack = mu/mu_norm_cor
        
        # Predict std 
        for lay in self.layers_out2:
            log_std = lay(log_std, z, edge_src, edge_dst, edge_attr, edge_length_embedded)
        
        log_std = torch.clamp(log_std, LOG_STD_MIN, LOG_STD_MAX) 
        std = torch.exp(log_std)

        # Construct a normal distribution for atomic shift norms
        pi_distribution = Normal(mu_norm, std)
        
        if deterministic:
            pi_action_norm = mu_norm
            pi_action = mu
        else:
            pi_action_norm = pi_distribution.rsample()
            pi_action = torch.vstack([mu_stack[i]*pi_action_norm[i] for i in range(len(mu_stack))])
        
        if with_logprob:
            logp_pi = pi_distribution.log_prob(pi_action_norm)
            logp_pi = torch_scatter.scatter_add(logp_pi, batch, dim=0) 
            logp_pi  = logp_pi.sum(axis=-1)
        else:
            logp_pi = None
        return Data(x = pi_action), logp_pi
    
class Q_func(Network):

    r"""The class for the SAC Critic, based on Tensor-Field Networks in the e3nn library 

    Parameters
    ----------
    irreps_in : `e3nn.o3.Irreps`
        representation of the input node features
    
    irreps_out : `e3nn.o3.Irreps` or None
        representation of the output node features
    
    irreps_node_attr : `e3nn.o3.Irreps`
        representation of the node attributes
    
    em_dim : int
        Number of neurons in the embedding layer for the force norm.

    max_radius : float
        maximum radius for the convolution

    layers : int
        number of gates (non linearities)
    
    mul : int 
        multiplicity of the Irreps in hidden layers

    lmax : int 
        maximum order of spherical harmonics in the convolutional layer 

    number_of_basis : int
        number of basis on which the edge length are projected
    
    radial_layers : int
        number of hidden layers in the radial fully connected network

    radial_neurons : int
        number of neurons in the hidden layers of the radial fully connected network    
    
    num_neighbors : float
        typical number of nodes convolved over

    """

    def __init__(self,  
                irreps_in, 
                irreps_out,
                irreps_node_attr,
                max_radius,
                em_dim, 
                layers = 2,
                mul = 20,
                lmax = 1, 
                number_of_basis=10,
                radial_layers=1,
                radial_neurons=100,
                num_neighbors=1.):            
        
        act = { 1: torch.nn.functional.silu,  -1: torch.tanh} 
        act_gates = {  1: torch.sigmoid,  -1: torch.tanh}
        self.irreps_in = o3.Irreps(irreps_in) if irreps_in is not None else None
        self.irreps_hidden = o3.Irreps([(mul, (l, p)) for l in range(lmax + 1) for p in [-1, 1]])
        self.irreps_out = o3.Irreps(irreps_out)
        self.irreps_node_attr = o3.Irreps(irreps_node_attr) if irreps_node_attr is not None else o3.Irreps("0e")
        self.irreps_edge_attr = o3.Irreps.spherical_harmonics(lmax)
        
        super().__init__(irreps_in,
        lmax, 
        irreps_node_attr, 
        max_radius, 
        number_of_basis)
        
        self.layers  = super().get_convolution(self.irreps_in, self.irreps_hidden, self.irreps_edge_attr, self.irreps_node_attr, 
                                                self.irreps_out, act, act_gates, layers, number_of_basis, radial_layers, radial_neurons, 
                                                num_neighbors)        
        self.em_act = nn.Linear(1, em_dim)        
        self.em1 = nn.Linear(1, em_dim) 
        
    def forward(self, data, actions) :
        
        x, z, edge_src, edge_dst, edge_attr, edge_length_embedded, batch = super().forward(data)
        action = actions.x
        action_norm = action.norm(dim=1) 
        action_norm_cor = action_norm + 1*(action_norm==0)
        action_stack = action/action_norm_cor[:, None] 
        action_ampl = F.leaky_relu(self.em_act(action_norm.unsqueeze(1)))
        forces_ampl = F.leaky_relu(self.em1(data.forces_norm))
        x = torch.hstack([x, data.forces_stack, forces_ampl, action_stack, action_ampl])
    
        for lay in self.layers:
            x = lay(x, z, edge_src, edge_dst, edge_attr, edge_length_embedded)
        x = torch_scatter.scatter_mean(x, batch, dim=0)  # take mean over atoms per example
        
        return torch.squeeze(x, -1) 


    
