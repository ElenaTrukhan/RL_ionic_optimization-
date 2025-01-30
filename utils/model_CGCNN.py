import torch 
from torch import nn 
import torch.nn.functional as F
from torch_geometric.nn import CGConv
from torch_geometric.nn import global_add_pool
from torch_geometric.data import Data

def gaussian_distance(distances, a, b, step):
    """Calculates Gaussian dtstances from the list (of lists) of distances """

    assert a < b
    assert b - a > step
    distances = torch.Tensor(distances)
    substr = torch.arange(a, b+step, step).to(distances.device)
    return torch.exp(-(distances[:, None] - substr)**2/step**2)

params = {
    "radius": 5, # cut-off radius
    "max_num_nbr": 30, # maximum number of neighbors to consider
    "dmin": 0, # min for Gaussian distance
    "dmax": 5, # max for Gaussian distance
    "step": 0.2 # step for Gaussian distance
}

class GraphConvNet_actor(nn.Module):
    r"""Crystal Graph Convolutional Neural Network for Actor.

    Parameters
    ----------
    num_node_features : int
        Number of graph node features.

    noise_clip : float 
        Limit for absolute value of target policy smoothing noise. 
    
    aggr : str 
        Aggregation operator.
        
    n_l1-nl4 : int 
        Hidden layers sizes. 
    """


    def __init__(self, num_node_features, aggr, n_l1, n_l2, n_l3, n_l4, noise_clip):
        super().__init__()
        self.noise_clip = noise_clip
 
        self.embedding = nn.Sequential(
            nn.Linear(num_node_features, n_l1),
            nn.LeakyReLU(),
            nn.Linear(n_l1, n_l2),
            nn.LeakyReLU(),
            nn.Linear(n_l2, n_l3),
            nn.LeakyReLU(),
        )
        num_edge_features = int((params["dmax"] - params["dmin"])/params["step"]) + 1
        self.conv1 = CGConv(n_l3, num_edge_features, aggr=aggr, batch_norm=False)
        self.conv2 = CGConv(n_l3, num_edge_features, aggr=aggr, batch_norm=False)
        self.conv3 = CGConv(n_l3, num_edge_features, aggr=aggr, batch_norm=False)
        self.pooling = global_add_pool

        self.desc = nn.Sequential(
            nn.Linear(n_l3, n_l4),
            nn.ReLU(),
            nn.Linear(n_l4, 3)
        )

    def forward(self, data, noise_scale = None):

        x, edge_index, edge_length, forces_stack, forces_norm = data.x, data.edge_index, data.edge_length, data.forces_stack, data.forces_norm
        edge_attr = gaussian_distance(edge_length, params["dmin"], params["dmax"], params["step"])
        x = torch.cat((x, forces_stack, forces_norm), dim=1)
        x = self.embedding(x)
        x = self.conv1(x, edge_index, edge_attr)
        x = F.leaky_relu(x)
        x = self.conv2(x, edge_index, edge_attr)
        x = F.leaky_relu(x)
        x = self.conv3(x, edge_index, edge_attr)
        x = self.desc(x)

        if noise_scale is not None: 
            epsilon = torch.randn_like(x) * noise_scale
            epsilon = torch.clamp(epsilon, -self.noise_clip, self.noise_clip)
            x += epsilon

        return Data(x = x)

class GraphConvNet_critic(nn.Module):
    r"""Crystal Graph Convolutional Neural Network for Critic.

    Parameters
    ----------
    num_node_features : int
        Number of graph node features.
    
    aggr : str 
        Aggregation operator.
        
    n_l1-nl5 : int 
        Hidden layers sizes. 
    """

    def __init__(self, num_node_features, aggr, n_l1, n_l2, n_l3, n_l4, n_l5):
        super().__init__()
        
            
        self.embedding = nn.Sequential(
            nn.Linear(num_node_features, n_l1),
            nn.LeakyReLU(),
            nn.Linear(n_l1, n_l2),
            nn.LeakyReLU(),
            nn.Linear(n_l2, n_l3),
            nn.LeakyReLU(),
        )
        num_edge_features = int((params["dmax"] - params["dmin"])/params["step"]) + 1
        self.conv1 = CGConv(n_l4+n_l3, num_edge_features, aggr=aggr, batch_norm=False)
        self.conv2 = CGConv(n_l4+n_l3, num_edge_features, aggr=aggr, batch_norm=False)
        self.conv3 = CGConv(n_l4+n_l3, num_edge_features, aggr=aggr, batch_norm=False)
        self.pooling = global_add_pool

        self.desc = nn.Sequential(
            nn.Linear(n_l4 + n_l3, n_l5),
            nn.ReLU(),
            nn.Linear(n_l5, 1),
        )
        self.ln_act = nn.Linear(3, n_l4)

    def forward(self, data, action):
        x, edge_index, edge_length, forces_stack, forces_norm = data.x, data.edge_index, data.edge_length, data.forces_stack, data.forces_norm
        edge_attr = gaussian_distance(edge_length, params["dmin"], params["dmax"], params["step"])
        x = torch.cat((x, forces_stack, forces_norm), dim=1)
        x = self.embedding(x)
        action_input = self.ln_act(action.x)
        state_action = torch.cat((x, action_input), dim = 1)
        state_action = self.conv1(state_action, edge_index, edge_attr)
        state_action = F.leaky_relu(state_action)
        state_action = self.conv2(state_action, edge_index, edge_attr)
        state_action = F.leaky_relu(state_action)
        state_action = self.conv3(state_action, edge_index, edge_attr)
        y = self.pooling(state_action, data.batch)
        y = self.desc(y)
        return torch.squeeze(y, -1)