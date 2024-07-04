import pandas as pd 
import torch 
import torch_geometric
from torch_geometric.data import Data
import numpy as np
from pymatgen.io.ase import AseAtomsAdaptor
from typing import Dict, Union


atomic_properties_url = "https://raw.githubusercontent.com/lenkakolenka/File/f75f040424669788d658bec8afe16f3e35d44d9e/atomic_properties.csv"
atomic_properties = pd.read_csv(atomic_properties_url, decimal=",", index_col=None)
atomic_properties = atomic_properties.drop('Unnamed: 0', axis=1)
mean = atomic_properties.mean()
std = atomic_properties.std()
standardized_atomic_properties = (atomic_properties-mean)/std

embedding = {}
for idx, column in enumerate(standardized_atomic_properties):
    embedding[idx+1] = list(standardized_atomic_properties[column])

params = {
    "radius": 5, # cut-off radius
    "max_num_nbr": 30, # maximum number of neighbors to consider
    "dmin": 0, # min for Gaussian distance
    "dmax": 5, # max for Gaussian distance
    "step": 0.2 # step for Gaussian distance
}

torch.set_default_dtype(torch.float64)

def to_graph(structure, forces, action = None):
    
    x = torch.vstack([torch.Tensor(embedding[number]) for number in structure.atomic_numbers])
    
    nbrs = structure.get_all_neighbors(params["radius"])
    if params["max_num_nbr"] is None:
        # we sort the neighbors by distance
        nbrs = [sorted(
            nbr, key=lambda atom: atom[1]) for nbr in nbrs]
    else:
        nbrs = [sorted(
            nbr, key=lambda atom: atom[1])[:params["max_num_nbr"]] for nbr in nbrs]

    edge_src = []
    edge_dst = []
    edge_length = []
    edge_vec = []  
    for idx, nbr in enumerate(nbrs): 
        for atom in nbr:
            edge_src.append(idx)
            edge_dst.append(atom[2])
            edge_length.append(atom[1])
            edge_vec.append((atom[0].coords - structure.cart_coords[idx]).tolist())
    edge_index = torch.vstack([torch.LongTensor(edge_src), torch.LongTensor(edge_dst)])
    pos = torch.tensor(AseAtomsAdaptor.get_atoms(structure).get_positions().tolist())
    
    if forces is not None: 
        forces = torch.tensor(forces.tolist())
        forces_norm = forces.norm(dim=1)
        forces_norm_cor = forces_norm + 1*(forces_norm==0)
        forces_stack = forces/forces_norm_cor[:, None]
#         x =  torch.hstack([x, forces_stack])
        forces_norm = forces_norm.unsqueeze(1)
    else: 
        forces_norm = None
    return Data(x= x, 
                edge_index=edge_index, 
                edge_vec = torch.tensor(edge_vec), 
                edge_length = torch.tensor(edge_length), 
                pos = pos, 
                forces_stack = forces_stack, 
                forces_norm = forces_norm) 
