import argparse 
import torch 
import pandas as pd
from utils.convert_to_graph_e3nn import to_graph 
from utils.model_e3nn import PeriodicNetwork_Pi
from utils.model_CGCNN import GraphConvNet_actor

from utils.env import get_sturct_lib_and_calcs_gen
from pymatgen.io.ase import AseAtomsAdaptor
from ase.optimize import BFGS
import numpy as np
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def relax_new(struct, max_step, eps, pi, device):
    force = struct.get_forces()
    o = to_graph(AseAtomsAdaptor.get_structure(struct), force).to(device)
    for i in range(max_step): 
        a = pi(o).x.detach().to('cpu').numpy()
        struct.translate(a)
        struct.wrap()
        forces = struct.get_forces()
        max_f = max((forces**2).sum(axis=1)**0.5)
        print(max_f)
        if max_f <= eps: 
            break 
        else: 
            struct_pm = AseAtomsAdaptor.get_structure(struct)
            o = to_graph(struct_pm, forces).to(device)


torch.set_default_dtype(torch.float64)
parser = argparse.ArgumentParser(description='TD3') 

#Environment parameters 
parser.add_argument('--structures_file', type=str, help='Path to the file with input structures')
parser.add_argument('--eps', type=float, default=0.1, help='Force threshold, eV/A')
parser.add_argument('--max_step', type=int, default=100, help='Maximum number of relaxation steps')
parser.add_argument('--model_type', type=str, default='e3nn(TD3)', help='Type of Actor model: CGCNN(TD3) | e3nn(TD3) | e3nn(SAC)')
parser.add_argument('--model_params', type=eval, default=[5,10,2,20,2, 20], help='Model parameters. For e3nn(TD3/SAC): [r_max, em_dim, pi_n_layers, pi_mul, pi_lmax, num_neighbors], for CGCNN(TD3): [num_node_features, aggr_a, n_l1, n_l2, n_l3, n_l4]')
parser.add_argument('--path_load', type=str, default=None, help = 'Path to the model checkpoints')    

args = parser.parse_args()

if args.model_type == 'CGCNN(TD3)': 
    actor_feat = {"num_node_features":args.model_params[0]+4, 
              "aggr": args.model_params[1],
              "n_l1":args.model_params[2], 
              "n_l2":args.model_params[3],
              "n_l3":args.model_params[4],
              "n_l4":args.model_params[5]}
    pi = GraphConvNet_actor(**actor_feat)
else: 
    actor_feat = {"max_radius" : args.model_params[0],  
                    "em_dim" :args.model_params[1], 
                "irreps_in":f"12x0e + 1x1o + {args.model_params[1]}x0e",
                "irreps_out":"1x1o",   
                "irreps_node_attr":"0e",   
                "layers" : args.model_params[2],                             
                "mul" : args.model_params[3],                                
                "lmax" : args.model_params[4],
                "num_neighbors" : args.model_params[5],  
                "reduce_output" : False}
    if args.model_type == 'e3nn(SAC)':
        actor_feat["irreps_out"] = f"{args.model_params[1]}x0e+{args.model_params[1]}x1o"
    pi = PeriodicNetwork_Pi(**actor_feat)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
pi.to(device)

if args.path_load is not None: 
    print('Loading models from {}'.format(args.path_load))
    checkpoint = torch.load(args.path_load, map_location = device)
    pi.load_state_dict(checkpoint['ac_pi'])

s_lib, calcs = get_sturct_lib_and_calcs_gen(args.structures_file)

for i in range(len(s_lib)): 
    np.random.seed(i)
    struct = AseAtomsAdaptor.get_atoms(s_lib[i])
    struct.calc = calcs[i] 
    relax = BFGS(struct)
    relax.run(fmax=0.3)
    trans = 0.5*np.random.rand(struct.get_positions().shape[0], 3)
    struct.translate(trans)
    max_f = max((struct.get_forces()**2).sum(axis=1)**0.5)
    print(f'Starting maximum force for structure # {i}: ', max_f, " ev/A")
    relax_new(struct, args.max_step, args.eps, pi, device)
    max_f = max((struct.get_forces()**2).sum(axis=1)**0.5)
    print(f'Final maximum force for structure # {i}: ', max_f, " ev/A")


