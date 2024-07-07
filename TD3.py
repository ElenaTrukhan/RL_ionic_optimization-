import argparse 
import torch 
import numpy as np
import time
import os 
from utils.env import Environment, get_sturct_lib_and_calcs_gen
from pymatgen.io.ase import AseAtomsAdaptor
import ase
from ase import Atoms
import pandas as pd
from pymatgen.core.structure import Structure
from ase.calculators.lj import LennardJones
import datetime
from utils.convert_to_graph_e3nn import to_graph 
from ase.optimize import BFGS
from IPython.display import clear_output
import e3nn
from utils.model_e3nn import PeriodicNetwork_Pi, PeriodicNetwork_Q
from utils.td3 import Agent, TD3Agent
from utils.useful_funcs import get_the_last_checkpoint, extract_number


torch.set_default_dtype(torch.float64)
parser = argparse.ArgumentParser(description='TD3') 
parser.add_argument('--structures_file', type=str, default='struct/AlandFe.csv')
parser.add_argument('--r_max', type=float, default=5)

#Environment parameters 
parser.add_argument('--r0', type=float, default=1.5)
parser.add_argument('--eps', type=float, default=0.1)
parser.add_argument('--EMAX', type=float, default=1e-6)
parser.add_argument('--lamb', type=float, default=0)
parser.add_argument('--reward_func', type=str, default='step')
parser.add_argument('--r_weights', type=eval, default=[1, 0, 0.5])

# Actor and Critic parameters 
parser.add_argument('--em_dim', type=float, default=10)
parser.add_argument('--noise_clip', type=float, default=0.1)
parser.add_argument('--pi_n_layers', type=int, default=2)
parser.add_argument('--pi_mul', type=int, default=20)
parser.add_argument('--pi_lmax', type=int, default=2)
parser.add_argument('--pi_num_neighbors', type=int, default=25)
parser.add_argument('--q_n_layers', type=int, default=2)
parser.add_argument('--q_mul', type=int, default=20)
parser.add_argument('--q_lmax', type=int, default=2)
parser.add_argument('--q_num_neighbors', type=int, default=25)

# Agent parameters  
parser.add_argument('--random_seed', type=int, default=972)
parser.add_argument('--replay_size', type=int, default=int(1e6))
parser.add_argument('--gamma', type=float, default=0.9999)
parser.add_argument('--polyak', type=float, default=0.995)
parser.add_argument('--pi_lr', type=float, default=1e-5)
parser.add_argument('--q_lr', type=float, default=1e-5)
parser.add_argument('--batch_size', type=int, default=100)
parser.add_argument('--start_steps', type=int, default=0)
parser.add_argument('--update_after', type=int, default=0)
parser.add_argument('--update_every', type=int, default=1)
parser.add_argument('--target_noise', type=float, default=0.05)
parser.add_argument('--policy_delay', type=int, default=2)
parser.add_argument('--trans_coef', type=float, default=0.5)
parser.add_argument('--noise', type=eval, default=[0.2,0.2])
parser.add_argument('--path_weights', type=str, default=None)
parser.add_argument('--with_weights', type=eval, default=False)

# Training args 

parser.add_argument('--path_to_the_main_dir', type=str, default='')
parser.add_argument('--path_load', type=str, default=None)
parser.add_argument('--train_ep', type=eval, default=[10000,1000])
parser.add_argument('--test_ep', type=eval, default=[10,100])
parser.add_argument('--env_name', type=str, default='')
parser.add_argument('--suffix', type=str, default='')
parser.add_argument('--plot_step', type=eval, default=None)
parser.add_argument('--e_lim', type=eval, default=None)
parser.add_argument('--net_lim', type=eval, default=None)
parser.add_argument('--save_result', type=eval, default=True)
parser.add_argument('--start_iter', type=int, default=0)
parser.add_argument('--test_random', type=eval, default=False)
parser.add_argument('--test_every', type=int, default=1000)
parser.add_argument('--save_every', type=int, default=1000)
parser.add_argument('--with_stop', type=eval, default=False)
parser.add_argument('--max_norm_max_step', type=int, default=100)
parser.add_argument('--max_norm_limit', type=float, default=0.015)
parser.add_argument('--force_limit', type=float, default=0.9)
parser.add_argument('--noise_level', type=float, default=20)
    

args = parser.parse_args()


s_lib, calcs = get_sturct_lib_and_calcs_gen(args.structures_file)

env_kwards = {"input_struct_lib": s_lib, "convert_to_graph_func": to_graph, 
                "calculator_lib": calcs, "r0":args.r0 , "eps" : args.eps, "EMAX": args.EMAX,  
                "lamb": args.lamb, "reward_func" : args.reward_func, "r_weights": args.r_weights}

actor_feat = {"em_dim" :args.em_dim, 
                "irreps_in":f"12x0e + 1x1o + {args.em_dim}x0e",
                "irreps_out":"1x1o",   
                 "noise_clip": args.noise_clip,
                "irreps_node_attr":"0e",   
                "layers" : args.pi_n_layers,                             
                "mul" : args.pi_mul,                                
                "lmax" : args.pi_lmax,                               
                "max_radius" : args.r_max,                      
                "num_neighbors" : args.pi_num_neighbors,          
                "reduce_output" : False}

critic_feat = {"em_dim": args.em_dim, 
                "irreps_in":f"12x0e + 1x1o + {args.em_dim}x0e + 1x1o + {args.em_dim}x0e",     
                "irreps_out":"1x0e",         
                "irreps_node_attr":"0e",   
                "layers" : args.q_n_layers,                             
                "mul" : args.q_mul,                                
                "lmax" : args.q_lmax,                               
                "max_radius" : args.r_max,                      
                "num_neighbors" : args.q_num_neighbors,           
                "reduce_output" : True}

ac_kwards = {"net_actor": PeriodicNetwork_Pi, "net_critic": PeriodicNetwork_Q, 
             "actor_feat": actor_feat, "critic_feat": critic_feat}

if not os.path.exists(path_to_the_main_dir):
    os.makedirs(path_to_the_main_dir)


if args.path_weights == 'last': 
    assert os.path.exists(args.path_to_the_main_dir + "/data")
    args.path_weights = get_the_last_checkpoint(args.path_to_the_main_dir + "/data")
        
if args.path_weights is not None: 
    
    assert os.path.exists(args.path_weights)
    df = pd.read_csv(args.path_weights)
    out = df['Weights'].values[-1].split("\n")
    wf = []
    for i in range(len(out)):
        wi_t = []
        wi = out[i]
        if i == 0: 
            if len(out) == 1: 
                wi = wi[1:][:-1].split(' ')
            else: 
                wi = wi[1:].split(' ')
        elif i == len(out)-1: 
            wi = wi[:-1].split(' ')[1:]
        else: 
            wi = wi.split(' ')

        for item in wi: 
            if item !=  '':
                wi_t.append(float(item))
        wf += wi_t

else: 
    wf = None
    
    
a = {"env_fn" : Environment , 
     "actor_critic" : Agent, 
     "env_kwards" : env_kwards, 
     "ac_kwargs" : ac_kwards, 
     "seed": args.random_seed, 
     "replay_size": args.replay_size, 
     "gamma": args.gamma, 
     "polyak": args.polyak, 
     "pi_lr": args.pi_lr, 
     "q_lr": args.q_lr, 
     "batch_size": args.batch_size, 
     "start_steps": args.start_steps,
     "update_after": args.update_after, 
     "update_every": args.update_every, 
     "target_noise": args.target_noise, 
     "noise_clip": args.noise_clip, 
     "policy_delay": args.policy_delay, 
     "trans_coef": args.trans_coef, 
     "noise": args.noise, 
     "init_rewards_for_weights" : wf, 
     "with_weights" : args.with_weights}


TD3_Agent = TD3Agent(**a)

if args.path_load == 'last': 
    assert os.path.exists(args.path_to_the_main_dir + "/checkpoints")
    args.path_load = get_the_last_checkpoint(args.path_to_the_main_dir + "/checkpoints")
    args.start_steps = extract_number(args.path_load)
    print('args_start_steps', args.start_steps)
if args.path_load is not None:
    assert os.path.exists(args.path_load)
    TD3_Agent.load_model(args.path_load)

   
        
a["path_load"] = args.path_load


if args.start_iter == 0:
    with open(args.path_to_the_main_dir + "/TD3_Agent_arguments.txt", 'w') as f:
        f.write(str(a))

b = {"train_ep" : args.train_ep, 
     "test_ep" : args.test_ep, 
     "path_to_the_main_dir" : args.path_to_the_main_dir, 
     "env_name" : args.env_name, 
     "suffix" : args.suffix, 
     "plot_step" : args.plot_step, 
    "e_lim" : args.e_lim, 
    "net_lim" : args.net_lim, 
    "save_result" : args.save_result, 
    "start_iter": args.start_iter,
    "test_random": args.test_random,
    "test_every": args.test_every, 
    "save_every": args.save_every, 
    'with_stop': args.with_stop, 
     "max_norm_max_step" : args.max_norm_max_step, 
     "max_norm_limit" : args.max_norm_limit, 
     "force_limit" : args.force_limit, 
     "noise_level" : args.noise_level}

with open(args.path_to_the_main_dir + "/TD3_Agent_train_arguments.txt", 'w') as f:
    f.write(str(b))
    
TD3_Agent.train(**b)    
