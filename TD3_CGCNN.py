import argparse 
import torch 
import os 
import pandas as pd
from utils.convert_to_graph_e3nn import to_graph 
from utils.model_CGCNN import GraphConvNet_actor, GraphConvNet_critic
from utils.td3 import Agent, TD3Agent
from utils.utils import get_the_last_checkpoint, extract_number
from utils.env import Environment, get_sturct_lib_and_calcs_gen


torch.set_default_dtype(torch.float64)
parser = argparse.ArgumentParser(description='TD3') 

#Environment parameters 
parser.add_argument('--structures_file', type=str, help='Path to the file with input structures')
parser.add_argument('--eps', type=float, default=0.1, help='Force threshold, eV/A')
parser.add_argument('--reward_func', type=str, default='step', help='Reward function type: force | log_force | step | hybrid')
parser.add_argument('--r_weights', type=eval, default=[1, 0, 0.5], help='Weights for the hybrid reward function')
parser.add_argument('--r0', type=float, default=1.5, help='Minimum possible distance between atoms. During relaxations, atoms shift back if the distance between them is less than r0')
parser.add_argument('--stop_numb', type=int, default=50, help='Maximum number of relaxation steps during which the Agent is allowed to perform actions that are corrected at each step')

# Agent parameters  
parser.add_argument('--random_seed', type=int, default=972, help = 'Seed for random number generators')
parser.add_argument('--replay_size', type=int, default=int(1e6), help = 'Maximum length of replay buffer')
parser.add_argument('--gamma', type=float, default=0.9999, help = 'Discount factor')
parser.add_argument('--polyak', type=float, default=0.995, help = 'Interpolation factor in polyak averaging for target networks')
parser.add_argument('--pi_lr', type=float, default=1e-5, help = 'Learning rate for Actor')
parser.add_argument('--q_lr', type=float, default=1e-5, help = 'Learning rate for Critic')
parser.add_argument('--batch_size', type=int, default=100, help = 'Minibatch size for SGD')
parser.add_argument('--start_steps', type=int, default=0, help = 'Number of steps for fake action selection, before running real policy')
parser.add_argument('--update_after', type=int, default=0, help = 'Number of env interactions to collect before starting to do gradient descent updates')
parser.add_argument('--update_every', type=int, default=1, help = 'Number of env interactions that should elapse between gradient descent updates')
parser.add_argument('--target_noise', type=float, default=0.05, help = 'Noise level for smoothing noise added to target policy')
parser.add_argument('--policy_delay', type=int, default=2, help = 'Policy will only be updated once every policy_delay times for each update of the Q-networks')
parser.add_argument('--trans_coef', type=float, default=0.5, help = 'Distortion parameter that determines how far the structure is shifted from the local minimum before the onset of the relaxation episode')
parser.add_argument('--noise', type=eval, default=[0.2,0.2], help = 'Noise level for exploration noise added to policy at training time')
parser.add_argument('--path_weights', type=str, default=None, help = 'Path to weights corresponding to the structures in the input dataset that represent their priority for selection during training')
parser.add_argument('--with_weights', type=eval, default=False, help = 'Flag indicating whether structures should be selected during training according to weights or uniformly')

# Training parameters 
parser.add_argument('--path_to_the_main_dir', type=str, default='', help = 'Path to the folder where the results are saved')
parser.add_argument('--path_load', type=str, default=None, help = 'Path to the model checkpoints')
parser.add_argument('--train_ep', type=eval, default=[10000,1000], help = '[M_train, N_train]')
parser.add_argument('--test_ep', type=eval, default=[10,100], help = '[M_test, N_test]')
parser.add_argument('--env_name', type=str, default='', help = 'Name of the experiment')
parser.add_argument('--e_lim', type=eval, default=None, help = 'y-limits for plots of maximum forces')
parser.add_argument('--net_lim', type=eval, default=None, help = 'y-limits for plots of losses')
parser.add_argument('--save_result', type=eval, default=True, help = 'Flag indicating whether learning curves are plotted and saved')
parser.add_argument('--start_iter', type=int, default=0, help = 'The initial iteration number')
parser.add_argument('--test_random', type=eval, default=False, help = 'Flag indicating whether to randomly select num_test_episodes structures to relax from the input dataset or to relax them one by one in order num_test_episodes times each')
parser.add_argument('--test_every', type=int, default=1000, help = 'Agent is tested every test_every training steps')
parser.add_argument('--save_every', type=int, default=1000, help = 'Model weights and plots are saved every save_every training steps')
parser.add_argument('--N_gr', type=int, default=100, help = 'N_gr in additional greedy exploration')
parser.add_argument('--d_r_max', type=float, default=0.015, help = 'Delta r_max in additional greedy exploration')
parser.add_argument('--f_max', type=float, default=0.1, help = 'f_max in additional greedy exploration')
parser.add_argument('--noise_level', type=float, default=20, help = 'Noise level in additional greedy exploration')
parser.add_argument('--nfake', type=int, default=10, help = 'Fake episodes are selected and added to the replay buffer every nfake training steps')

# Actor and Critic parameters 
parser.add_argument('--num_node_features', type=int, default=12, help='Number of graph node features without forces')
parser.add_argument('--aggr_a', type=str, default="mean", help='The aggregation operator for Actor: add | mean | max')
parser.add_argument('--n_l_a', type=eval, default=[30,50, 120,60], help = 'Hidden layers sizes for Actor [n_l1. n_l2, n_l3, n_l4]')
parser.add_argument('--noise_clip', type=float, default=0.1, help='Limit for absolute value of target policy smoothing noise')
parser.add_argument('--aggr_c', type=str, default="mean", help='The aggregation operator for Critic: add | mean | max')
parser.add_argument('--n_l_c', type=eval, default=[30,50, 120,20,60], help = 'Hidden layers sizes for Critic [n_l1. n_l2, n_l3, n_l4, n_l5]')

args = parser.parse_args()

s_lib, calcs = get_sturct_lib_and_calcs_gen(args.structures_file)

env_kwards = {"input_struct_lib": s_lib, "convert_to_graph_func": to_graph, 
                "calculator_lib": calcs, "r0":args.r0 , "eps" : args.eps, "EMAX": args.EMAX,  
                "reward_func" : args.reward_func, "r_weights": args.r_weights, "stop_numb": args.stop_numb}

actor_feat = {"num_node_features": args.num_node_features + 4, 
              "aggr": args.aggr_a,
              "n_l1":args.n_l_a[0], 
              "n_l2":args.n_l_a[1],
              "n_l3":args.n_l_a[2],
              "n_l4":args.n_l_a[3], 
              "noise_clip": args.noise_clip}

critic_feat = {"num_node_features": args.num_node_features + 4, 
               "aggr": args.aggr_c,
               "n_l1":args.n_l_c[0], 
               "n_l2":args.n_l_c[1],
               "n_l3":args.n_l_c[2],
               "n_l4":args.n_l_c[3], 
               "n_l5":args.n_l_c[4]}

ac_kwards = {"net_actor": GraphConvNet_actor, "net_critic": GraphConvNet_critic, "actor_feat": actor_feat, "critic_feat": critic_feat}

if not os.path.exists(args.path_to_the_main_dir):
    os.makedirs(args.path_to_the_main_dir)

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
    "e_lim" : args.e_lim, 
    "net_lim" : args.net_lim, 
    "save_result" : args.save_result, 
    "start_iter": args.start_iter,
    "test_random": args.test_random,
    "test_every": args.test_every, 
    "save_every": args.save_every, 
    'with_stop': args.with_stop, 
     "N_gr" : args.N_gr, 
     "d_r_max" : args.d_r_max, 
     "f_max" : args.f_max, 
     "noise_level" : args.noise_level, 
     "nfake" : args.nfake}

with open(args.path_to_the_main_dir + "/TD3_Agent_train_arguments.txt", 'w') as f:
    f.write(str(b))
    
TD3_Agent.train(**b)    
