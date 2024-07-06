import torch 
from copy import deepcopy
import numpy as np
from torch.optim import Adam
import time
from utils.replay_memory import ReplayMemory
import itertools
from utils.create_plot import create_plots
import os 
from pymatgen.io.ase import AseAtomsAdaptor
import ase
from ase import Atoms
from torch_geometric.data import Data, Batch
import pandas as pd
from pymatgen.core.structure import Structure
from ase.calculators.lj import LennardJones
import datetime
import e3nn 

torch.set_default_dtype(torch.float64)

from torch import nn 

class Agent(nn.Module): 
    def __init__(self, 
        net_actor, 
        net_critic,
        actor_feat, 
        critic_feat,
                ):
        
        super().__init__()
        
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu")
        self.q1 = net_critic(**critic_feat).to(self.device)
        self.q2 = net_critic(**critic_feat).to(self.device)
        
        self.pi = net_actor(**actor_feat).to(self.device)
        
        
    def act(self, o, noise_scale): 
        with torch.no_grad():
             return self.pi(data = o.to(self.device), noise_scale = noise_scale)

def aver_list(l, n): 
    r = len(l)%n
    c = len(l)-r
    av = np.array(l[:len(l)-r])
    if c !=0:
        av = np.average(np.array(l[:len(l)-r]).reshape(-1, n), axis=1)
    if r != 0: 
        av = np.append(av, sum(l[len(l)-r:])/r)
    return av
    

class TD3Agent: 
    def __init__(self,
                 env_fn, 
                 actor_critic, 
                 env_kwards = dict(), 
                 ac_kwargs = dict(), 
                 seed=0, 
                 replay_size=int(1e6), 
                 gamma=0.99, 
                 polyak=0.995, 
                 pi_lr=1e-3, 
                 q_lr=1e-3, 
                 batch_size=100, 
                 start_steps=10000, 
                 update_after=1000, 
                 update_every=50, 
                 target_noise=0.2, 
                 noise_clip=0.5, 
                 policy_delay=2, 
                 save_freq=1,
                 trans_coef = 0.1, 
                 noise = [0.01, 0.001], 
                 init_rewards_for_weights = None,
                 with_weights = False):
                
        
        torch.manual_seed(seed)
        np.random.seed(seed)
        
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu")
        self.env_fn = env_fn
        self.env =  env_fn(**env_kwards) 
                
        self.ac = actor_critic(**ac_kwargs)
        self.ac_targ = deepcopy(self.ac)
        self.q_params = itertools.chain(self.ac.q1.parameters(), self.ac.q2.parameters())
        
        for p in self.ac_targ.parameters():
            p.requires_grad = False
            
        # Set up optimizers for policy and q-function
        self.pi_optimizer = Adam(self.ac.pi.parameters(), lr=pi_lr)
        self.q_optimizer = Adam(self.q_params, lr=q_lr)
        
        self.memory = ReplayMemory(buffer_capacity=replay_size, batch_size = batch_size)
        
        self.target_noise = target_noise
        self.noise_clip = noise_clip
        self.gamma = gamma
        self.policy_delay  = policy_delay
        self.polyak = polyak
        self.trans_coef = trans_coef
        self.noise = noise
        self.batch_size = batch_size
        self.update_after = update_after
        self.update_every = update_every
        self.start_steps = start_steps 
        self.test_labels = []
        
        self.with_weights = False if len(env_kwards["input_struct_lib"]) ==1 else with_weights
        
        if self.with_weights: 
            if init_rewards_for_weights is not None:
                assert len(init_rewards_for_weights) == len(env_kwards["input_struct_lib"]), 'Len(init_weights) should be the same as len(input_struct_lib)'
                self.rewards_for_weights = np.array(init_rewards_for_weights) 
            else: 
                self.rewards_for_weights = [] 
                for i in range(len(env_kwards["input_struct_lib"])): 
                    o, _, _, _ = self.env.reset(self.trans_coef, i), False, 0, 0
                    _, r, _, _, f, _, _ = self.env.step(self.get_action(o, 0), 0)
#                     self.rewards_for_weights.append(r)
                    self.rewards_for_weights.append(f)
                self.rewards_for_weights = np.array(self.rewards_for_weights)
            self.env.weights = self.rewards_for_weights/self.rewards_for_weights.sum()
        else: 
            L = len(env_kwards["input_struct_lib"]) 
            self.env.weights = np.ones(L)/L 
        
            
    def update_weights(self, r_new, num): 
        self.rewards_for_weights[num] = r_new 
        self.env.weights = self.rewards_for_weights/self.rewards_for_weights.sum()
        
    def compute_loss_q(self,batch):
            
        device = self.device
        o =  Batch.from_data_list(batch["state"].tolist()).to(device) 
        o2 = Batch.from_data_list(batch["next_state"].tolist()).to(device)  
        a = Batch.from_data_list(batch["action"].tolist()).to(device)
        r = torch.FloatTensor(batch["reward"]).to(device)
        d = torch.FloatTensor(batch["done"]).to(device)
        
        q1 = self.ac.q1(o,a)        
        q2 = self.ac.q2(o,a)

        # Bellman backup for Q functions
        with torch.no_grad():
            pi_targ = self.ac_targ.pi(data = o2, noise_scale = self.target_noise)
            # Target Q-values
            q1_pi_targ = self.ac_targ.q1(o2, pi_targ)
            q2_pi_targ = self.ac_targ.q2(o2, pi_targ)
            
            q_pi_targ = torch.min(q1_pi_targ, q2_pi_targ)
            backup = r + self.gamma * (1 - d) * q_pi_targ
        # MSE loss against Bellman backup
        loss_q1 = ((q1 - backup)**2).mean()
        loss_q2 = ((q2 - backup)**2).mean()
        loss_q = loss_q1 + loss_q2

        return loss_q

    def compute_loss_pi(self, batch):
        device = self.device
        o = Batch.from_data_list(batch["state"].tolist()).to(device)
        a_pr = self.ac.pi(o)
        q2_pi = self.ac.q2(o, a_pr)
        return -q2_pi.mean()
    
    def update(self, data, timer):
        return_dict = {"loss_q": None, "loss_pi": None}
        # First run one gradient descent step for Q1 and Q2
        self.q_optimizer.zero_grad()
        loss_q = self.compute_loss_q(data)
        loss_q.backward()
        self.q_optimizer.step()
        
        return_dict["loss_q"] = loss_q.detach().cpu().item()
        
        if (timer+1) % self.policy_delay == 0:
            
            for p in self.q_params:
                p.requires_grad = False
                
            self.pi_optimizer.zero_grad()
            loss_pi = self.compute_loss_pi(data)
            loss_pi.backward()
            self.pi_optimizer.step()
            
            for p in self.q_params:
                p.requires_grad = True
            
            return_dict["loss_pi"] = loss_pi.detach().cpu().item()
            with torch.no_grad():
                for p, p_targ in zip(self.ac.parameters(), self.ac_targ.parameters()): 
                    p_targ.data.mul_(self.polyak)
                    p_targ.data.add_((1 - self.polyak) * p.data)
        return return_dict
    
    def get_action(self, o, noise_scale):
        a = self.ac.act(o = o, noise_scale = noise_scale).to('cpu') 
        return a
    
    def test_agent(self, num_test_episodes, max_test_steps, test_random = False):
        
        self.test_labels = []
        L = 1 if test_random else len(self.env.input_lib.keys())
        N_ep = num_test_episodes*L
        scores = np.zeros(N_ep)
        disc_scores = np.zeros(N_ep)
        last_steps = np.zeros(N_ep)
        forces_last_step = np.zeros(N_ep)
        
        for j in range(N_ep):
            np.random.seed(j)
            it_eps = 0
            num = None if test_random else j % L 
            o, d, ep_ret, ep_disc_ret, ep_len = self.env.reset(self.trans_coef, num), False, 0, 0, 0
            self.test_labels.append(self.env.num)
            while not(d or (ep_len == max_test_steps)):
                o, r, d, _, f, e, _ = self.env.step(self.get_action(o, None), ep_len+1)
                ep_ret += r
                ep_disc_ret += r*(self.gamma**ep_len)
                ep_len += 1
            scores[j] = ep_ret
            last_steps[j] = ep_len
            forces_last_step[j] = f 
            disc_scores[j] = ep_disc_ret
            if self.with_weights: 
                self.update_weights(f, self.env.num)
            
        data_to_save_test = {"Score": scores.mean(), "Last_step": last_steps.mean(), "Maximum_force": forces_last_step.mean(), "Disc_score": disc_scores.mean(),
                            "Score_std": scores.std(), "Last_step_std": last_steps.std(), "Maximum_force_std": forces_last_step.std(), "Disc_score_std": disc_scores.std(), "Test_labels": self.test_labels, 
                            "Score_med": np.median(scores), "Last_step_med": np.median(last_steps), "Maximum_force_med": np.median(forces_last_step), "Disc_score_med": np.median(disc_scores)}
        
        return data_to_save_test
    
    def train(self, train_ep, test_ep, path_to_the_main_dir, env_name, suffix, test_every, start_iter = 0, save_every= 1000, save_memory = 5000, plot_step = None, e_lim = None, net_lim = None, save_result = True, test_random = False, with_stop = False, steps_rel = 50,
             max_norm_max_step = 30, max_norm_limit = 0.015, force_limit = 0.1, noise_level = 10) : 
        pi_losses, q_losses, delta_e_train, max_force, local_reward, sticks = [], [], [], [], [], []  
        t_total = 0 
        
        columns_test = ['Score', 'Last_step', 'Maximum_force', "Disc_score", 'Score_std', 'Last_step_std', "Maximum_force_std", "Disc_score_std" , "Test_labels", 'Score_med', 'Last_step_med', "Maximum_force_med", "Disc_score_med"] 
        columns_train = ["Total_reward", "Last_step_train", "Stop_label_train", "Env_name", "Weights"]
        df_test = pd.DataFrame(None, columns=columns_test)
        df_train = pd.DataFrame(None, columns=columns_train) 
        os.makedirs(path_to_the_main_dir + "/" +'data/') if not os.path.exists(path_to_the_main_dir + "/" + 'data/') else None
        
        for i in range(train_ep[0]): 
            o, ep_ret, ep_len = self.env.reset(self.trans_coef), 0, 0
            max_norm = [] 
            count_max_norm = 0
            for t in range(train_ep[1]):
                    
                if t_total >= self.start_steps:  
                    if count_max_norm == max_norm_max_step: 
                        n_c = self.ac.pi.noise_clip 
                        self.ac.pi.noise_clip  = 20 
                        a = self.get_action(o, noise_level)
                        self.ac.pi.noise_clip = n_c
                        count_max_norm = 0 
                    else: 
                        noise_scale = ((self.noise[1] - self.noise[0])/train_ep[1]) * t + self.noise[0]
                        a = self.get_action(o, noise_scale)
                else: 
                    prev_pos = self.env.current_ase_structure.get_positions() 
                    prev_state = self.env.current_ase_structure.copy()
                    prev_state.calc = self.env.current_ase_structure.calc
                    dyn = BFGS(prev_state)
                    dyn.run(fmax=self.env.eps, steps=steps_rel)
                    clear_output(wait=True)
                    a = Data(x=torch.from_numpy(prev_state.get_positions() - prev_pos))
                t_total +=1 
                o2, r, d, a2, f, d_e, s = self.env.step(a, t+1)
                ep_ret += r
                ep_len += 1
                delta_e_train.append(d_e)
                max_force.append(f)
                local_reward.append(ep_ret)
                
                self.memory.record(o.to('cpu'), a2, r, o2.to('cpu'), d)
                    
                o = o2 

                if t_total >= self.update_after and len(self.memory) >= self.batch_size and t_total % self.update_every == 0:
                    for j in range(self.update_every): 
                        batch = self.memory.sample()
                        losses = self.update(batch, t)
                        pi_losses.append(losses["loss_pi"])
                        q_losses.append(losses["loss_q"])
                
                max_norm.append(a2.x.norm(dim = 1).max().item())
                
                av = min(len(max_norm), 10)
                if np.array(max_norm)[-av:].mean() <= max_norm_limit and f >= force_limit: 
                    count_max_norm += 1 
                else: 
                    count_max_norm = 0
                    
                if plot_step is not None and t_total%plot_step == 0:   
                    data_list = {"Current total reward of the episode": [["Total reward"], [local_reward], None, None, sticks], 
                                 "Losses_Pi": [["Pi"],[np.array(pi_losses)[np.array(pi_losses)!= None]], net_lim, None, (np.array(sticks)/self.policy_delay).astype(int)],
                                "Losses_Q": [["Q"],[aver_list(q_losses, self.policy_delay)], net_lim, None, (np.array(sticks)/self.policy_delay).astype(int)], 
                                "Weights": [["Weights"], [self.env.weights], None, "o", None]}
                    if self.env.reward_func == "energy":
                        data_list["Delta E"]  = [["Delta E"],[delta_e_train], e_lim, None, sticks]
                    else: 
                        data_list["Max force"]  = [["Max force"],[max_force], e_lim, None, sticks]
                    create_plots(data_list)
                        
                if t_total% test_every == 0 and test_ep is not None: 
                    prev_state = self.env.current_ase_structure.copy()
                    prev_calc = self.env.current_ase_structure.calc
                    prev_num = self.env.num
                    data_to_save_test = self.test_agent(test_ep[0], test_ep[1], test_random)
                    self.env.current_ase_structure = prev_state
                    self.env.num = prev_num
                    self.env.current_ase_structure.calc = prev_calc
                    df_test = df_test.append(data_to_save_test, ignore_index=True)
                    df_test.to_csv(f"{path_to_the_main_dir}/data/df_{env_name}_{suffix}_test_si{start_iter}.csv") 
                        
#                 if (t_total+1)% save_memory == 0:
#                         self.memory.save_buffer(path_to_the_main_dir)          
                        
                if t_total % save_every == 0: 
                    self.save_model(path_to_the_main_dir, env_name, f"{i + start_iter}")

                    np.save(f"{path_to_the_main_dir}/pi_losses.npy",  pi_losses)
                    np.save(f"{path_to_the_main_dir}/q_losses.npy",  q_losses)

                    if save_result: 
                        name = f"_train_start_iter{start_iter}.png"  
                        last_step_done,last_step_stop = [],[]
                        for key, item in zip(df_train["Stop_label_train"].values,df_train["Last_step_train"].values): 
                            if key: 
                                last_step_stop.append(item)
                                last_step_done.append(None)
                            else: 
                                last_step_done.append(item)
                                last_step_stop.append(None)
                        data_list = {"Total reward of the episode": [["Total reward"], [df_train["Total_reward"].values], None, None, None], 
                                                 "Losses_Pi": [["Pi"],[np.array(pi_losses)[np.array(pi_losses)!= None]], net_lim, None, (np.array(sticks)/self.policy_delay).astype(int)],
                                            "Losses_Q": [["Q"],[aver_list(q_losses, self.policy_delay)], net_lim, None, (np.array(sticks)/self.policy_delay).astype(int)], 
                                              "Last step of the episode": [["Last step done", "Last step stop"], [last_step_done, last_step_stop], None, "o", None],
                                     "Weights": [["Weights"], [self.env.weights], None, "o", None]
                                    }
                        if train_ep[0]==1:
                            data_list["Total reward of the episode"][1][0] = local_reward

                        if self.env.reward_func == "energy":
                            data_list["Delta E"]  = [["Delta E"],[delta_e_train], e_lim, None, sticks]
                        else: 
                            data_list["Max force"]  = [["Max force"],[max_force], e_lim, None, sticks]

                        create_plots(data_list = data_list, save = True, show = False, suffix = suffix, path_to_the_main_dir = path_to_the_main_dir, env_name = env_name, name = name)
                        
                        if test_ep is not None: 
                            name = f"_test_start_iter{start_iter}.png"
                            keys = df_test['Score'].values != None
                            data_list = {"Disc_score":[["Disc_score"],[df_test['Disc_score'].values[keys]], None, None, None], "Last step test": [["Last step test"],[df_test['Last_step'].values[keys]], None, None, None]}
                            if self.env.reward_func == "energy":
                                data_list["Energy change"]  = [["Difference in energy"],[df_test['Difference_in_energy'].values[keys]], e_lim, None, None]
                            else: 
                                data_list["Max force"]  = [["Max force"],[df_test['Maximum_force'].values[keys]], e_lim, None, None, None]

                            create_plots(data_list = data_list, save = True, show = False, suffix = suffix, path_to_the_main_dir = path_to_the_main_dir, env_name = env_name, name = name)
                        
                if with_stop: 
                    if d or s:
                        sticks.append(t_total-1)
                        break
                else:
                    if d: 
                        break 
                if t + 1 == train_ep[1]: 
                    sticks.append(t_total-1)
                    
#             if self.with_weights: 
# #                 self.update_weights(ep_ret/ep_len, self.env.num)
#                 self.update_weights(f, self.env.num)
            data_to_save_train = {"Total_reward":ep_ret, "Last_step_train":ep_len, "Stop_label_train":s, 
                                  "Env_name":self.env.current_ase_structure.get_chemical_formula() + "_" + str(self.env.num), "Weights": self.env.weights}     
            df_train = df_train.append(data_to_save_train, ignore_index=True)
            df_train.to_csv(f"{path_to_the_main_dir}/data/df_{env_name}_{suffix}_train_si{start_iter}.csv") 
    
    def reset_env(self, new_env_kwards): 
        self.env =  self.env_fn(**new_env_kwards)
        
    def save_model(self, path_to_the_main_dir, env_name, suffix=""):
        if not os.path.exists(path_to_the_main_dir +"/checkpoints"):
            os.makedirs(path_to_the_main_dir + "/checkpoints")
        ckpt_path = path_to_the_main_dir + "/checkpoints/" +  "td3_checkpoint_{}_{}".format(env_name, suffix)
        print('Saving models to {}'.format(ckpt_path)) 
        
        torch.save({'ac_pi': self.ac.pi.state_dict(),
                    'ac_pi_t' : self.ac_targ.pi.state_dict(),
                    'ac_q1': self.ac.q1.state_dict(),
                    'ac_q2': self.ac.q2.state_dict(),
                    'ac_q1_t': self.ac_targ.q1.state_dict(),
                    'ac_q2_t': self.ac_targ.q2.state_dict(),
                    'pi_optim': self.pi_optimizer.state_dict(),
                    'q_optim': self.q_optimizer.state_dict()}, ckpt_path)
        return ckpt_path
    
    def load_model(self, ckpt_path):
        print('Loading models from {}'.format(ckpt_path))
        if ckpt_path is not None:
            checkpoint = torch.load(ckpt_path)
            self.ac.pi.load_state_dict(checkpoint['ac_pi'])
            self.ac_targ.pi.load_state_dict(checkpoint['ac_pi_t'])
            self.ac.q1.load_state_dict(checkpoint['ac_q1'])
            self.ac.q2.load_state_dict(checkpoint['ac_q2'])
            self.ac_targ.q1.load_state_dict(checkpoint['ac_q1_t'])
            self.ac_targ.q2.load_state_dict(checkpoint['ac_q2_t'])
            self.q_optimizer.load_state_dict(checkpoint['q_optim'])
            self.pi_optimizer.load_state_dict(checkpoint['pi_optim'])
            
    def load_actor(self, ckpt_path):
        print('Loading models from {}'.format(ckpt_path))
        if ckpt_path is not None:
            checkpoint = torch.load(ckpt_path)
            self.ac.pi.load_state_dict(checkpoint['ac_pi'])
            self.ac_targ.pi.load_state_dict(checkpoint['ac_pi_t'])
            self.pi_optimizer.load_state_dict(checkpoint['pi_optim'])
            
    def load_critic(self, ckpt_path):
        print('Loading models from {}'.format(ckpt_path))
        if ckpt_path is not None:
            checkpoint = torch.load(ckpt_path)
            self.ac.q1.load_state_dict(checkpoint['ac_q1'])
            self.ac.q2.load_state_dict(checkpoint['ac_q2'])
            self.ac_targ.q1.load_state_dict(checkpoint['ac_q1_t'])
            self.ac_targ.q2.load_state_dict(checkpoint['ac_q2_t'])
            self.q_optimizer.load_state_dict(checkpoint['q_optim'])
            
    def load_pretrained_actor(self, ckpt_path):
        print('Loading models from {}'.format(ckpt_path))
        if ckpt_path is not None:
            checkpoint = torch.load(ckpt_path)
            self.ac.pi.load_state_dict(checkpoint['model'])
            self.ac_targ.pi.load_state_dict(checkpoint['model'])
            
    def test_agent_sep(self, num_test_episodes, max_test_steps, test_random = False):
        
        self.test_labels = []
        L = 1 if test_random else len(self.env.input_lib.keys())
        N_ep = num_test_episodes*L
        scores = np.zeros(N_ep)
        disc_scores = np.zeros(N_ep)
        last_steps = np.zeros(N_ep)
        forces_last_step = np.zeros(N_ep)
        
        for j in range(N_ep):
            np.random.seed(j)
            it_eps = 0
            num = None if test_random else j % L 
            o, d, ep_ret, ep_disc_ret, ep_len = self.env.reset(self.trans_coef, num), False, 0, 0, 0
            self.test_labels.append(self.env.num)
            while not(d or (ep_len == max_test_steps)):
                o, r, d, _, f, e, _ = self.env.step(self.get_action(o, None), ep_len+1)
                ep_ret += r
                ep_disc_ret += r*(self.gamma**ep_len)
                ep_len += 1
            scores[j] = ep_ret
            last_steps[j] = ep_len
            forces_last_step[j] = f 
            disc_scores[j] = ep_disc_ret
            
        data_to_save_test = {"Score": scores, "Last_step": last_steps, "Maximum_force": forces_last_step, "Disc_score": disc_scores}
                            
        return data_to_save_test
