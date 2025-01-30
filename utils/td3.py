import torch 
from copy import deepcopy
import numpy as np
from torch.optim import Adam
from utils.replay_memory import ReplayMemory
import itertools
from utils.utils import create_plots
import os 
from torch_geometric.data import Batch
import pandas as pd
from torch import nn 
from utils.utils import aver_list

class Agent(nn.Module):
    r"""The class of TD3 Agent.  

    Parameters
    ----------
    net_actor : Class
        Class for Actor. 

    net_critic: 
        Class for Critic.

    actor_feat: dict 
        Actor paramters. 
    
    critic_feat: dict 
        Critic paramters. 

    """
 
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

        r"""Predict action for a given state.  

        Parameters
        ----------
        o : Input state 

        noise_scale : float 
            Noise level for smoothing noise added to policy. 

        Returns
        ----------
            Action 
        """
        with torch.no_grad():
             return self.pi(data = o.to(self.device), noise_scale = noise_scale)    

class TD3Agent: 
    r"""The class of TD3 Agent for structure relaxation process.  

    Parameters
    ----------
    env_fn : Class
        Class for Environment describing structure relaxation process. 

    env_kwards: dict 
        Environment paramters. 
    
    ac_kwargs: dict 
        TD3 Agent paramters. 
    
    seed: int 
        Seed for random number generators.
    
    replay_size: int 
        Maximum length of replay buffer. 

    gamma: float 
        Discount factor. 
    
    polyak: float 
        Interpolation factor in polyak averaging for target networks.
    
    pi_lr: float 
         Learning rate for Actor.
        
    q_lr: float 
         Learning rate for Critic.
    
    batch_size: int
        Minibatch size for SGD.
    
    start_steps: int
        Number of steps for fake action selection, before running real policy. 
    
    update_after: int
        Number of env interactions to collect before starting to do gradient descent updates.
    
    update_every: int
        Number of env interactions that should elapse between gradient descent updates.
    
    target_noise: float
        Noise level for smoothing noise added to target policy. 

    noise_clip: float
        Limit for absolute value of target policy smoothing noise.
    
    policy_delay: int 
        Policy will only be updated once every policy_delay times for each update of the Q-networks.
    
    trans_coef: float 
        Distortion parameter that determines how far the structure is shifted from the local minimum before the onset of the relaxation episode.
    
    noise: list 
        Noise level for exploration noise added to policy at training time. 

    init_rewards_for_weights: list 
        Weights corresponding to the structures in the input dataset that represent their priority for selection during training.

    with_weights: bool
        Flag indicating whether structures should be selected during training according to weights or uniformly. 

    """
    def __init__(self,
                 env_fn, 
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
                 trans_coef = 0.1, 
                 noise = [0.01, 0.001], 
                 init_rewards_for_weights = None,
                 with_weights = False):
                
        
        torch.manual_seed(seed)
        np.random.seed(seed)
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu")
        self.env =  env_fn(**env_kwards) 
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

        # Create actor-critic module and target networks
        self.ac = Agent(**ac_kwargs)
        self.ac_targ = deepcopy(self.ac)
        self.q_params = itertools.chain(self.ac.q1.parameters(), self.ac.q2.parameters())

        # Freeze target networks with respect to optimizers (only update via polyak averaging)
        for p in self.ac_targ.parameters():
            p.requires_grad = False
            
        # Set up optimizers for Actor and Critic
        self.pi_optimizer = Adam(self.ac.pi.parameters(), lr=pi_lr)
        self.q_optimizer = Adam(self.q_params, lr=q_lr)

        # Replay Buffer
        self.memory = ReplayMemory(buffer_capacity=replay_size, batch_size = batch_size)
        
        # Initialization of priority weights of structures for training
        self.with_weights = False if len(env_kwards["input_struct_lib"]) == 1 else with_weights
        if self.with_weights: 
            if init_rewards_for_weights is not None:
                assert len(init_rewards_for_weights) == len(env_kwards["input_struct_lib"]), 'Len(init_weights) should be the same as len(input_struct_lib)'
                self.rewards_for_weights = np.array(init_rewards_for_weights) 
            else: 
                self.rewards_for_weights = [] 
                for i in range(len(env_kwards["input_struct_lib"])): 
                    o, _, _, _ = self.env.reset(self.trans_coef, i), False, 0, 0
                    _, _, _, _, f, _ = self.env.step(self.get_action(o, 0), 0)
                    self.rewards_for_weights.append(f)
                self.rewards_for_weights = np.array(self.rewards_for_weights)
            self.env.weights = self.rewards_for_weights/self.rewards_for_weights.sum()
        else: 
            L = len(env_kwards["input_struct_lib"]) 
            self.env.weights = np.ones(L)/L 
        
    def compute_loss_q(self, batch):
        r"""Calculate loss for Critic.

        Parameters
        ----------
        batch : `torch_geometric.data.Batch`
            Minibatch for SGD
        
        Returns
        ----------
        float
            Critic loss  
        """

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
        r"""Calculate loss for Actor.

        Parameters
        ----------
        batch : `torch_geometric.data.Batch`
            Minibatch for SGD
        
        Returns
        ----------
        float
            Actor loss  
        """
        device = self.device
        o = Batch.from_data_list(batch["state"].tolist()).to(device)
        a_pr = self.ac.pi(o)
        q2_pi = self.ac.q2(o, a_pr)
        return -q2_pi.mean()
    
    def update(self, data, timer):
        r"""Update Actor and Critic models 

        Parameters
        ----------
        data : `torch_geometric.data.Batch`
            Minibatch for SGD
        
        timer : int 
            Number of training step 
        Returns
        ----------
        dict 
            Dictionary with losses  
        """
        return_dict = {"loss_q": None, "loss_pi": None}
        # Run one gradient descent step for Q1 and Q2
        self.q_optimizer.zero_grad()
        loss_q = self.compute_loss_q(data)
        loss_q.backward()
        self.q_optimizer.step()
        return_dict["loss_q"] = loss_q.detach().cpu().item()
        
        # Possibly update pi and target networks
        if (timer+1) % self.policy_delay == 0:
            
            # Freeze Q-networks 
            for p in self.q_params:
                p.requires_grad = False
            
            # One gradient descent step for pi.
            self.pi_optimizer.zero_grad()
            loss_pi = self.compute_loss_pi(data)
            loss_pi.backward()
            self.pi_optimizer.step()
            return_dict["loss_pi"] = loss_pi.detach().cpu().item()

            # Unfreeze Q-networks
            for p in self.q_params:
                p.requires_grad = True

            # Update target networks by polyak averaging.
            with torch.no_grad():
                for p, p_targ in zip(self.ac.parameters(), self.ac_targ.parameters()): 
                    p_targ.data.mul_(self.polyak)
                    p_targ.data.add_((1 - self.polyak) * p.data)

        return return_dict
    
    def get_action(self, o, noise_scale):
        r"""Predict atomic shifts for a given structure 

        Parameters
        ----------
        o : `torch_geometric.data.Data`
            Crystal graph of a given structure.
        
        noise_scale : int 
            Noise level for smoothing noise added to policy. 
        Returns
        ----------
        `torch_geometric.data.Data` 
            Crystal graph with atomic shifts
        """
        a = self.ac.act(o = o, noise_scale = noise_scale).detach().to('cpu') 
        return a
    
    def update_weights(self, r_new, num): 
        
        r"""Update priority weights of structures for training. 

        Parameters
        ----------
        r_new : float
            The value used as a criteria for prioritization for a given structure with number num 
        num : int 
            Number of the structure from input dataset
        """
        self.rewards_for_weights[num] = r_new 
        self.env.weights = self.rewards_for_weights/self.rewards_for_weights.sum()

    def test_agent(self, num_test_episodes, max_test_steps, test_random = False, separate = False):

        r"""Test Agent.

        Parameters
        ----------
        num_test_episodes : int
            Number of testing episodes.
        
        max_test_steps : int 
            Maximum number of steps in one relaxation episode. 
        
        test_random: bool 
            Flag indicating whether to randomly select num_test_episodes structures to relax from the input dataset or to relax them one by one in order num_test_episodes times each.  
        
        separate: bool 
            Flag indicating whether to avarage the results over all testing episodes.  
        Returns
        ----------
        dict 
            Dictionary with statistics on scores, number of relaxation steps, maximum force at the last relaxation step, discounted scores collected during testing. 
        """
        prev_state = self.env.current_ase_structure.copy()
        prev_calc = self.env.current_ase_structure.calc
        prev_num = self.env.num

        self.test_labels = []
        L = 1 if test_random else len(self.env.input_lib.keys())
        N_ep = num_test_episodes*L
        scores = np.zeros(N_ep)
        disc_scores = np.zeros(N_ep)
        last_steps = np.zeros(N_ep)
        forces_last_step = np.zeros(N_ep)
        
        for j in range(N_ep):
            np.random.seed(j)
            num = None if test_random else j % L 

            # Reset the structure to be relaxed
            o, d, ep_ret, ep_disc_ret, ep_len = self.env.reset(self.trans_coef, num, correct = False), False, 0, 0, 0
            self.test_labels.append(self.env.num)

            # Relaxation 
            while not(d or (ep_len == max_test_steps)):
                o, r, d, _, f, _ = self.env.step(self.get_action(o, None))
                ep_ret += r
                ep_disc_ret += r*(self.gamma**ep_len)
                ep_len += 1
            scores[j] = ep_ret
            last_steps[j] = ep_len
            forces_last_step[j] = f 
            disc_scores[j] = ep_disc_ret
            if self.with_weights: 
                self.update_weights(f, self.env.num)
        
        if separate: 
            data_to_save_test = {"Score": scores, "Last_step": last_steps, "Maximum_force": forces_last_step, "Disc_score": disc_scores}
        else: 
            data_to_save_test = {"Score": scores.mean(), "Last_step": last_steps.mean(), "Maximum_force": forces_last_step.mean(), "Disc_score": disc_scores.mean(),
                                "Score_std": scores.std(), "Last_step_std": last_steps.std(), "Maximum_force_std": forces_last_step.std(), "Disc_score_std": disc_scores.std(), "Test_labels": self.test_labels, 
                                "Score_med": np.median(scores), "Last_step_med": np.median(last_steps), "Maximum_force_med": np.median(forces_last_step), "Disc_score_med": np.median(disc_scores)}
        self.env.current_ase_structure = prev_state
        self.env.num = prev_num
        self.env.current_ase_structure.calc = prev_calc
        return data_to_save_test
    
    def train(self, train_ep, test_ep, path_to_the_main_dir, env_name, test_every, start_iter = 0, save_every= 1000, e_lim = None, net_lim = None, 
              save_result = True, test_random = False, N_gr = 30, d_r_max = 0.015, f_max = 0.1, noise_level = 10, nfake = 10): 
        
        r"""Train Agent.

        Parameters
        ----------
        train_ep/test_ep: list
            [Number of training/testing episodes, Maximum number of steps in one training/testing episode].
        
        path_to_the_main_dir: str 
            Path to the folder where the results are saved. 

        env_name : str
            Name of the experiment.
        
        test_every : int 
            Agent is tested every test_every training steps.
        
        start_iter : int 
            The initial iteration number. Should be specified if the model is saved in the same directory as the experiment with the same name to prevent overwriting the results.

        save_every: int 
            Model weights and plots are saved every save_every training steps.

        e_lim/net_lim: list 
            y-limits for plots of maximum forces/losses. 

        save_result: bool 
            Flag indicating whether learning curves are plotted and saved.

        test_random: bool 
            test_random parameter for test function.  
        
        N_gr, d_r_max, f_max: float
            N_gr, Delta r_max, f_max parameters in additional greedy exploration. 
        
        noise_level: float 
            Noise level in additional greedy exploration. 
        
        nfake: int 
            Fake episodes are selected and added to the replay buffer every nfake training steps. Note: Fake exploration can improve model performance, 
            but if nfake is too small, it can also lead to instability since the experience in the replay buffer will not correlate with the current policy.
        """
        
        pi_losses, q_losses, max_force, local_reward, sticks = [], [], [], [], [] 
        t_total = 0 
        df_test = pd.DataFrame(None, columns=['Score', 'Last_step', 'Maximum_force', "Disc_score", 'Score_std', 'Last_step_std', "Maximum_force_std", "Disc_score_std" , "Test_labels", 'Score_med', 'Last_step_med', "Maximum_force_med", "Disc_score_med"])
        df_train = pd.DataFrame(None, columns=["Total_reward", "Last_step_train", "Stop_label_train", "Env_name", "Weights"]) 
        os.makedirs(path_to_the_main_dir + "/" +'data/', exist_ok = True)
        
        for i in range(train_ep[0]): 

            # Reset structure to be relaxed
            o, ep_ret, ep_len = self.env.reset(self.trans_coef), 0, 0
            max_norm = [] 
            c_gr = 0

            # Training relaxation
            for t in range(train_ep[1]):
                if t_total >= self.start_steps:  
                    if c_gr == N_gr: 
                        # Additional greedy exploration 
                        self.ac.pi.noise_clip  = noise_level*2 
                        a = self.get_action(o, noise_level)
                        self.ac.pi.noise_clip = self.noise_clip
                        c_gr = 0 
                    else: 
                        a = self.get_action(o, ((self.noise[1] - self.noise[0])/train_ep[1]) * t + self.noise[0])
                    o2, r, d, a2, f, s = self.env.step(a)
                else: 
                    #   Fake steps in the beginning of training 
                    o2, r, d, a2, f = self.env.fake_step()
                    s = False
                t_total +=1 
                ep_ret += r
                ep_len += 1
                max_force.append(f)
                local_reward.append(ep_ret)
                
                # Store transition in Replay Buffer
                self.memory.record(o.to('cpu'), a2, r, o2.to('cpu'), d)

                # Sample fake step 
                if (t+1) % nfake == 0: 
                    o2_f, r_f, d_f, a_f, _ = self.env.fake_step()
                    self.memory.record(o.to('cpu'), a_f, r_f, o2_f.to('cpu'), d_f)
                    
                o = o2 

                # Update models 
                if t_total >= self.update_after and len(self.memory) >= self.batch_size and t_total % self.update_every == 0:
                    for j in range(self.update_every): 
                        batch = self.memory.sample()
                        losses = self.update(batch, t)
                        pi_losses.append(losses["loss_pi"])
                        q_losses.append(losses["loss_q"])
                
                # Check if the counter for additional greedy exploration should be increased
                max_norm.append(a2.x.norm(dim = 1).max().item())
                if np.array(max_norm)[-min(len(max_norm), 10):].mean() <= d_r_max and f >= f_max: 
                    c_gr += 1 
                else: 
                    c_gr = 0

                # Testing    
                if t_total% test_every == 0 and test_ep is not None: 
                    data_to_save_test = self.test_agent(test_ep[0], test_ep[1], test_random)
                    df_test = df_test.append(data_to_save_test, ignore_index=True)
                    df_test.to_csv(f"{path_to_the_main_dir}/data/df_{env_name}_test_si{start_iter}.csv") 
                
                # Save model weights and plots
                if t_total % save_every == 0: 
                    self.save_results(path_to_the_main_dir, env_name, i, start_iter, save_result, df_train, df_test, pi_losses, q_losses , net_lim, e_lim, sticks, max_force, local_reward, train_ep, test_ep)
                
                if d or s:
                    sticks.append(t_total-1)
                    break
                if t + 1 == train_ep[1]: 
                    sticks.append(t_total-1)
                    
            data_to_save_train = {"Total_reward":ep_ret, "Last_step_train":ep_len, "Stop_label_train":s, "Env_name":self.env.current_ase_structure.get_chemical_formula() + "_" + str(self.env.num), "Weights": self.env.weights}     
            df_train = df_train.append(data_to_save_train, ignore_index=True)
            df_train.to_csv(f"{path_to_the_main_dir}/data/df_{env_name}_train_si{start_iter}.csv") 
    
    def save_results(self,path_to_the_main_dir, env_name, i, start_iter, save_result, df_train, df_test, pi_losses, q_losses , net_lim, e_lim, sticks, max_force, local_reward, train_ep, test_ep): 
        self.save_model(path_to_the_main_dir, env_name, f"{i + start_iter}")
        if save_result:       
            name = f"_train_start_iter{start_iter}.png"
            last_step_done, last_step_stop = [],[]  
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
                        "Weights": [["Weights"], [self.env.weights], None, "o", None], "Max force": [["Max force"],[max_force], e_lim, None, sticks]}
            if train_ep[0]==1:
                data_list["Total reward of the episode"][1][0] = local_reward
            create_plots(data_list = data_list, save = True, show = False,  path_to_the_main_dir = path_to_the_main_dir, env_name = env_name, name = name)
            if test_ep is not None: 
                name = f"_test_start_iter{start_iter}.png"
                keys = df_test['Score'].values != None
                data_list = {"Disc_score":[["Disc_score"],[df_test['Disc_score'].values[keys]], None, None, None], 
                            "Last step test": [["Last step test"],[df_test['Last_step'].values[keys]], None, None, None],
                            "Max force":[["Max force"],[df_test['Maximum_force'].values[keys]], e_lim, None, None, None]}
                create_plots(data_list = data_list, save = True, show = False, path_to_the_main_dir = path_to_the_main_dir, env_name = env_name, name = name)
        
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
            
    