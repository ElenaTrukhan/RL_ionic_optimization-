import numpy as np 
import torch
import os 

class ReplayMemory:
    r"""The class for Replay Buffer.  

    Parameters
    ----------
    buffer_capacity : int
        Capacity of the Replay Buffer. 

    batch_size: 
        Minibatch size for SGD. 

    reward_func: str
        Reward function type. Possible options: force, log_force, step, hybrid 
    
    convert_to_graph_func: 
        Function used to convert pymatgen Structure into Crystal graph 
    
    r0: float
        Minimum possible distance between atoms. During relaxations, atoms shift back if the distance between them is less than r0
    
    eps: float 
        Force threshold, eV/A
    
    stop_numb: int 
        Maximum number of relaxation steps during which the Agent is allowed to perform actions that are corrected at each step, because the atoms are shifted too close to each other or too far apart.
        The minimum distance is controlled by the r0 parameter, and the maximum distance is adjusted by the params["radius"] parameter used in convert_to_graph_func. 

    r_weights: list 
        Weights for the hybrid reward function. 
    """
    def __init__(self, buffer_capacity=100000, batch_size = 24):
        # Number of "experiences" to store at max
        self.buffer_capacity = buffer_capacity
        self.batch_size = batch_size

        # Its tells us num of times record() was called.
        self.buffer_counter = 0

        # Instead of list of tuples as the exp.replay concept go
        # We use different np.arrays for each tuple element
        self.state_buffer = np.array([None for i in range(0, self.buffer_capacity)])
        self.action_buffer = np.array([None for i in range(0, self.buffer_capacity)])
        self.reward_buffer = np.zeros(buffer_capacity, dtype=np.float32)
        self.next_state_buffer = np.array([None for i in range(0, self.buffer_capacity)])
        self.done_buffer = np.zeros(buffer_capacity, dtype=np.bool_)
        
    # Takes (s,a,r,s') obervation tuple as input
    def record(self, state, action, rew, next_state, done):
        # Set index to zero if buffer_capacity is exceeded,
        # replacing old records
        index = self.buffer_counter % self.buffer_capacity

        self.state_buffer[index] = state
        self.action_buffer[index] = action
        self.reward_buffer[index] = rew
        self.next_state_buffer[index] = next_state
        self.done_buffer[index] = done

        self.buffer_counter += 1

    # Sample a batch for models update     
    def sample(self): 
        high = min(self.buffer_counter, self.buffer_capacity)
        idxes = np.random.randint(0, high, self.batch_size)
        batch = dict(state=self.state_buffer[idxes], 
                    action=self.action_buffer[idxes], 
                    reward=self.reward_buffer[idxes], 
                    next_state=self.next_state_buffer[idxes],
                    done = self.done_buffer[idxes])
        return batch 
    
    def __len__(self):
        return min(self.buffer_counter, self.buffer_capacity)
    
       