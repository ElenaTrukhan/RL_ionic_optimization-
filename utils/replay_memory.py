import numpy as np 
import torch
import os 

class ReplayMemory:
    
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
        
    def sample(self): 
        high = min(self.buffer_counter, self.buffer_capacity)
        idxes = np.random.randint(0, high, self.batch_size)
        batch = dict(state=self.state_buffer[idxes], 
                    action=self.action_buffer[idxes], 
                    reward=self.reward_buffer[idxes], 
                    next_state=self.next_state_buffer[idxes],
                    done = self.done_buffer[idxes])
#         return {k: torch.as_tensor(v, dtype=torch.float32) for k,v in batch.items()}
        return batch 
    def __len__(self):
        return min(self.buffer_counter, self.buffer_capacity)
    
    def save_buffer(self, path_to_the_main_dir): 
        path = path_to_the_main_dir + "/" + 'memory/'
        if not os.path.exists(path):
            os.makedirs(path)
        np.save(path+'state.npy', self.state_buffer)
        np.save(path+'action.npy', self.action_buffer)
        np.save(path+'reward.npy', self.reward_buffer)
        np.save(path+'next_state.npy', self.next_state_buffer)
        np.save(path+'done.npy', self.done_buffer) 

    def load_buffer(self, path_to_load): 
        self.state_buffer = np.load(path_to_load+'/state.npy', allow_pickle = True)
        self.action_buffer = np.load(path_to_load+'/action.npy', allow_pickle = True)
        self.reward_buffer = np.load(path_to_load+'/reward.npy', allow_pickle = True)
        self.next_state_buffer = np.load(path_to_load+'/next_state.npy', allow_pickle = True)
        self.done_buffer =   np.load(path_to_load+'/done.npy', allow_pickle = True)
        for it, item in enumerate(self.state_buffer): 
            if item is None: 
                self.buffer_counter = it
                break
       