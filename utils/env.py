import ase 
from ase import Atoms
from ase.optimize import BFGS
from pymatgen.core.structure import Structure
from pymatgen.io.ase import AseAtomsAdaptor
from ase.calculators.lj import LennardJones
import numpy as np
import torch 
from copy import deepcopy
import pandas as pd 
from utils.calcs_func import func_for_calc


params = {
    "radius": 5, # cut-off radius
    "max_num_nbr": 30, # maximum number of neighbors to consider
    "dmin": 0, # min for Gaussian distance
    "dmax": 5, # max for Gaussian distance
    "step": 0.2 # step for Gaussian distance
}

from ase.calculators.eam import EAM
EAM_Al = EAM(potential = "EAM/Al.eam.alloy")
EAM_AlFe = EAM(potential = "EAM/AlFe.fs")
EAM_Fe = EAM(potential = "EAM/Fe.eam.fs")

def get_sturct_lib(name): 
    df = pd.read_csv(name)
    lib = [] 
    for item, nsite in zip(df["structure"],df["nsites"]):
        struct = Structure.from_str(item, fmt= "cif")
        if nsite == 1: 
            struct.make_supercell([2, 1, 1])
        if nsite> 4 :
            continue
        lib.append(struct)
    return lib


def get_sturct_lib_and_calcs_gen(name):
    df = pd.read_csv(name) 
    lib,calcs = [], []
    for item, nsite in zip(df["structure"],df["nsites"]):
        struct = Structure.from_str(item, fmt= "cif")
        if nsite == 1: 
            struct.make_supercell([2, 1, 1])
        calcs.append(func_for_calc(struct))
        lib.append(struct)
    return lib, calcs

def get_func_true2(cut_state, r0, rmax = params["radius"]): 
    cut_state.wrap()
    nbrs = AseAtomsAdaptor.get_structure(cut_state).get_all_neighbors(r0) 
    cond1 = True 
    for item in nbrs: 
        if len(item) > 0: 
            cond1 = False
            break 
    if not cond1: 
        return False
    cond2 = False
    nbrs2 = AseAtomsAdaptor.get_structure(cut_state).get_all_neighbors(rmax)
    for item in nbrs2: 
        if len(item) > 0: 
            cond2 = True 
            break
    return cond2

def get_func_true(state, rmin, rmax = params["radius"]): 
    state.wrap()
    state_str = AseAtomsAdaptor.get_structure(state)
    rmax_cond = False
    n = len(state_str.sites)
    for i in range(n): 
        for j in range(i, n): 
            ind = 1 if i==j else None
            r = state_str.get_distance(i,j,ind)
            if r < rmin: 
                return False 
            if r <= rmax: 
                rmax_cond = True 
    return rmax_cond

# поставить counter, чтобы обрывать 
def correct_action(cut_state, action, r0, counter_max = 20, eps = 1e-3, order = 4): 
   
    cond1 = get_func_true(cut_state, r0)
    if cond1: 
        return(0)   
    else: 
        counter, a_low, a_high = 0, 0, 1
        a_mid = (a_low + a_high)/2
        while round(a_high - a_low, order) > eps:
            state_ase = cut_state.copy()   
            back_trans = -a_mid*action
            state_ase.translate(back_trans)
            state_ase.wrap()
            cond1 = get_func_true(state_ase, r0)
            if cond1: 
                a_high = a_mid
            else: 
                a_low = a_mid    
            a_mid = (a_low + a_high)/2
            counter += 1 
            if counter > counter_max: 
                break 
    return(a_high)



class Environment: 
    def __init__(self, 
                 input_struct_lib, 
                 convert_to_graph_func,  
                 reward_func = "force",
                 calculator_lib = None, 
                 r0:float = 0.1,
                 eps: float = 1e-6,
                 EMAX: float = 0.1,
                 lamb: float = 1,
                 stop_numb: int = 50, 
                 r_weights = None,
                ): 
        
        self.input_lib = {}
        if calculator_lib is None: 
            calculator_lib = [LennardJones()]*len(input_struct_lib)
        self.to_graph = convert_to_graph_func
        
        if reward_func == "hybrid": 
            assert r_weights is not None 
            assert len(r_weights) == 3
            self.r_weights = r_weights
        self.eps = eps
        self.r0 = r0
        self.lamb = lamb
        self.EMAX = EMAX
        self.stop_count = 0 
        self.stop_max_count = stop_numb
        self.reward_func = reward_func
        
        if reward_func == "energy": 
            self.true_energy_lib = {}
        
        for it, struct_calc in enumerate(zip(input_struct_lib, calculator_lib)): 
            struct, calc = struct_calc
            struct_ase = AseAtomsAdaptor.get_atoms(struct)
            struct_ase.calc = calc
            relax = BFGS(struct_ase)
            relax.run(fmax=eps)
            self.input_lib[it] = [struct_ase, calc]
            
            if reward_func == "energy":
                self.true_energy_lib[it] =  struct_ase.get_potential_energy()        
        self.len = it + 1
        self.num = 0
        self.current_structure = None
        self.current_ase_structure = None
        self.weights = None
        
        
    def reset(self, trans_coef, num = None): 
        self.num = num if num is not None else np.random.choice(self.len, 1, p=self.weights)[0]
        input_ase_struct, calc = self.input_lib[self.num]
        
        self.current_ase_structure = input_ase_struct.copy() 
        trans = trans_coef*np.random.rand(self.current_ase_structure.get_positions().shape[0], 3)
        self.current_ase_structure.translate(trans)
        a_back = correct_action(self.current_ase_structure, trans, self.r0)
        if a_back != 0:
            self.current_ase_structure.translate(-a_back*trans)  
        self.current_structure = AseAtomsAdaptor.get_structure(self.current_ase_structure)
        if self.reward_func == "energy":
            self.true_energy = self.true_energy_lib[self.num]
        self.current_ase_structure.calc = calc
        forces = self.current_ase_structure.get_forces()
        struct_graph = self.to_graph(self.current_structure, forces)
        self.stop_count = 0 
        return struct_graph
    
    def step(self, action, step): 
        init_positions = self.current_ase_structure.get_positions()
        self.current_ase_structure.translate(action.x.cpu())
        self.current_ase_structure.wrap()
        next_positions = self.current_ase_structure.get_positions()
        actual_action = next_positions - init_positions
        stop = False 
        a_back = correct_action(self.current_ase_structure, actual_action, self.r0)
        if a_back != 0:
            self.current_ase_structure.translate(-a_back*actual_action)
            action.x += -a_back*torch.tensor(actual_action)
            if round(a_back, 4) == 1: 
                self.stop_count += 1
                if self.stop_count > self.stop_max_count: 
                    stop = True
                    self.stop_count = 0 
            else: 
                self.stop_count = 0 
        
        self.current_structure = AseAtomsAdaptor.get_structure(self.current_ase_structure)
        forces = self.current_ase_structure.get_forces()
        next_struct_graph = self.to_graph(self.current_structure, forces)
        max_f = max((forces**2).sum(axis=1)**0.5)
        
        if self.reward_func == "energy":
            delta_energy = self.current_ase_structure.get_potential_energy() - self.true_energy
        else: 
            delta_energy = None
            
        done = max_f <= self.eps
        
        if self.reward_func == "log_force":
            reward = -np.log10(max_f) - self.lamb*abs(action.x.cpu().numpy()).sum()
        if self.reward_func == "force":
            reward = -max_f - self.lamb*abs(action.x.cpu().numpy()).sum()
        if self.reward_func == "energy":
            reward = -np.sqrt(abs(delta_energy)) - self.lamb*abs(action.x.cpu().numpy()).sum()
            done = done or abs(delta_energy) < self.EMAX
        if self.reward_func == "step": 
            reward = done-1
            
        if self.reward_func == "hybrid":
            r1 = -max_f 
            r2 = -np.log10(max_f) 
            r3 = done-1
            reward = self.r_weights[0]*r1+ self.r_weights[1]*r2 + self.r_weights[2]*r3 - self.lamb*abs(action.x.cpu().numpy()).sum()
            
        
        return next_struct_graph, reward, done, action, max_f, delta_energy, stop  