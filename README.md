# Reinforcement learning model based on graph convolutional networks for structure relaxation 

Official code release for the paper "Acceleration of crystal structure relaxation with Deep Reinforcement Learning" 

## Environment

Required for installation packages are in file `requirements.txt`

## Train RL Agent 

### Twin-delayed DDPG 

One need to launch `python3 TD3_train.py`. One needs to define the path to the structures in `--structures_file` argument. It should be presented as `.csv` file with two columns: the first one representing number of cites and the second one correspoding to the `.cif` file of this structure. Function `get_sturct_lib_and_calcs_gen` in `utils.env` automatially generate two lists: one with a dataset of structures for training ang another one with potentials used for forces and energies in the form of `ase` [Calculators](https://wiki.fysik.dtu.dk/ase/ase/calculators/calculators.html) object. 
