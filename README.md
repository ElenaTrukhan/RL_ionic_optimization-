# Reinforcement learning model based on graph convolutional networks for structure relaxation 

Official code release for the paper "Acceleration of crystal structure relaxation with Deep Reinforcement Learning" 

## Environment

Required for installation packages are in file `requirements.txt`

## Train RL Agent 

### Twin-delayed DDPG 

One needs to launch `python3 TD3_train.py`. One needs to define the path to the structures in `--structures_file` argument. It should be presented as `.csv` file with two columns: the first one representing number of cites and the second one correspoding to the `.cif` file of this structure. Function `get_sturct_lib_and_calcs_gen` in `utils.env` automatially generates two lists: one with a dataset of structures for training and another one with potentials in the form of `ase` [Calculators](https://wiki.fysik.dtu.dk/ase/ase/calculators/calculators.html) class. To get potentials `get_sturct_lib_and_calcs_gen` calls the function `func_for_calc` in `utils.calcs_func` which by default matches each structure from `--structures_file` EAM potentials in `EAM` folder, so to set potentials for your system, rewrite `func_for_calc` function. 
