# Reinforcement learning model based on graph convolutional networks for structure relaxation 

Official code release for the paper "Acceleration of crystal structure relaxation with Deep Reinforcement Learning" 

## Environment

Required for installation packages are in file `requirements.txt`

## Train RL Agent 

In all codes one needs to define the path to the structures in `--structures_file` argument. It should be presented as `.csv` file with two columns: the first one representing number of cites and the second one correspoding to the `.cif` file of this structure. Function `get_sturct_lib_and_calcs_gen` in `utils.env` automatially generates two lists: one with a dataset of structures for training and another one with potentials in the form of `ase` [Calculators](https://wiki.fysik.dtu.dk/ase/ase/calculators/calculators.html) class. To get potentials `get_sturct_lib_and_calcs_gen` calls the function `func_for_calc` in `utils.calcs_func` which by default matches each structure from `--structures_file` EAM potentials in `EAM` folder, so to set potentials for your system, rewrite `func_for_calc` function. 

### Twin-delayed DDPG 

`python3 TD3_e3nn.py --path_to_the_main_dir 'outputs/$exp_name' --path_load 'last' --path_weights 'last' --structures_file "structures/AlFe_cubic.csv" --reward_func "force" --env_name "AlFe_cubic" --eps 0.01 --start_iter 0 --random_seed 3211 --nfake 0 --test_ep [10,100] --with_weights False --r_weights [1,1,1] --r0 1.5 --stop_numb 1e06 --r_max 5 --em_dim 10 --noise_clip 0.1 --pi_n_layers 2 --pi_mul 20 --pi_lmax 2 --num_neighbors 25 --q_n_layers 2 --q_mul 20 --q_lmax 2 --replay_size 1e6 --gamma 0.9999 --polyak 0.995 --pi_lr 1e-05 --q_lr 1e-05 --batch_size 100 --start_steps 0 --update_after 0 --update_every 1 --target_noise 0.05 --policy_delay 2 --trans_coef 0.5 --noise [0.2,0.2] --train_ep [500,1000] --save_result True --test_random False --expl_mode 'state' --test_every 1000 --save_every 1000 --N_gr 1e6 --d_r_max 0 --f_max 0.001 --noise_level 29`
