# script for testing multiple parameters in one command


import json
import numpy as np
from nest.lib.hl_api_info import SetStatus
import subprocess
import os
import argparse

def parse_arguments():
    parser = argparse.ArgumentParser(description='S1-M1 Network Simulation Test Suite')
    parser.add_argument('--platform', type=str, default='Local',
                       help='Platform to run the simulation (default: Local)')
    parser.add_argument('--whichSim', type=str, default='sim_S1_M1',
                       help='Which simulation to run (default: sim_S1_M1)')
    parser.add_argument('--customSim', type=str, default='test_simParams.py',
                       help='Custom simulation parameters file (default: test_simParams.py)')
    return parser.parse_args()

def run_simulation(psg_params, weight_matrix, rmd_sd, result_file):
    try:
        from runpy import run_path
        file_params = run_path('test_simParams.py', init_globals=globals())
        sim_params = file_params['simParams']
    except:
        raise ImportError('The simulation parameters could not be loaded. Please make sure that the file `simParams.py` exists and is a valid python defining the variable "simParams".')

    sim_params['Psg_params'] = psg_params
    sim_params['weight_matrix'] = weight_matrix
    sim_params['msd'] = rmd_sd

    
    with open('test_simParams.py', 'w') as f:
        params_str = json.dumps(sim_params)
        params_str = params_str.replace('true', 'True').replace('false', 'False').replace('null', 'None')
        f.write("simParams = " + params_str)
    
    print(f"Running simulation with Psg_params: {json.dumps(psg_params)}")
    print(f"Weight matrix: {json.dumps(weight_matrix)}")
    
    cmd = [
        "python", "run.py",
        "--platform", "Local",
        "--whichSim", "sim_S1_M1",
        "--customSim", "test_simParams.py"
    ]
    
    try:
        # run command
        subprocess.run(cmd, check=True)
        
      
        latest_dir = max([d for d in os.listdir('.') if d.startswith('20')], key=os.path.getctime)
        result_path = os.path.join(latest_dir, 'log', 's1_m1_firing_rates.txt')
        
        # record results
        if os.path.exists(result_path):
            os.system(f'cp {result_path} {result_file}')
            print(f"Results saved to {result_file}\n")
        else:
            print(f"Warning: No results file found in {latest_dir}\n")
            
    except subprocess.CalledProcessError as e:
        print(f"Error running simulation")
        print(f"Error message: {e}\n")
        return False
    return True

def main():
    
    if not os.path.exists('results'):
        os.makedirs('results')


    # random_seeds = [582936, 104857, 739201, 628395, 193746, 847261, 320495, 756238, 649130, 285974, 123456]

    # random_seeds = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 123456]
    random_seeds = [10]


    
    # define different psg_params combinations
    # Psg_params_list = [
    #     [{'rate': 100.0}, {'rate': 0.0}, {'rate': 0.0}, {'rate': 0.0}],
    #     [{'rate': 0.0}, {'rate': 100.0}, {'rate': 0.0}, {'rate': 0.0}],
    #     [{'rate': 0.0}, {'rate': 0.0}, {'rate': 100.0}, {'rate': 0.0}],
    #     [{'rate': 0.0}, {'rate': 0.0}, {'rate': 0.0}, {'rate': 100.0}]
    # ]


    # define different psg_params combinations
    Psg_params_list = [
        [{'rate': 100.0}, {'rate': 0.0}, {'rate': 0.0}, {'rate': 0.0}],
        [{'rate': 0.0}, {'rate': 100.0}, {'rate': 0.0}, {'rate': 0.0}],
        [{'rate': 0.0}, {'rate': 0.0}, {'rate': 100.0}, {'rate': 0.0}],
        [{'rate': 0.0}, {'rate': 0.0}, {'rate': 0.0}, {'rate': 100.0}]
    ]




    # define different weight matrix combinations
    weight_matrix_list = [
        # [[1.8, 1.8], [1.8, 2.8], [2.8, 1.8], [1.8, 1.8]],
        # [[5.8, 1.8], [1.8, 5.8], [1.8, 1.8], [1.8, 1.8]],
        # [[5.8, 0.8], [0.8, 5.8], [0.8, 5.8], [5.8, 0.8]], 
        # [[3.8, 3.8], [3.8, -0.8], [-0.8, -0.8], [-0.8, 3.8]],
        # [[2.8, 0.8], [0.8, 2.8], [0.8, 2.8], [2.8, 0.8]],
        [[3.8, 0.8], [0.8, 3.8], [0.8, 3.8], [3.8, 0.8]],


        # [[5.8, 1.8], [1.8, 5.8], [1.8, 5.8], [5.8, 1.8]], 
        # [[7.8, 0.8], [0.8,7.8], [0.8, 7.8], [7.8, 0.8]], 
        # [[7.8, 1.8], [1.8, 7.8], [1.8, 7.8], [7.8, 1.8]], 

        # [[3.8, 0.8], [0.8, 3.8], [0.8, 3.8], [3.8, 0.8]], 
        # [[1.8, 1.8], [1.8, 2.8], [2.8, 1.8], [1.8, 1.8]],
        # [[2.8, 2.8], [2.8, 2.8], [2.8, 2.8], [2.8, 2.8]],
        # [[3.8, 1.8], [1.8, 3.8], [3.8, 1.8], [1.8, 3.8]],
        # [[3.8, 3.8], [3.8, -0.8], [-0.8, -0.8], [-0.8, 3.8]]  
    ]


    # Iterate through all parameter combinations
    for sd_idx, rmd_sd in enumerate(random_seeds):
        for weight_idx, weight_matrix in enumerate(weight_matrix_list):
            for psg_idx, Psg_params in enumerate(Psg_params_list):
            # save
                result_file = f'results/time{sd_idx}_weight{weight_idx}_run_psg{psg_idx}.txt'
                
                # run simulation
                success = run_simulation(Psg_params, weight_matrix, rmd_sd, result_file)
                
                if not success:
                    print(f"Failed to run simulation with Psg_params {psg_idx} and weight_matrix {weight_idx}")
                    # continue
                    break

if __name__ == '__main__':
    main() 