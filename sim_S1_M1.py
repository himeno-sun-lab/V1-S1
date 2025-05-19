#!/usr/bin/env python
# -*- coding: utf-8 -*-

import fetch_params
import ini_all
import nest_routine
import nest
import time

def main():

    # 1) reads parameters
    sim_params = fetch_params.read_sim()
    ctxM1Params = fetch_params.read_ctx_M1()
    ctxS1Params = fetch_params.read_ctx()


    # 2) initialize nest
    nest_routine.initialize_nest(sim_params)

    # 3) instantiates regions
    start_time = time.time()
    s1_layers = ini_all.instantiate_ctx(ctxS1Params, sim_params['scalefactor'], sim_params['initial_ignore'], 'S1')
    m1_layers = ini_all.instantiate_ctx(ctxM1Params, sim_params['scalefactor'], sim_params['initial_ignore'], 'M1')


    # 4) connect S1 l5A and L5B to M1 L23   
    conn_params = sim_params['conn_params']
    nest_routine.connect_specific_layers_ctx_uniform(s1_layers['S1_L5A_Pyr'], m1_layers['M1_L23_CC'], conn_params)
    nest_routine.connect_specific_layers_ctx_uniform(s1_layers['S1_L5B_Pyr'], m1_layers['M1_L23_CC'], conn_params)


    # 5) create columns in S1 and M1
    l23_params = sim_params['m1_circle_center']
    circle_center_params = sim_params['s1_circle_center']
    gids_L5A = nest_routine.get_columns_data('S1_L5A_Pyr', circle_center_params, sim_params['channels_radius'])
    gids_L5B = nest_routine.get_columns_data('S1_L5B_Pyr', circle_center_params, sim_params['channels_radius'])
    gids_L23 = nest_routine.get_columns_data('M1_L23_CC', l23_params, sim_params['channels_radius'])
    
    detector_S1_L5 = {}
    detector_M1_L23 = {}
    S1_columns = {}
    M1_columns = {}

    stimulation_start_time = sim_params['stimulation_start_time']

    for i in range(len(gids_L5A)):
        gid_L5A_list = [item[0] for item in gids_L5A[i]]
        gid_L5B_list = [item[0] for item in gids_L5B[i]]
        column_neurons = gid_L5A_list + gid_L5B_list

        nest_routine.add_poisson_generator_column(column_neurons, stimulation_start_time, param=sim_params['Psg_params'][i])
        detector = nest_routine.layer_spike_detector(column_neurons, layer_name=f"S1_L5_col{i}", ignore_time=0)

        detector_S1_L5[f"col{i}"] = detector
        S1_columns[f"col{i}"] = column_neurons

    for i in range(len(gids_L23)):
        gids_L23_list = [item[0] for item in gids_L23[i]]
        detector = nest_routine.layer_spike_detector(gids_L23_list, layer_name=f"M1_L23_col{i}", ignore_time=0)

        detector_M1_L23[f"col{i}"] = detector
        M1_columns[f"col{i}"] = gids_L23_list

    print("Start connect across brain region")

    # Establish 4-to-2 column connections
    for s1_col_idx in range(4): 
        for m1_col_idx in range(2):  
            source_nodes = nest.NodeCollection(S1_columns[f'col{s1_col_idx}'])
            target_nodes = nest.NodeCollection(M1_columns[f'col{m1_col_idx}'])
            info = nest.GetConnections(source=source_nodes, target=target_nodes)
            print(sim_params['weight_matrix'][s1_col_idx][m1_col_idx])
            nest.SetStatus(info, {'weight': sim_params['weight_matrix'][s1_col_idx][m1_col_idx]})

    with open('./log/performance.txt', 'a') as file:
        file.write('S1_M1_Construction_Time ' + str(time.time() - start_time) + '\n')

    # 6) detectors for s1 and m1
    detectors_S1 = {}
    detectors_M1 = {}

    start_time = time.time()

    for layer_name in s1_layers.keys():
        detectors_S1[layer_name] = nest_routine.layer_spike_detector(s1_layers[layer_name], layer_name, 0)

    for layer_name in m1_layers.keys():
        detectors_M1[layer_name] = nest_routine.layer_spike_detector(m1_layers[layer_name], layer_name, 0)

    with open('./log/performance.txt', 'a') as file:
        file.write('Detectors_Elapse_Time ' + str(time.time() - start_time) + '\n')

    # 7) simulation
    simulation_time = sim_params['simDuration']
    print('Simulation Started:')
    start_time = time.time()
    nest.Simulate(simulation_time)
    with open('./log/performance.txt', 'a') as file:
        file.write('Simulation_Elapse_Time ' + str(time.time() - start_time) + '\n')

    # 8) save results
    print("\nS1 firing rate:")
    with open('log/s1_m1_firing_rates.txt', 'w') as f:
        f.write("Psg_params:\n")
        for i, param in enumerate(sim_params['Psg_params']):
            f.write(f"col{i}: {param}\n")
        
        f.write("\nWeight Matrix (S1 -> M1):\n")
        f.write("S1\\M1 | col0 | col1\n")
        f.write("------|------|------\n")
        for i, weights in enumerate(sim_params['weight_matrix']):
            f.write(f"col{i}  | {weights[0]:.1f}  | {weights[1]:.1f}\n")
        
        f.write("\nS1 firing rates:\n")
        for layer_name in s1_layers.keys():
            rate = nest_routine.average_fr(
                detectors_S1[layer_name], 
                sim_params['simDuration'], 
                nest_routine.count_layer(s1_layers[layer_name])
            )
            print(f'layer {layer_name} fires at  {rate} Hz')
            f.write(f'layer {layer_name} fires at {rate} Hz\n')

        f.write("\nM1 firing rates:\n")
        print("\nM1 firing rate: ")
        for layer_name in m1_layers.keys():
            rate = nest_routine.average_fr(
                detectors_M1[layer_name], 
                sim_params['simDuration'], 
                nest_routine.count_layer(m1_layers[layer_name])
            )
            print(f'layer {layer_name} fires at {rate} Hz')
            f.write(f'layer {layer_name} fires at {rate} Hz\n')

        f.write("\nS1 Column firing rates:\n")
        print("\nS1 Column firing rate: ")
        for column in detector_S1_L5.keys():
            rate = nest_routine.average_fr_pre(
                detector_S1_L5[column],
                len(S1_columns[column]),
                stimulation_start_time,
                simulation_time
            )
            print(f'{column} fires at {rate} Hz')
            f.write(f'{column} fires at {rate} Hz\n')

        f.write("\nM1 Column firing rates:\n")
        print("\nM1 Column firing rate: ")
        for column in detector_M1_L23.keys():
            rate = nest_routine.average_fr_pre(
                detector_M1_L23[column],
                len(M1_columns[column]),
                stimulation_start_time,
                simulation_time
            )
            print(f'{column} fires at {rate} Hz')
            f.write(f'{column} fires at {rate} Hz\n')

if __name__ == '__main__':
    main() 