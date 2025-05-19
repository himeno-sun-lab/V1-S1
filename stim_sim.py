#!/usr/bin/env python
# -*- coding: utf-8 -*-

##
## stim_sim.py
##
## Experiment to test that ctx and bg are connected
##
## layers 2 and 3 are excited with a series of pulses
## Layer 5A ..... are excited with a series of pulses
import fetch_params
import ini_all
import nest_routine
import nest
# import nest
import nest.topology as ntop
import numpy as np
import time



def main():
    # 1) reads parameters
    sim_params = fetch_params.read_sim()
    ctx_params = fetch_params.read_ctx()
    ctx_M1_params = fetch_params.read_ctx_M1()
    th_params = fetch_params.read_th()
    bg_params = fetch_params.read_bg()
    cb_params = fetch_params.read_cb()
    conn_params = fetch_params.read_conn()

    # 1.5) initialize nest
    nest_routine.initialize_nest(sim_params)

    start_time = time.time()
    # 2) instantiates regions
    ctx_layers, ctx_layers_excitatory = ini_all.instantiate_ctx(ctx_params, sim_params['scalefactor'], sim_params['initial_ignore'])
    ctx_M1_layers = ini_all.instantiate_ctx_M1(ctx_M1_params, sim_params['scalefactor'],sim_params['initial_ignore'])
    th_layers = ini_all.instantiate_th(th_params, sim_params['scalefactor'],sim_params['initial_ignore'])
    cb_layers_S1 = ini_all.instantiate_cb('S1', sim_params['scalefactor'], sim_params)
    cb_layers_M1 = ini_all.instantiate_cb('M1', sim_params['scalefactor'], sim_params)
    
    if sim_params['channels']: #if True
        bg_params['channels'] = True #for channels input tasks
    else:
        bg_params['channels'] = False #resting state
    bg_layers,ctx_bg_input = ini_all.instantiate_bg(bg_params, fake_inputs=True,
                                       ctx_inputs={'M1': {'layers': ctx_M1_layers, 'params': ctx_M1_params},
                                                   'S1': {'layers': ctx_layers, 'params': ctx_params}, 'M2': None},
                                       scalefactor=sim_params['scalefactor'])
    with open('./log/' + 'performance.txt', 'a') as file:
        file.write('Instantiates_Regions_Time ' + str(time.time() - start_time) + '\n')

    # 3) interconnect regions
    start_time = time.time()
    _ = nest_routine.connect_region_ctx_cb(ctx_layers['S1_L5B_Pyr'], cb_layers_S1['CB_S1_layer_pons'], 'S1')
    _ = nest_routine.connect_region_ctx_cb(ctx_M1_layers['M1_L5B_PT'], cb_layers_M1['CB_M1_layer_pons'], 'M1')
    _ = nest_routine.connect_region_ctx_th(ctx_layers, th_layers, 'S1')
    _ = nest_routine.connect_region_ctx_th(ctx_M1_layers, th_layers, 'M1')
    _ = nest_routine.connect_region_th_ctx(th_layers, ctx_layers, 'S1')
    _ = nest_routine.connect_region_th_ctx(th_layers, ctx_M1_layers, 'M1')
    _ = nest_routine.connect_region_cb_th(cb_layers_S1, th_layers, 'S1')
    _ = nest_routine.connect_region_cb_th(cb_layers_M1, th_layers, 'M1')
    _ = nest_routine.connect_region_bg_th(bg_layers, th_layers)
    with open('./log/' + 'performance.txt', 'a') as file:
        file.write('Interconnect_Regions_Time ' + str(time.time() - start_time) + '\n')

    # 2.5) detectors
    detectors = {}
    start_time = time.time()

    for layer_name in bg_layers.keys():
        detectors[layer_name] = nest_routine.layer_spike_detector(bg_layers[layer_name], layer_name, sim_params['initial_ignore'])
    for layer_name in ctx_layers.keys():
        detectors[layer_name] = nest_routine.layer_spike_detector(ctx_layers[layer_name], layer_name, sim_params['initial_ignore'])
    for layer_name in ctx_M1_layers.keys():
        detectors[layer_name] = nest_routine.layer_spike_detector(ctx_M1_layers[layer_name], layer_name, sim_params['initial_ignore'])
    for layer_name in cb_layers_S1.keys():
        detectors[layer_name] = nest_routine.layer_spike_detector(cb_layers_S1[layer_name], layer_name, sim_params['initial_ignore'])
    for layer_name in cb_layers_M1.keys():
        detectors[layer_name] = nest_routine.layer_spike_detector(cb_layers_M1[layer_name], layer_name, sim_params['initial_ignore'])
    for layer_name in th_layers['TH_S1_EZ'].keys():
        detectors['TH_S1_EZ' + '_' + layer_name] = nest_routine.layer_spike_detector(th_layers['TH_S1_EZ'][layer_name], 'TH_S1_EZ_'+layer_name, sim_params['initial_ignore'])
    for layer_name in th_layers['TH_S1_IZ'].keys():
        detectors['TH_S1_IZ' + '_' + layer_name] = nest_routine.layer_spike_detector(th_layers['TH_S1_IZ'][layer_name], 'TH_S1_IZ_'+layer_name, sim_params['initial_ignore'])
    for layer_name in th_layers['TH_M1_EZ'].keys():
        detectors['TH_M1_EZ' + '_' + layer_name] = nest_routine.layer_spike_detector(th_layers['TH_M1_EZ'][layer_name], 'TH_M1_EZ_'+layer_name, sim_params['initial_ignore'])
    for layer_name in th_layers['TH_M1_IZ'].keys():
        detectors['TH_M1_IZ' + '_' + layer_name] = nest_routine.layer_spike_detector(th_layers['TH_M1_IZ'][layer_name], 'TH_M1_IZ_'+layer_name, sim_params['initial_ignore'])
    with open('./log/' + 'performance.txt', 'a') as file:
        file.write('Detectors_Elapse_Time ' + str(time.time() - start_time) + '\n')

    simulation_time = sim_params['simDuration']+sim_params['initial_ignore']
    print('Simulation Started:')
    start_time = time.time()
    nest.Simulate(simulation_time)
    with open('./log/' + 'performance.txt', 'a') as file:
        file.write('Simulation_Elapse_Time ' + str(time.time() - start_time) + '\n')
    print ('Simulation Finish')
    # 5) output average firing rate
    start_time = time.time()
    print('Simulation debrief:')

    with open('./log/' + 'performance.txt', 'a') as file:
        file.write('Output_average_firing_rate ' + str(time.time() - start_time) + '\n')

if __name__ == '__main__':
    main()
