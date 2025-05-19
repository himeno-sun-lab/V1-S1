#!/usr/bin/env python
# -*- coding: utf-8 -*-
#########################################################
##
## stim_sim_nao.py
##
##  Action selection script and action 
##  integrated to NAO environment @ UEC
##  set up bg_parmas["channels"] as TRUE in BG params ...
##
##
#########################################################

import fetch_params
import ini_all
import nest_routine
import nest
import nest.topology as ntop
import numpy as np
import time
import math
from wbnao import WBNao
import time


def main():
    # 1) reads parameters
    sim_params = fetch_params.read_sim()
    ctx_params = fetch_params.read_ctx()
    ctx_M1_params = fetch_params.read_ctx_M1()
    th_params = fetch_params.read_th()
    bg_params = fetch_params.read_bg()
    cb_params = fetch_params.read_cb()

    # 2) initialize nest
    nest_routine.initialize_nest(sim_params)
    
    # 3) instantiates regions
    ctx_layers, ctx_layers_excitatory = ini_all.instantiate_ctx(ctx_params, sim_params['scalefactor'], sim_params['initial_ignore'])
    ctx_M1_layers = ini_all.instantiate_ctx_M1(ctx_M1_params, sim_params['scalefactor'],sim_params['initial_ignore'])
    
    th_layers = ini_all.instantiate_th(th_params, sim_params['scalefactor'],sim_params['initial_ignore'])
    
    cb_layers_S1 = ini_all.instantiate_cb('S1', sim_params['scalefactor'])
    cb_layers_M1 = ini_all.instantiate_cb('M1', sim_params['scalefactor'])
   
    bg_params['circle_center'] = nest_routine.get_channel_centers(bg_params,hex_center=[0,0],ci=6,hex_radius=0.240)
    bg_layers,ctx_bg_input = ini_all.instantiate_bg(bg_params, fake_inputs=True,
                                       ctx_inputs={'M1': {'layers': ctx_M1_layers, 'params': ctx_M1_params},
                                                   'S1': {'layers': ctx_layers, 'params': ctx_params}, 'M2': None},
                                       scalefactor=sim_params['scalefactor'])

    # 4) interconnect regions
    _ = nest_routine.connect_region_ctx_cb(ctx_layers['S1_L5B_Pyr'], cb_layers_S1['CB_S1_layer_pons'], 'S1')
    _ = nest_routine.connect_region_ctx_cb(ctx_M1_layers['M1_L5B_PT'], cb_layers_M1['CB_M1_layer_pons'], 'M1')
    _ = nest_routine.connect_region_ctx_th(ctx_layers, th_layers, 'S1')
    _ = nest_routine.connect_region_ctx_th(ctx_M1_layers, th_layers, 'M1')
    _ = nest_routine.connect_region_th_ctx(th_layers, ctx_layers, 'S1')
    _ = nest_routine.connect_region_th_ctx(th_layers, ctx_M1_layers, 'M1')
    _ = nest_routine.connect_region_cb_th(cb_layers_S1, th_layers, 'S1')
    _ = nest_routine.connect_region_cb_th(cb_layers_M1, th_layers, 'M1')
    _ = nest_routine.connect_region_bg_th(bg_layers, th_layers)

    # 5) create spike detectors 
    detectors = {}
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


    delta_t= 500.
    # Run the network for 500 ms to get a steady state....
    nest.Simulate(delta_t)

    # 5) Check if File exists to run simulation, if it doesn't exists wait some time and then check again.
    maximum_iterations = 1
    iteration_counter = 0
    #file_path = './log/LCR.txt'
    #delta_t = 500.
    total_sim_time = delta_t * maximum_iterations 
    senders, spiketimes = [],[]
    my_actions = [2,3,1,2,3,1]

    wbnaox = WBNao()
    wbnaox.clean()

    while iteration_counter < maximum_iterations:
        #while not os.path.exists(file_path):
        #    time.sleep(30)              #wait some time until NEO deposits a file.
        if True:#os.path.isfile(file_path):
            direction = wbnaox.getDesiredDirection()
            print(direction)
            circle_gid_detector,circle_j_gids_nb=nest_routine.apply_direction_stimulus(ctx_bg_input,direction,time_start=100.,time_stop=400.) #apply stimulation between 100-400 ms (300 ms) ....
            nest.Simulate(delta_t)
            rate=[]
            # if direction =='L':
            #      rate=[0,0,0,1,0,0]
            # if direction =='R':
            #      rate=[0,0,1,0,0,0]
            # if direction =='C':
            #      rate=[1,0,0,0,0,0]
            for circle_name in circle_gid_detector.keys():
                rate.append(nest_routine.average_fr_pre(circle_gid_detector[circle_name], circle_j_gids_nb[circle_name],
                                                        start_time=iteration_counter * 500. + 100.,
                                                        end_time=(iteration_counter + 1) * 500. - 100.))
            iteration_counter +=1       #will access up to maximum_iterations times to a file
            wbnaox.putPopulationVector(rate)
if __name__ == '__main__':
    main()
