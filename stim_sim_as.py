#!/usr/bin/env python
# -*- coding: utf-8 -*-
#########################################################
##
## stim_sim.py
##
##  Action selection script
## set up "channels" as TRUE in BG params ...
##
##
#########################################################

import fetch_params
import ini_all
import nest_routine
import nest
# import nest
import nest.topology as ntop
import numpy as np
import time
import math

#change channels True!!!!!
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
    cb_layers_S1 = ini_all.instantiate_cb('S1', sim_params['scalefactor'])
    cb_layers_M1 = ini_all.instantiate_cb('M1', sim_params['scalefactor'])
    
    #define the centers that will connect ctx to bg, and store them at bg_params['circle_center']
    #centers must be within grid 2D dimensions.
    start_time = time.time()
    if bg_params['channels']:
        if len(bg_params['circle_center'])==0: #must be done before bg instantiation.
            for i in np.arange(6):
                x_y = nest_routine.hex_corner([0,0],0.240,i) #center, radius, vertex id # gives x,y of an hexagon vertexs.
                bg_params['circle_center'].append(x_y)
            np.savetxt('./log/centers.txt',bg_params['circle_center']) #save the centers.
            print('generated centers: ',bg_params['circle_center'])

    bg_layers,ctx_bg_input = ini_all.instantiate_bg(bg_params, fake_inputs=True,
                                       ctx_inputs={'M1': {'layers': ctx_M1_layers, 'params': ctx_M1_params},
                                                   'S1': {'layers': ctx_layers, 'params': ctx_params}, 'M2': None},
                                       scalefactor=sim_params['scalefactor'])

    ################## Arm movement #################################
    ####### testing input to M1 and S1 ################################
     
    bg_PG_ctx = {}
    bg_PG_ctx['M1_L5A_CS'] = nest.Create('poisson_generator',6)
    bg_PG_ctx['M1_L5B_PT'] = nest.Create('poisson_generator',6)
    syn_PG_M1_L5A_CS={'weight': 11.5, 'delay': 1.5} #syn_PG_M1_L5A_CS={'weight': 11.0, 'delay': 1.5}    #good one -> syn_PG_M1_L5A_CS={'weight': 10.0, 'delay': 1.5}
    syn_PG_M1_L5B_PT={'weight': 13., 'delay': 1.5} #syn_PG_M1_L5B_PT={'weight': 12.5, 'delay': 1.5}    #good one -> syn_PG_M1_L5B_PT={'weight': 10.0, 'delay': 1.5}
    bg_PG_ctx['S1_L5A_Pyr'] = nest.Create('poisson_generator',6)
    bg_PG_ctx['S1_L5B_Pyr'] = nest.Create('poisson_generator',6)
    syn_PG_S1_L5A_Pyr={'weight': 2.,'delay': 1.5} #syn_PG_S1_L5A_Pyr={'weight': 2.0,'delay': 1.5}
    syn_PG_S1_L5B_Pyr={'weight': 2.2, 'delay': 1.5} #syn_PG_S1_L5B_Pyr={'weight': 2.2, 'delay': 1.5}  ## good one -> syn_PG_S1_L5B_Pyr={'weight': 4., 'delay': 1.5}


    start_stim = 2000.
    stop_stim = 2300.
    #x = np.arange(-180,180,45)
    x = np.arange(-180,180,60)
    scaling_stim = 30.#110.
    a1=17.7*scaling_stim #/3.
    a2=31.3*scaling_stim #/3.
    y_CSN = 2+a1*(1.+np.sin((x+90.)/360.*2.*math.pi))/2.      # input for M1 L5A (gives rates, they should be in the order of 1300 Hz?)
    y_PTN = 15.+a2*(1.+np.sin((x+90.)/360.*2.*math.pi))/2.    # input  for M1 L5B
    print('y_CSN:   ',y_CSN)
    print('y_PTN:   ',y_PTN)

    #ctx_bg_input['M1']:   #2 elements, 1st element is L5B, 2nd is L5A
    for j in np.arange(len(ctx_bg_input['M1'][0])): #for each circle of gids
        circle_j_gids = [k[0] for k in ctx_bg_input['M1'][0][j]] 
        circle_j_pos = [k[1] for k in ctx_bg_input['M1'][0][j]]
        print('circle M1 L5B_PT ',j,' contains #: ',len(circle_j_gids))
        nest.Connect(pre=[bg_PG_ctx['M1_L5B_PT'][j]],post=circle_j_gids,syn_spec=syn_PG_M1_L5B_PT)
        np.savetxt('./log/circle_'+str(j)+'_M1_L5B_PT_gids.txt',circle_j_gids)
        np.savetxt('./log/circle_'+str(j)+'_M1_L5B_PT_pos.txt',circle_j_pos)

    for j in np.arange(len(ctx_bg_input['M1'][1])): #for each circle of gids
        circle_j_gids = [k[0] for k in ctx_bg_input['M1'][1][j]] 
        circle_j_pos = [k[1] for k in ctx_bg_input['M1'][1][j]]
        print('circle M1 L5A_CS ',j,' contains #: ',len(circle_j_gids))
        nest.Connect(pre=[bg_PG_ctx['M1_L5A_CS'][j]],post=circle_j_gids,syn_spec=syn_PG_M1_L5A_CS)
        np.savetxt('./log/circle_'+str(j)+'_M1_L5A_CS_gids.txt',circle_j_gids)
        np.savetxt('./log/circle_'+str(j)+'_M1_L5A_CS_pos.txt',circle_j_pos)
    
    ##### Donwload also circles from S1 ####################
    for j in np.arange(len(ctx_bg_input['S1'][0])): #for each circle of gids
        circle_j_gids = [k[0] for k in ctx_bg_input['S1'][0][j]] 
        circle_j_pos = [k[1] for k in ctx_bg_input['S1'][0][j]]
        print('circle S1 L5B_Pyr ',j,' contains #: ',len(circle_j_gids))
        nest.Connect(pre=[bg_PG_ctx['S1_L5B_Pyr'][j]],post=circle_j_gids,syn_spec=syn_PG_S1_L5B_Pyr)
        np.savetxt('./log/circle_'+str(j)+'_S1_L5B_Pyr_gids.txt',circle_j_gids)
        np.savetxt('./log/circle_'+str(j)+'_S1_L5B_Pyr_pos.txt',circle_j_pos)
    for j in np.arange(len(ctx_bg_input['S1'][1])): #for each circle of gids
        circle_j_gids = [k[0] for k in ctx_bg_input['S1'][1][j]] 
        circle_j_pos = [k[1] for k in ctx_bg_input['S1'][1][j]]
        print('circle S1 L5A_Pyr ',j,' contains #: ',len(circle_j_gids))
        nest.Connect(pre=[bg_PG_ctx['S1_L5A_Pyr'][j]],post=circle_j_gids,syn_spec=syn_PG_S1_L5A_Pyr)
        np.savetxt('./log/circle_'+str(j)+'_S1_L5A_Pyr_gids.txt',circle_j_gids)
        np.savetxt('./log/circle_'+str(j)+'_S1_L5A_Pyr_pos.txt',circle_j_pos)
    
    #### Set up rates and times for the PG's
    print('circular stimulus len: ',len(y_CSN))
    #for j,i in zip([0,1,2,3,4,5],[3,2,1,0,5,4]):  # [PG indexes] , [Stimulus rate]  # 0 degree preferred direction
    for j,i in zip([0,1,2,3,4,5],[1,2,3,4,5,0]):  # [PG indexes] , [Stimulus rate]  # 90 degree preferred direction
        print(j,i)
        nest.SetStatus([bg_PG_ctx['M1_L5A_CS'][j]],{'rate':y_CSN[i],'start':start_stim,'stop':stop_stim})
        nest.SetStatus([bg_PG_ctx['M1_L5B_PT'][j]],{'rate':y_PTN[i],'start':start_stim,'stop':stop_stim})

        nest.SetStatus([bg_PG_ctx['S1_L5A_Pyr'][j]],{'rate':y_CSN[i],'start':start_stim,'stop':stop_stim})
        nest.SetStatus([bg_PG_ctx['S1_L5B_Pyr'][j]],{'rate':y_PTN[i],'start':start_stim,'stop':stop_stim})
    
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
    #for layer_name in cb_layers_S1.keys():
    #    detectors[layer_name] = nest_routine.layer_spike_detector(cb_layers_S1[layer_name], layer_name, sim_params['initial_ignore'])
    #for layer_name in cb_layers_M1.keys():
    #    detectors[layer_name] = nest_routine.layer_spike_detector(cb_layers_M1[layer_name], layer_name, sim_params['initial_ignore'])
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
    for layer_name in bg_layers.keys():
        rate = nest_routine.average_fr(detectors[layer_name], sim_params['simDuration'],nest_routine.count_layer(bg_layers[layer_name]))
        print('Layer ' + layer_name + " fires at " + str(rate) + " Hz")
    for layer_name in ctx_layers.keys():
        rate = nest_routine.average_fr(detectors[layer_name], sim_params['simDuration'],nest_routine.count_layer(ctx_layers[layer_name]))
        print('Layer ' + layer_name + " fires at " + str(rate) + " Hz")
    for layer_name in ctx_M1_layers.keys():
        rate = nest_routine.average_fr(detectors[layer_name], sim_params['simDuration'],nest_routine.count_layer(ctx_M1_layers[layer_name]))
        print('Layer ' + layer_name + " fires at " + str(rate) + " Hz")
        with open( './log/' + 'report.txt', 'a' ) as file:
            file.write('Layer '+layer_name+" fires at "+str(rate)+" Hz" + '\n' )
            file.write( str( rate ) + " Hz" + '\n' )
    
    #for layer_name in cb_layers_S1.keys():
    #    rate = nest_routine.average_fr(detectors[layer_name], sim_params['simDuration'],
    #                                   nest_routine.count_layer(cb_layers_S1[layer_name]))
    #    print('Layer ' + layer_name + " fires at " + str(rate) + " Hz")
    #for layer_name in cb_layers_M1.keys():
    #    rate = nest_routine.average_fr(detectors[layer_name], sim_params['simDuration'],
    #                                   nest_routine.count_layer(cb_layers_M1[layer_name]))
    #    print('Layer ' + layer_name + " fires at " + str(rate) + " Hz")
    
    for layer_name in th_layers['TH_S1_EZ'].keys():
        rate = nest_routine.average_fr(detectors['TH_S1_EZ' + '_' + layer_name], sim_params['simDuration'],
                                       nest_routine.count_layer(th_layers['TH_S1_EZ'][layer_name]))
        print('Layer ' + 'TH_S1_EZ'+layer_name + " fires at " + str(rate) + " Hz")
    for layer_name in th_layers['TH_S1_IZ'].keys():
        rate = nest_routine.average_fr(detectors['TH_S1_IZ' + '_' + layer_name], sim_params['simDuration'],
                                       nest_routine.count_layer(th_layers['TH_S1_IZ'][layer_name]))
        print('Layer ' + 'TH_S1_IZ'+ layer_name + " fires at " + str(rate) + " Hz")

    for layer_name in th_layers['TH_M1_EZ'].keys():
        rate = nest_routine.average_fr(detectors['TH_M1_EZ' + '_' + layer_name], sim_params['simDuration'],
                                       nest_routine.count_layer(th_layers['TH_M1_EZ'][layer_name]))
        print('Layer ' + 'TH_M1_EZ' +layer_name + " fires at " + str(rate) + " Hz")
    for layer_name in th_layers['TH_M1_IZ'].keys():
        rate = nest_routine.average_fr(detectors['TH_M1_IZ' + '_' + layer_name], sim_params['simDuration'],
                                       nest_routine.count_layer(th_layers['TH_M1_IZ'][layer_name]))
        print('Layer ' + 'TH_M1_IZ' + layer_name+ " fires at " + str(rate) + " Hz")
    
    with open('./log/' + 'performance.txt', 'a') as file:
        file.write('Output_average_firing_rate ' + str(time.time() - start_time) + '\n')

if __name__ == '__main__':
    main()
