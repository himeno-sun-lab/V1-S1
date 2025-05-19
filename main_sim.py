#!/usr/bin/env python
# -*- coding: utf-8 -*-

##
## main_sim.py
##
## This is the main simulation file.
## 1) it reads parameters from the current directory through calls to fetch_params.py
## 2) it instantiates each regions through calls to functions defined in ini_all.py
## 3) it instantiates inter-regional connections
## 4) it starts the simulation
## 5) it outputs average firing rates


import fetch_params
import ini_all
import nest_routine


def main():

  # 1) reads parameters
  sim_params = fetch_params.read_sim()
  ctx_params = fetch_params.read_ctx()
  ctx_M1_params = fetch_params.read_ctx_M1()
  '''
  ctx_M2_params = fetch_params.read_ctx_M2()
  '''
  #th_params = fetch_params.read_th()
  #bg_params = fetch_params.read_bg()
  #cb_params = fetch_params.read_cb()

  # 1.5) initialize nest
  nest_routine.initialize_nest(sim_params)

  # 2) instantiates regions
  ctx_layers = ini_all.instantiate_ctx(ctx_params, sim_params['initial_ignore'])
  ctx_M1_layers = ini_all.instantiate_ctx_M1(ctx_M1_params, sim_params['initial_ignore'])
  '''
  ctx_M2_layers = ini_all.instantiate_ctx_M2(ctx_M2_params, sim_params['initial_ignore'])
  '''
  #bg_layers = ini_all.instantiate_bg(bg_params, fake_inputs=True, ctx_inputs=nest_routine.identify_proj_neurons_ctx_bg(ctx_layers))
  #bg_layers = ini_all.instantiate_bg(bg_params, fake_inputs=True, ctx_inputs=nest_routine.identify_proj_neurons_ctx_bg(ctx_layers))
  #th_layers = ini_all.instantiate_th(th_params)
  #cb_layers = ini_all.instantiate_cb()


  # 3) interconnect regions
  #print (ctx_layers['L5APY'])
  #print (cb_layers['layer_gr'])
  #_ = nest_routine.connect_region_ctx_cb(ctx_layers['L5APY'], cb_layers['layer_gr'], 'S1')
  # _ = nest_routine.connect_region_ctx_cb(ctx_M1_layers['L5BPT'], cb_layers['layer_gr'], 'M1')
  # _ = nest_routine.connect_region_ctx_th(ctx_layers, th_layers, 'S1')
  # _ = nest_routine.connect_region_ctx_th(ctx_M1_layers, th_layers, 'M1')
  # _ = nest_routine.connect_region_cb_th(cb_layers, th_layers, 'S1')
  # _ = nest_routine.connect_region_cb_th(cb_layers, th_layers, 'M1')

  # 2.5) detectors
  detectors = {}
  #for layer_name in bg_layers.keys():
  #detectors[layer_name] = nest_routine.layer_spike_detector(bg_layers[layer_name], layer_name)
  for layer_name in ctx_layers.keys():
    detectors[layer_name] = nest_routine.layer_spike_detector(ctx_layers[layer_name], layer_name, sim_params['initial_ignore'])
  for layer_name in ctx_M1_layers.keys():
    detectors[layer_name] = nest_routine.layer_spike_detector(ctx_M1_layers[layer_name], layer_name, sim_params['initial_ignore'])
  '''
  for layer_name in ctx_M2_layers.keys():
    detectors[layer_name] = nest_routine.layer_spike_detector(ctx_M2_layers[layer_name], layer_name)
  '''
  #for layer_name in cb_layers.keys():
    #detectors[layer_name] = nest_routine.layer_spike_detector(cb_layers[layer_name], layer_name)
  
  # 4) launch the simulation
  print('Starting the simulation...')
  nest_routine.run_simulation(sim_params)

  # 5) output average firing rate
  print('Simulation debrief:')
  #for layer_name in bg_layers.keys():
    #rate = nest_routine.average_fr(detectors[layer_name], sim_params['simDuration'], bg_params['nb'+layer_name])
    #print('Layer '+layer_name+" fires at "+str(rate)+" Hz")
  for layer_name in ctx_layers.keys():
    rate = nest_routine.average_fr(detectors[layer_name], sim_params['simDuration'], nest_routine.count_layer(ctx_layers[layer_name]))
    print('Layer ctx S1 '+layer_name+" fires at "+str(rate)+" Hz")
  for layer_name in ctx_M1_layers.keys():
    rate = nest_routine.average_fr(detectors[layer_name], sim_params['simDuration'], nest_routine.count_layer(ctx_M1_layers[layer_name]))
    print('Layer ctx M1 ' + layer_name + " fires at " + str(rate) + " Hz")
  '''
  for layer_name in ctx_M2_layers.keys():
    rate = nest_routine.average_fr(detectors[layer_name], sim_params['simDuration'], nest_routine.count_layer(ctx_M2_layers[layer_name]))
    print('Layer ' + layer_name + " fires at " + str(rate) + " Hz")  
  '''

if __name__ == '__main__':
    main()
