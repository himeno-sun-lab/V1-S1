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
import numpy as np
import time
import nest.voltage_trace
import nest.raster_plot
import os

# simulation circle 0.1 mm input for any layer except layer 1 

def main():
  # 1) reads parameters
  sim_params = fetch_params.read_sim()
  ctx_M1_params = fetch_params.read_ctx()

  # 1.5) initialize nest
  nest_routine.initialize_nest(sim_params)

  # 2) instantiates regions
  start_time = time.time()
  ctx_layers = ini_all.instantiate_ctx(ctx_M1_params, sim_params['scalefactor'], sim_params['initial_ignore'], 'M1')
  with open('./log/'+'performance.txt', 'a') as file:
    file.write('S1_Construction_Time '+str(time.time()-start_time)+'\n')


  detectors = {}
  for layer_name in ctx_layers.keys():
    detectors[layer_name] = nest_routine.layer_spike_detector(ctx_layers[layer_name], layer_name,
                                                              sim_params['initial_ignore'])

  # macro_circle_center_params = nest_routine.get_macro_channel_centers(sim_params, hex_center=[0, 0],
  #                                                                     ci=sim_params['macro_columns_nb'],
  #                                                                     hex_radius=sim_params['hex_radius'])
  # micro_circle_center_params = nest_routine.get_micro_channel_centers(macro_circle_center_params,
  #                                                                     ci=sim_params['micro_columns_nb'],
  #                                                                     hex_radius=sim_params[
  #                                                                                  'channels_radius'] * 2. / 3.)
  # ctx_S1_macro_columns_gids = {}
  # ctx_S1_micro_columns_gids = {}
  #
  # for layer_name in ctx_layers.keys():
  #   ctx_S1_macro_columns_gids[layer_name] = nest_routine.get_macro_columns_data(layer_name, macro_circle_center_params,sim_params['channels_radius'], layer_name)
  #   for i in range(len(macro_circle_center_params)):
  #     ctx_S1_micro_columns_gids[layer_name + '_macro_column_' + str(i)] = nest_routine.get_micro_columns_data(
  #       layer_name, micro_circle_center_params['macro_channel_' + str(i)], sim_params['channels_radius'] / 3.,
  #       layer_name + '_macro_column_' + str(i))

  simulation_time = sim_params['simDuration'] + sim_params['initial_ignore']
  print('Simulation Started:')
  start_time = time.time()
  nest.Simulate(simulation_time)
  with open('./log/' + 'performance.txt', 'a') as file:
    file.write('Simulation_Elapse_Time ' + str(time.time() - start_time) + '\n')
  print ('Simulation Finish')

  # ctx_S1_macro_columns_activity = {}
  # ctx_S1_micro_columns_activity = {}
  # for layer_name in ctx_layers.keys():
  #   ctx_S1_macro_columns_activity[layer_name] = nest_routine.get_firing_rate_macro_column(layer_name,ctx_S1_macro_columns_gids[layer_name], start_time, end_time)
  #
  #   print(ctx_S1_macro_columns_activity[layer_name])
  #   for i in range(sim_params['macro_columns_nb']):
  #     ctx_S1_micro_columns_activity[layer_name + '_macro_column_' + str(i)] = nest_routine.get_firing_rate_micro_column(layer_name, ctx_S1_micro_columns_gids[layer_name + '_macro_column_' + str(i)], start_time, end_time, i)
  #     print (ctx_S1_micro_columns_activity[layer_name + '_macro_column_' + str(i)])

  # print('Simulation debrief:')
  # for layer_name in ctx_layers.keys():
  #   rate = nest_routine.get_firing_rate_from_gdf_files(layer_name, detectors[layer_name], sim_params['simDuration'], nest_routine.count_layer(ctx_layers[layer_name]))
  #   print('Layer ' + layer_name + " fires at " + str(rate) + " Hz")



if __name__ == '__main__':
    main()
