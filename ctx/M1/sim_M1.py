#!/usr/bin/env python
# -*- coding: utf-8 -*-

# This is the all program for simulation of "primary motor cortex" including 6 layers
# eventhough L4 is somehow neglectable for M1

# this is the program which uses the "baseCTXM1Params.py" file as a initial values for the M1 and also
#  "M1_internal_connection.pickle" file as a source for the connection matrix.
################
# this new generation (190120) do not use npz numpy files as a input for the position of nodes and neurons, instead
# calculate it through program and use memory for it
# also this program uses "copymodel" nest method to define cell populations.
################


import fetch_params
import ini_all
import nest_routine
import nest
# import nest.topology as ntop
import numpy as np
import time
from nest.lib.hl_api_info import SetStatus


# 1) reads parameters
def main():
  sim_params = fetch_params.read_sim()
  ctx_params = fetch_params.read_ctx()

  # 2) initialize nest
  nest_routine.initialize_nest( sim_params )

  # 3) instantiates regions
  start_time = time.time()
  ctx_layers, ctx_layers_excitatory =ini_all.instantiate_ctx(ctx_params, sim_params['scalefactor'])
  with open( './log/' + 'performance.txt', 'a' ) as file:
    file.write( 'S1_Construction_Time ' + str( time.time() - start_time ) + '\n' )
    # _=get_connection_summary(ctx_params, ctx_layers)

  # 4) detectors
  detectors = {}
  start_time = time.time()
  for layer_name in ctx_layers.keys():
    detectors [layer_name] = nest_routine.layer_spike_detector( ctx_layers [layer_name], layer_name )
  with open( './log/' + 'performance.txt', 'a' ) as file:
    file.write( 'Detectors_Elapse_Time ' + str( time.time() - start_time ) + '\n' )

  simulation_time = sim_params ['simDuration']
  print ('Simulation Started:')
  start_time=time.time()
  nest.Simulate(simulation_time)
  with open('./log/'+'performance.txt', 'a') as file:
    file.write('Simulation_Elapse_Time '+str(time.time()-start_time)+'\n')

  for layer_name in ctx_layers.keys():
    rate = nest_routine.average_fr(detectors[layer_name], sim_params['simDuration'], nest_routine.count_layer(ctx_layers[layer_name]))
    print('Layer '+layer_name+" fires at "+str(rate)+" Hz")

if __name__ == '__main__':
    main()
