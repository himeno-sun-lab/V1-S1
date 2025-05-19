#!/usr/bin/env python
# _*_coding: utf-8 _*_

##
## merge_conn_func.py
##
## This is the script to unify connection of differert population in same manner.
## There is two ways for connection of neurons provided by nest
##
## The first way is used when whole population of one defined Layer is going to connect to whole pop of another layer.
## In this case "nest.ConnectLayers" is used.   This is the recommended way for any connection.
##
## The second way is eceptional state and is used if 1) the posiiton of neurons are not known 2) just a fraction of neurons
## in source or target population are going to be used for active connection


import nest
import nest.topology as ntop

import numpy as np
import time

import fetch_params
import ini_all
import nest_routine

def connect_pre_post_layers(pre_pop, post_pop, conn_dict):

    ntop.ConnectLayers(pre_pop, post_pop, conndict)


def connect_pre_post_sublayers(pre_subpop, post_subpop, conn_spec):

    nest.Connect(pre_subpop, post_subpop)




#################################
     ## initialization ##
#################################
ctx_params = fetch_params.read_ctx()
bg_params = fetch_params.read_bg()
cb_params = fetch_params.read_cb()

# nest initialization
nest_routine.initialize_nest(sim_params)

# region definition
ctx_layers=ini_all.instantiate_ctx(ctx_params)
cb_layers = ini_all.instantiate_cb()
bg_layers = ini_all.instantiate_bg(bg_params, fake_inputs=True, ctx_inputs=nest_routine.identify_proj_neurons_ctx_bg(ctx_layers))


#################################
## cortex-cerebellum connection
#################################
start_time=time.time()


cb_layers = ini_all.instantiate_cb()
conn_spec = {"rule": "one_to_one"}
connect_pre_post_layers(ctx_layers['L5APY'], cb_layers['layer_gr'], conn_spec)


# 2.5) detectors
detectors = {}
start_time=time.time()
for layer_name in bg_layers.keys():
  detectors[layer_name] = nest_routine.layer_spike_detector(bg_layers[layer_name], layer_name)
for layer_name in ctx_layers.keys():
  detectors[layer_name] = nest_routine.layer_spike_detector(ctx_layers[layer_name], layer_name)
for layer_name in cb_layers.keys():
  detectors[layer_name] = nest_routine.layer_spike_detector(cb_layers[layer_name], layer_name)
with open('./log/'+'performance.txt', 'a') as file:
  file.write('Detectors_Elapse_Time '+str(time.time()-start_time)+'\n')

# 3) stimulation and simulation

layer2_gids = nest.GetNodes(ctx_layers['L2PY'])[0]
layer3_gids = nest.GetNodes(ctx_layers['L3PY'])[0]
layer5A_gids = nest.GetNodes(ctx_layers['L5APY'])[0]
stimu_duration = 1000.  # ms
stim_amplitude = 600.
stim_start = 1000.
stimu_interval=1000.
params = {'amplitude': stim_amplitude, 'start': float(stim_start), 'stop': float(stim_start + stimu_duration)}
laser = nest.Create("dc_generator", 1, params)
#rate_noise = 30.0  # firing rate of Poisson neurons (Hz)
#noise = nest.Create("poisson_generator", 1, params={'rate': rate_noise})
nest.Connect(laser, layer2_gids)
nest.Connect(laser, layer3_gids)


