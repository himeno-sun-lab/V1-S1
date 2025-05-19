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
# import nest.Topology as ntop
import numpy as np
import time
import nest.voltage_trace
import nest.raster_plot
import os


# simulation circle 0.1 mm input for any layer except layer 1 

def main():
  # 1) reads parameters
  sim_params = fetch_params.read_sim()
  ctx_params = fetch_params.read_ctx()

  # 1.5) initialize nest
  nest_routine.initialize_nest(sim_params)

  # 2) instantiates regions
  start_time = time.time()
  ctx_layers =ini_all.instantiate_ctx(ctx_params, sim_params['scalefactor'], sim_params['initial_ignore'], 'S1')
  with open('./log/'+'performance.txt', 'a') as file:
    file.write('S1_Construction_Time '+str(time.time()-start_time)+'\n')

  # out degree connection figure
  # all_neuron_GIDS = []
  # for pre_l in ctx_layers.keys():
  #   for post_l in ctx_layers.keys():
  #     all_neuron_GIDS.extend(list(nest.GetNodes(ctx_layers[post_l])[0]))
  #   center_n = ntop.FindCenterElement(ctx_layers[pre_l])
  #   connectome_n = nest.GetConnections(center_n, all_neuron_GIDS)
  #   conn_list = np.zeros((len(connectome_n), 9))
  #   for i_ in range(len(connectome_n)):
  #     info = nest.GetStatus([connectome_n[i_]])
  #     weight = info[0]['weight']
  #     pre_neuron_GID = connectome_n[i_][0]
  #     post_neuron_GID = connectome_n[i_][1]
  #     conn_list[i_, 0] = pre_neuron_GID
  #     conn_list[i_, 1] = post_neuron_GID
  #     conn_list[i_, 2] = weight
  #     conn_list[i_, 3:6] = np.asarray(ntop.GetPosition([pre_neuron_GID])[0])
  #     conn_list[i_, 6:] = np.asarray(ntop.GetPosition([post_neuron_GID])[0])
  #   np.savetxt('log/conn_out_exp_%s.csv' % (pre_l),conn_list, delimiter=",", fmt='%.5f')
  # print (nest.GetStatus([neuron], {"V_m"}))

  #2.5) detectors

  # #############################################################################################################
  # def layer_spike_detector(layer_gid, layer_name, ignore_time,
  #                          params={"withgid": True, "withtime": True, "to_file": True}):
  #   # def layer_spike_detector(layer_gid, layer_name, params={"withgid": True, "withtime": True, "to_file": True, 'fbuffer_size': 8192}):
  #   print('spike detector for ' + layer_name)
  #   params.update({'label': layer_name, "start": 1000., "stop":2000.})  #
  #   ini_time = ignore_time - 500
  #   ini_time_ini = ignore_time - 500
  #   # first adding poisson generator one for all neuron types different for the inhibitory and excitatory neurons
  #   name_list = ['S1_L2_Pyr', 'S1_L3_Pyr', 'S1_L4_Pyr', 'S1_L5A_Pyr', 'S1_L5B_Pyr', 'S1_L6_Pyr']
  #
  #   # Add detector for all neuron types
  #   detector = nest.Create("spike_detector", params=params)
  #
  #   nest.Connect(pre=nest.GetNodes(layer_gid)[0], post=detector)
  #
  #   # add multimeter just for to record V_m and conductance (inhibitory and excitatory) of a single cell each cell population
  #   nest_mm = nest.Create('multimeter',
  #                         params={"interval": 0.1, "record_from": ["V_m", "g_ex", "g_in"], "withgid": True,
  #                                 "to_file": True,
  #                                 'label': layer_name, 'withtime': True,
  #                                 'use_gid_in_filename': True})  # "start": float(ignore_time),
  #   nest.Connect(nest_mm, [nest.GetNodes(layer_gid)[0][7]])
  #
  #   # voltmeter_con=nest.Create("voltmeter", params = {'withgid':True, 'withtime':True})
  #   voltmeter_single = nest.Create("voltmeter", params={'to_file': False, 'label': layer_name, "withgid": True,
  #                                                       "withtime": True})  # 'use_gid_in_filename': True,   "start": float(ignore_time),
  #   # nest.Connect(pre=voltmeter_con, post= nest.GetNodes(layer_gid)[0])
  #   nest.Connect(voltmeter_single, [nest.GetNodes(layer_gid)[0][7]])
  #
  #   return detector, nest_mm, voltmeter_single
  #poisson_gens = {}
  # detectors = {}
  # multimeters = {}
  # voltmeters = {}
  # start_time = time.time()
  # for layer_name in ctx_layers.keys():
  #   detectors[layer_name], multimeters[layer_name], voltmeters[layer_name] = layer_spike_detector(ctx_layers[layer_name], layer_name, sim_params['initial_ignore'])
  # with open('./log/'+'performance.txt', 'a') as file:
  #   file.write('Detectors_Elapse_Time '+str(time.time()-start_time)+'\n')
  #
  # simulation_time = sim_params['simDuration'] + sim_params['initial_ignore']
  # print('Simulation Started:')
  # start_time = time.time()
  # nest.Simulate(simulation_time)
  # with open('./log/' + 'performance.txt', 'a') as file:
  #   file.write('Simulation_Elapse_Time: ' + str(time.time() - start_time) + '\n')

  # ###############
  # # plotting the garphs
  # import matplotlib.pyplot as plt
  # i=1
  # for layer_name in ctx_layers.keys():
  #   plt.figure()
  #   nest.voltage_trace.from_device(voltmeters[layer_name], title=layer_name)
  #   if not os.path.exists('./log/voltage/'):
  #       os.makedirs('./log/voltage/')
  #   save_results_to = './log/voltage/'
  #   plt.savefig(save_results_to + str(i) +'.jpeg')
  #   i=i+1
  #
  # ###############
  # i=1
  # for layer_name in ctx_layers.keys():
  #   plt.figure()
  #   events = nest.GetStatus(multimeters[layer_name])[0]["events"]
  #   t = events["times"]
  #   plt.plot(t, events["g_ex"], t, events["g_in"])
  #   plt.xlabel("time (ms)")
  #   plt.ylabel("synaptic conductance (nS)")
  #   plt.title(layer_name)
  #   plt.legend(("g_exc", "g_inh"))
  #   if not os.path.exists('./log/conductance/'):
  #       os.makedirs('./log/conductance/')
  #   save_results_to = './log/conductance/'
  #   plt.savefig(save_results_to + str(i) +'.jpeg')
  #   i=i+1
  # ###############
  # i=1
  # for layer_name in ctx_layers.keys():
  #   try :
  #     nest.raster_plot.from_device(detectors[layer_name], title = layer_name, hist=False)
  #     font = {'family' : 'normal', 'weight' : 'normal', 'size': 42}
  #     plt.rcParams["figure.figsize"] = [64,48]
  #     plt.rc('font', **font)
  #     plt.rc('xtick', labelsize=40)
  #     plt.rc('ytick', labelsize=40)
  #     if not os.path.exists('./log/raster/'):
  #         os.makedirs('./log/raster/')
  #     save_results_to = './log/raster/'
  #     plt.savefig(save_results_to + str(i) +'.jpeg', bbox_inches='tight')
  #     i=i+1
  #   except:
  #     pass
  ##############
  layers = ['S1_L1_SBC', 'S1_L1_ENGC', 'S1_L2_Pyr', 'S1_L2_PV', 'S1_L2_SST', 'S1_L2_VIP', 'S1_L3_Pyr', 'S1_L3_PV', 'S1_L3_SST', 'S1_L3_VIP', 'S1_L4_Pyr', 'S1_L4_PV', 'S1_L4_SST', 'S1_L5A_Pyr', 'S1_L5A_PV', 'S1_L5A_SST', 'S1_L5B_Pyr', 'S1_L5B_PV', 'S1_L5B_SST', 'S1_L6_Pyr', 'S1_L6_PV', 'S1_L6_SST']

  # poisson_frequency = 1800.0
  # for model_name, neurons in ctx_layers.items():
  #   pg = nest.Create('poisson_generator', params={'rate': poisson_frequency})
    
  #   nest.Connect(pg, neurons)
  #   print("connected poisson generator to " + model_name)

  # Define a dictionary with specific Poisson rates for each layer
  poisson_rates = {
      'S1_L1_SBC': 1500.0,
      'S1_L1_ENGC': 1800.0,
      'S1_L2_Pyr': 2000.0,
      'S1_L2_PV': 1700.0,
      'S1_L2_SST': 1600.0,
      'S1_L2_VIP': 1900.0,
      'S1_L3_Pyr': 2100.0,
      'S1_L3_PV': 1800.0,
      'S1_L3_SST': 1700.0,
      'S1_L3_VIP': 2000.0,
      'S1_L4_Pyr': 2200.0,
      'S1_L4_PV': 1900.0,
      'S1_L4_SST': 1800.0,
      'S1_L5A_Pyr': 2300.0,
      'S1_L5A_PV': 2000.0,
      'S1_L5A_SST': 1900.0,
      'S1_L5B_Pyr': 2400.0,
      'S1_L5B_PV': 2100.0,
      'S1_L5B_SST': 2000.0,
      'S1_L6_Pyr': 5000.0,
      'S1_L6_PV': 2200.0,
      'S1_L6_SST': 2100.0
  }

  # Loop through the layers and assign Poisson generators with specific rates
  for model_name, neurons in ctx_layers.items():
      poisson_frequency = poisson_rates.get(model_name, 1800.0)  # Default to 1800.0 if not specified
      pg = nest.Create('poisson_generator', params={'rate': poisson_frequency})
      nest.Connect(pg, neurons, syn_spec={'weight': 5.0, 'delay': 1.5})
      print(f"Connected Poisson generator to {model_name} with rate {poisson_frequency} Hz")

  spike_recorders = {layer: nest.Create("spike_recorder") for layer in layers}

  for layer in layers:
      nest.Connect(ctx_layers[layer], spike_recorders[layer])

  # detectors = {}
  # for layer_name in ctx_layers.keys():
  #   detectors[layer_name] = nest_routine.layer_spike_detector(ctx_layers[layer_name], layer_name,
  #                                                             sim_params['initial_ignore'])

  # macro_circle_center_params = nest_routine.get_macro_channel_centers(sim_params, hex_center=[0, 0],
  #                                                                     ci=sim_params['macro_columns_nb'],
  #                                                                     hex_radius=sim_params['hex_radius'])
  # micro_circle_center_params = nest_routine.get_micro_channel_centers(macro_circle_center_params,
  #                                                                     ci=sim_params['micro_columns_nb'],
  #                                                                     hex_radius=sim_params[
  #                                                                                  'channels_radius'] * 2. / 3.)
#   ctx_S1_macro_columns_gids = {}
#   ctx_S1_micro_columns_gids = {}

#   for layer_name in ctx_layers.keys():
#     ctx_S1_macro_columns_gids[layer_name] = nest_routine.get_macro_columns_data(layer_name, macro_circle_center_params,sim_params['channels_radius'], layer_name)
#     for i in range(len(macro_circle_center_params)):
#       ctx_S1_micro_columns_gids[layer_name + '_macro_column_' + str(i)] = nest_routine.get_micro_columns_data(
#         layer_name, micro_circle_center_params['macro_channel_' + str(i)], sim_params['channels_radius'] / 3.,
#         layer_name + '_macro_column_' + str(i))

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

  #   print(ctx_S1_macro_columns_activity[layer_name])
  #   for i in range(sim_params['macro_columns_nb']):
  #     ctx_S1_micro_columns_activity[layer_name + '_macro_column_' + str(i)] = nest_routine.get_firing_rate_micro_column(layer_name, ctx_S1_micro_columns_gids[layer_name + '_macro_column_' + str(i)], start_time, end_time, i)
  #     print (ctx_S1_micro_columns_activity[layer_name + '_macro_column_' + str(i)])

  print('Simulation debrief:')
  # for layer_name in ctx_layers.keys():
  #   rate = nest_routine.get_firing_rate_from_gdf_files(layer_name, detectors[layer_name], sim_params['simDuration'], nest_routine.count_layer(ctx_layers[layer_name]))
  #   print('Layer ' + layer_name + " fires at " + str(rate) + " Hz")

  import numpy as np

  start_time = 400.0
  end_time = 1000.0


  simulation_time = end_time - start_time  

  spike_frequencies = {}
  neuron_types = {}

  for layer, neurons in ctx_params['S1']['neuro_info'].items():
      for neuron_type, params in neurons.items():
          neuron_types[f'S1_{layer}_{neuron_type}'] = params['EorI']

  for layer, recorder in spike_recorders.items():
    
      events = recorder.get("events")
      senders = events["senders"]
      times = events["times"]
      
      mask = (times >= start_time) & (times <= end_time)
      filtered_senders = senders[mask]
      
      unique_senders, spike_counts = np.unique(filtered_senders, return_counts=True)
      
      frequencies = spike_counts / (simulation_time / 1000.0)  
      
      spike_frequencies[layer] = {
          "mean_frequency": np.mean(frequencies),
          "frequencies": dict(zip(unique_senders, frequencies))  
      }

  print("Poisson Frequencies:", poisson_frequency)
  for layer, data in spike_frequencies.items():
      print(f"Layer: {layer}, {neuron_types.get(layer)}, Mean Spike Frequency: {data['mean_frequency']:.2f} Hz")


print('Running.......')

if __name__ == '__main__':
    main()
