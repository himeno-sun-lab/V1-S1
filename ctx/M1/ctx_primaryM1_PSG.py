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

# This is the same "ctx_primaryM1" which simulates M1 completely plus dedicated one poisson spike generator
# for each neuron type to produce resting activity for M1

import fetch_params
import ini_all
import nest_routine
import nest
import nest.topology as ntop
import numpy as np
import time
import collections
from nest.lib.hl_api_info import SetStatus
import numpy as np
import pandas as pd
import math
import os
import nest.raster_plot
import nest.voltage_trace
import matplotlib.pyplot as plt
import pylab

#################
##  functions  ##
#################
def read_sim():
  try:
    from runpy import run_path
    file_params = run_path('baseSimParams.py', init_globals=globals())
    sim_params = file_params['simParams']
    return sim_params
  except:
    raise ImportError('The simulation parameters could not be loaded. Please make sure that the file `simParams.py` exists and is a valid python defining the variable "simParams".')

##################
def read_ctx_M1():
    try:
        from runpy import run_path
        #file_params = run_path('baseCTXM1Params.py', init_globals=globals())         #This os the file which reads the initial values from first confirmed connection parameters
        file_params = run_path( 'baseCTXM1Params_resting.py', init_globals=globals() )   #Here initila parameters are changing to find out best condition for the resting state
        ctx_M1_params = file_params['ctxM1Params']
        return ctx_M1_params
    except:
        raise ImportError(
            'The cortex-M1-region parameters could not be loaded. Please make sure that the file `baseCTXM1Params.py` exists and is a valid python defining the variable "ctxM1Params".')

################################
def initialize_nest(sim_params):
  nest.set_verbosity("M_WARNING")
  nest.SetKernelStatus({"overwrite_files": sim_params['overwrite_files']}) # should we erase previous traces when redoing a simulation?
  nest.SetKernelStatus({'local_num_threads': int(sim_params['nbcpu'])})
  nest.SetKernelStatus({"data_path": 'log/logPSG'})
  if sim_params['dt'] != '0.1':
    nest.SetKernelStatus({'resolution': float(sim_params['dt'])})

#############################################################
def copy_neuron_model(elements, neuron_info, new_model_name):
  configuration = {}
  # Membrane potential in mV
  configuration ['V_m'] = 0.0
  # Leak reversal Potential (aka resting potential) in mV
  configuration ['E_L'] = -70.0
  # Membrane Capacitance in pF
  configuration ['C_m'] = 250.0
  # Refractory period in ms
  configuration ['t_ref'] = float( neuron_info ['absolute_refractory_period'] )
  # Threshold Potential in mV
  configuration ['V_th'] = float( neuron_info ['spike_threshold'] )
  # Reset Potential in mV
  configuration ['V_reset'] = float( neuron_info ['reset_value'] )
  # Excitatory reversal Potential in mV
  configuration ['E_ex'] = float( neuron_info ['E_ex'] )
  # Inhibitory reversal Potential in mV
  configuration ['E_in'] = float( neuron_info ['E_in'] )
  # Leak Conductance in nS
  configuration ['g_L'] = 250. / float( neuron_info ['membrane_time_constant'] )
  # Time constant of the excitatory synaptic exponential function in ms
  configuration ['tau_syn_ex'] = float( neuron_info ['tau_syn_ex'] )
  # Time constant of the inhibitory synaptic exponential function in ms
  configuration ['tau_syn_in'] = float( neuron_info ['tau_syn_in'] )
  # Constant Current in pA
  configuration ['I_e'] = float( neuron_info ['I_ex'] )
  nest.CopyModel( elements, new_model_name, configuration )
  return new_model_name

############################################################################################
def get_connection_summary(region_params, layers):
    import matplotlib.pyplot as plt
    from scipy import stats
    M1_internal_connection = np.load( '../../ctx/M1/' + region_params ['M1'] ['connection_info'] ['M1toM1'] )
    columns = ['pre_nodes_num', 'post_nodes_num', 'p_center', 'sigma', 'analogical_conn_num', 'conn_num', 'weight_per_neuron', 'weight_per_connection']

    for pre_l in layers.keys():
      rows_out = []
      rows_in = []
      cell_text_out = []
      cell_text_in = []
      conn_num_out_excitatory = 0.
      conn_num_out_inhibitory = 0.
      conn_num_in_excitatory = 0.
      conn_num_in_inhibitory = 0.
      for post_l in layers.keys():
        print( pre_l, post_l )
        p_center_out = M1_internal_connection [pre_l] [post_l] ['p_center']
        sigma_out = M1_internal_connection [pre_l] [post_l] ['sigma'] / 1000.
        p_center_in = M1_internal_connection [post_l] [pre_l] ['p_center']
        sigma_in = M1_internal_connection [post_l] [pre_l] ['sigma'] / 1000.
        analogical_total_conn_num = 0.0
        pre_l_nodes = nest.GetNodes( layers [pre_l] ) [0]
        post_l_nodes = nest.GetNodes( layers [post_l] ) [0]
        # out-degree
        analogical_total_conn_num = 0.0
        E_weight_per_conn_out = 0.0
        E_weight_per_neuron_out = 0.0
        I_weight_per_conn_out = 0.0
        I_weight_per_neuron_out = 0.0

        E_weight_per_conn_in = 0.0
        E_weight_per_neuron_in = 0.0
        I_weight_per_conn_in = 0.0
        I_weight_per_neuron_in = 0.0

        if p_center_out != 0. and sigma_out != 0.:
          rows_out.append( post_l )
          print( "layer " + pre_l + " out-degree output" )
          connectome = nest.GetConnections( pre_l_nodes, post_l_nodes )
          connection_weights = list( nest.GetStatus( connectome, "weight" ) )
          weight_per_connection = 0
          if len( connection_weights ) > 0:
            weight_per_connection = sum( connection_weights ) / len( connection_weights )
          weight_per_neuron = 0.0
          weight_per_neuron = sum( connection_weights ) / len( pre_l_nodes )
          conn_num_out = len( connectome )
          analogical_total_conn_num = len(post_l_nodes)*p_center_out * (stats.norm( scale=sigma_out).cdf(sigma_out * 3.)- stats.norm(scale=sigma_out).cdf(-sigma_out * 3.))
          cell_text_out.append([len( pre_l_nodes ), len( post_l_nodes ), M1_internal_connection [pre_l] [post_l] ['p_center'], M1_internal_connection [pre_l] [post_l] ['sigma'], analogical_total_conn_num,
             conn_num_out / len( pre_l_nodes ), weight_per_neuron, weight_per_connection] )

          if post_l in ['M1_L23_CC', 'M1_L5A_CC', 'M1_L5A_CS', 'M1_L5A_CT', 'M1_L5B_CC', 'M1_L5B_CS', 'M1_L5B_PT', 'M1_L6_CT']:
            conn_num_out_excitatory += conn_num_out
            E_weight_per_conn_out += weight_per_connection
            E_weight_per_neuron_out += weight_per_neuron
          else:
            conn_num_out_inhibitory += conn_num_out
            I_weight_per_conn_out += weight_per_connection
            I_weight_per_neuron_out += weight_per_neuron
        # in- degree
        analogical_total_conn_num = 0.0
        if p_center_in != 0. and sigma_in != 0.:
          rows_in.append( post_l )
          print( "layer " + pre_l + " in-degree output" )
          connectome = nest.GetConnections( post_l_nodes, pre_l_nodes )
          connection_weights = list( nest.GetStatus( connectome, "weight" ) )
          conn_num_in = len( connectome )
          weight_per_connection = 0
          if len( connection_weights ) > 0:
            weight_per_connection = sum( connection_weights ) / len( connection_weights )
          weight_per_neuron = 0.0
          weight_per_neuron = sum( connection_weights ) / len( post_l_nodes )
          analogical_total_conn_num = len( pre_l_nodes ) * p_center_in * (
                    stats.norm( scale=sigma_in ).cdf( sigma_in * 3. ) - stats.norm( scale=sigma_in ).cdf(-sigma_in * 3. ))
          cell_text_in.append(
            [len( post_l_nodes ), len( pre_l_nodes ), M1_internal_connection [post_l] [pre_l] ['p_center'],
             M1_internal_connection [post_l] [pre_l] ['sigma'], analogical_total_conn_num,
             conn_num_in / len( pre_l_nodes ), weight_per_neuron, weight_per_connection] )
          if post_l in ['M1_L23_CC', 'M1_L5A_CC', 'M1_L5A_CS', 'M1_L5A_CT', 'M1_L5B_CC', 'M1_L5B_CS', 'M1_L5B_PT', 'M1_L6_CT']:
            conn_num_in_excitatory += conn_num_in
            E_weight_per_conn_in += weight_per_connection
            E_weight_per_neuron_in += weight_per_neuron
          else:
            conn_num_in_inhibitory += conn_num_in
            I_weight_per_conn_in += weight_per_connection
            I_weight_per_neuron_in += weight_per_neuron
      # table_out = plt.figure()
      if len( rows_out ) > 0:
        columns [5] = 'conn_num/pre_neuron_nb'
        rows_out.append( 'Total Excitatory' )
        cell_text_out.append(
          ['/', '/', '/', '/', '/', conn_num_out_excitatory / len( pre_l_nodes ), E_weight_per_neuron_out, E_weight_per_conn_out] )
        rows_out.append( 'Total Inhibitory' )
        cell_text_out.append(
          ['/', '/', '/', '/', '/', conn_num_out_inhibitory / len( pre_l_nodes ), I_weight_per_neuron_out, I_weight_per_conn_out] )

        rows_out.append( 'Total' )
        cell_text_out.append( ['/', '/', '/', '/', '/',
                               conn_num_out_excitatory / len( pre_l_nodes ) + conn_num_out_inhibitory / len(
                                 pre_l_nodes ), E_weight_per_neuron_out + I_weight_per_neuron_out, E_weight_per_conn_out + I_weight_per_conn_out] )
        rows_out.append( 'I/E' )
        if conn_num_out_excitatory > 0. and E_weight_per_neuron_out != 0 and E_weight_per_conn_out != 0:
          cell_text_out.append( ['/', '/', '/', '/', '/', conn_num_out_inhibitory / conn_num_out_excitatory,
                                 I_weight_per_neuron_out / E_weight_per_neuron_out, I_weight_per_conn_out / E_weight_per_conn_out] )
        else:
          cell_text_out.append( ['/', '/', '/', '/', '/', '/', 0.0, 0.0] )

        fig = plt.figure( figsize=(20, 10) )
        ax = plt.subplot( 111, frame_on=False )
        ax.xaxis.set_visible( False )
        ax.yaxis.set_visible( False )
        ax.title.set_text( pre_l + ' cell out degree connections' )
        plt.table( cellText=cell_text_out, rowLabels=rows_out, colLabels=columns, loc='center' )
        plt.savefig( 'log/' + pre_l + '_cell_out_degree_connections.png' )
      # table_in = plt.figure()
      if len( rows_in ) > 0:
        columns [5] = 'conn_num/post_neuron_nb'
        rows_in.append( 'Total Excitatory' )
        cell_text_in.append(['/', '/', '/', '/', '/', conn_num_in_excitatory / len( pre_l_nodes ), E_weight_per_neuron_in, E_weight_per_conn_in] )
        rows_in.append( 'Total Inhibitory' )
        cell_text_in.append(['/', '/', '/', '/', '/', conn_num_in_inhibitory / len( pre_l_nodes ), I_weight_per_neuron_in, I_weight_per_conn_in] )
        rows_in.append( 'Total' )
        cell_text_in.append( ['/', '/', '/', '/', '/', conn_num_in_excitatory / len( pre_l_nodes ) + conn_num_in_inhibitory / len( pre_l_nodes ),
                              E_weight_per_neuron_in + I_weight_per_neuron_in, E_weight_per_conn_in + I_weight_per_conn_in] )
        rows_in.append( 'I/E' )
        if conn_num_in_excitatory > 0. and E_weight_per_neuron_in != 0 and E_weight_per_conn_in != 0:
          cell_text_in.append( ['/', '/', '/', '/', '/', conn_num_in_inhibitory / conn_num_in_excitatory,
                                I_weight_per_neuron_in / E_weight_per_neuron_in, I_weight_per_conn_in / E_weight_per_conn_in] )
        else:
          cell_text_in.append( ['/', '/', '/', '/', '/', '/', 0.0, 0.0] )

        fig = plt.figure( figsize=(20, 10) )
        ax = plt.subplot( 111, frame_on=False )
        ax.xaxis.set_visible( False )
        ax.yaxis.set_visible( False )
        ax.title.set_text( pre_l + ' cell in degree connections' )
        plt.table( cellText=cell_text_in, rowLabels=rows_in, colLabels=columns, loc='center' )
        plt.savefig( 'log/' + pre_l + '_cell_in_degree_connections.png' )

#############################################################################################
def gen_neuron_postions_ctx(layer_thickness, nbneuron, M1_layer_size, scalefactor, pop_name):
  neuron_per_grid = math.pow((nbneuron / layer_thickness), 1.0 / 3)
  Sub_Region_Architecture = [0, 0, 0]
  Sub_Region_Architecture [0] = int(np.round( neuron_per_grid * M1_layer_size[0] * scalefactor[0]))
  Sub_Region_Architecture [1] = int(np.round( neuron_per_grid * M1_layer_size[1] * scalefactor[1]))
  Sub_Region_Architecture [2] = int(np.round( neuron_per_grid * layer_thickness))

  Neuron_pos_x = np.linspace( -0.5 * scalefactor [0], 0.5 * scalefactor [0], num=Sub_Region_Architecture [0], endpoint=True )
  Neuron_pos_y = np.linspace( -0.5 * scalefactor [1], 0.5 * scalefactor [1], num=Sub_Region_Architecture [1], endpoint=True )
  Neuron_pos_z = np.linspace( -layer_thickness / 2., layer_thickness / 2., num=Sub_Region_Architecture [2], endpoint=False )

  Neuron_pos = []
  for i in range( Sub_Region_Architecture [0] ):
    for j in range( Sub_Region_Architecture [1] ):
      for k in range( Sub_Region_Architecture [2] ):
        Neuron_pos.append( [Neuron_pos_x [i], Neuron_pos_y [j], Neuron_pos_z [k]] )
  #np.savez( 'ctx' + pop_name, Neuron_pos=Neuron_pos )
  return Neuron_pos

###########################################################
def create_layers_ctx(extent, center, positions, elements):
  # Neuron_pos_list=positions[:, :3].tolist()
  # nest.SetDefaults(elements, {"I_e": float(neuron_info['I_ex']),
  #                             "V_th": float(neuron_info['spike_threshold']),
  #                             "V_reset": float(neuron_info['reset_value']),
  #                             "t_ref": float(neuron_info['absolute_refractory_period']),
  #                             "g_L": 250./float(neuron_info['membrane_time_constant']),
  #                             "E_L":float(neuron_info['E_rest']), \
  #                             "E_ex": float(neuron_info['E_ex']),\
  #                             "E_in": float(neuron_info['E_in']), \
  #                             "tau_syn_ex": float(neuron_info['tau_syn_ex']), \
  #                             "tau_syn_in": float(neuron_info['tau_syn_in'])})
  newlayer = ntop.CreateLayer( {'extent': extent, 'center': center, 'positions': positions, 'elements': elements} )
  # Neurons = nest.GetNodes(newlayer)
  return newlayer

##############################################################
def create_layers_ctx_M1(extent, center, positions, elements):
  newlayer = ntop.CreateLayer( {'extent': extent, 'center': center, 'positions': positions, 'elements': elements} )
  return newlayer

##########################################################################
def connect_layers_ctx_M1(pre_SubSubRegion, post_SubSubRegion, conn_dict):
  sigma_x = conn_dict ['sigma'] / 1000.
  sigma_y = conn_dict ['sigma'] / 1000.
  weight_distribution = conn_dict ['weight_distribution']
  if weight_distribution == 'lognormal':
    conndict = {'connection_type': 'divergent',
                'mask': {'spherical': {'radius': 2.0}},
                'kernel': {'gaussian2D': {'p_center': float( conn_dict ['p_center'] ), 'sigma_x': sigma_x, 'sigma_y': sigma_y}},
                'weights': {'lognormal': {'mu': float( conn_dict ['weight'] ), 'sigma': 1.0}},
                'delays': float( conn_dict ['delay'] ),
                'allow_autapses': False,
                'allow_multapses': False}
  else:
    conndict = {'connection_type': 'divergent',
                'mask': {'spherical': {'radius': 2.0}},
                'kernel': {'gaussian2D': {'p_center': float( conn_dict ['p_center'] ), 'sigma_x': sigma_x, 'sigma_y': sigma_y}},
                'weights': float( conn_dict ['weight'] ),
                'delays': float( conn_dict ['delay'] ),
                'allow_autapses': False,
                'allow_multapses': False}
  if conn_dict['p_center'] != 0.0 and sigma_x != 0.0 and conn_dict['weight'] != 0.0:
    ntop.ConnectLayers( pre_SubSubRegion, post_SubSubRegion, conndict )


###########################################################
def save_layers_position(layer_name, layer_gid, positions):
  np.array(nest.GetNodes(layer_gid)[0])
  gid_and_positions = np.column_stack((np.array( nest.GetNodes(layer_gid)[0]), positions))
  np.savetxt( 'log/logPSG/' + layer_name + '.txt', gid_and_positions, fmt='%1.3f' )


###################################################
def instantiate_ctx_M1(ctx_M1_params, scalefactor):
  region_name = 'M1'
  # set the parameters for M1 model
  M1_Layer_Name = ctx_M1_params [region_name] ['structure_info'] ['Layer_Name']
  M1_layer_size = ctx_M1_params [region_name] ['structure_info'] ['region_size']
  M1_layer_size = np.array( M1_layer_size )
  M1_layer_thickness = ctx_M1_params ['M1'] ['structure_info'] ['layer_thickness']
  topo_extend = [M1_layer_size [0] * int( scalefactor [0] ) + 1., M1_layer_size [1] * int( scalefactor [1] ) + 1., M1_layer_size [2] + 1.]
  topo_center = [0.0, 0.0, 0.0]
  SubSubRegion_Excitatory = []
  SubSubRegion_Inhibitory = []
  SubSubRegion_Excitatory_ntype = []
  SubSubRegion_Inhibitory_ntype = []
  ctx_M1_layers = {}
  for l in range( len(M1_Layer_Name)):
    print( '###########################################')
    print( 'start to create layer in M1: ' + M1_Layer_Name [l] )
    ## Fix for multiple nodes ##
    ctx_M1_params [region_name] ['neuro_info'] [M1_Layer_Name [l]] = collections.OrderedDict(sorted( ctx_M1_params [region_name] ['neuro_info'] [M1_Layer_Name [l]].items(), key=lambda t: t [0] ) )
    ##########################
    for n_type in ctx_M1_params[region_name]['neuro_info'][M1_Layer_Name [l]].keys():
      n_type_index = ctx_M1_params [region_name]['neuro_info'][M1_Layer_Name [l]][n_type] ['n_type_index']
      print(M1_Layer_Name[l])
      print('n_type_index:', n_type_index)
      print(n_type )
      n_type_info = ctx_M1_params [region_name] ['neuro_info'] [M1_Layer_Name [l]] [n_type]
      neuronmodel = copy_neuron_model(ctx_M1_params [region_name] ['neuro_info'] [M1_Layer_Name [l]] [n_type] ['neuron_model'], n_type_info, region_name + '_' + M1_Layer_Name [l] + '_' + n_type)
      nbneuron = n_type_info ['Cellcount_mm2']
      Neuron_pos = gen_neuron_postions_ctx(M1_layer_thickness [l], nbneuron, M1_layer_size, scalefactor, 'Neuron_pos_' + region_name + '_' + M1_Layer_Name [l] + '_' + n_type)
      if n_type_info ['EorI'] == "E":
        SubSubRegion_Excitatory.append(create_layers_ctx_M1(topo_extend, topo_center, Neuron_pos, neuronmodel))
        SubSubRegion_Excitatory_ntype.append([M1_Layer_Name [l], n_type])
        ctx_M1_layers [region_name + '_' + M1_Layer_Name [l] + '_' + n_type] = SubSubRegion_Excitatory [-1]
        save_layers_position(region_name + '_' + M1_Layer_Name[l] + '_' + n_type, SubSubRegion_Excitatory [-1], Neuron_pos)
      elif n_type_info ['EorI'] == "I":
        SubSubRegion_Inhibitory.append(create_layers_ctx_M1( topo_extend, topo_center, Neuron_pos, neuronmodel ) )
        SubSubRegion_Inhibitory_ntype.append( [M1_Layer_Name [l], n_type] )
        ctx_M1_layers [region_name + '_' + M1_Layer_Name [l] + '_' + n_type] = SubSubRegion_Inhibitory [-1]
        save_layers_position(region_name + '_' + M1_Layer_Name[l] + '_' + n_type, SubSubRegion_Inhibitory [-1], Neuron_pos)
      else:
        print( 'Error: Unknow E or I' )

  print( "Start to connect the layers" )
  # M1_internal_connection = np.load( 'ctx/' + ctx_M1_params['M1']['connection_info']['M1toM1'])
  ctx_M1_internal_connection = np.load( ctx_M1_params ['M1'] ['connection_info'] ['M1toM1'] )
  from collections import defaultdict
  for pre_layer_name in ctx_M1_layers.keys():
    for post_layer_name in ctx_M1_layers.keys():
      print( 'start to connect ' + pre_layer_name + ' with ' + post_layer_name )
      connect_layers_ctx_M1(ctx_M1_layers [pre_layer_name], ctx_M1_layers [post_layer_name], ctx_M1_internal_connection [pre_layer_name] [post_layer_name] )
  return ctx_M1_layers


#############################################################################################################
def layer_spike_detector(layer_gid, layer_name, ignore_time, params={"withgid": True, "withtime": True, "to_file": True}):
#def layer_spike_detector(layer_gid, layer_name, params={"withgid": True, "withtime": True, "to_file": True, 'fbuffer_size': 8192}):
  print ('spike detector for '+layer_name)
  params.update({'label': layer_name, "start": float(ignore_time)}) #  , "start": float(ignore_time)
  ini_time = ignore_time - 150
  ini_time_ini = ignore_time - 300
  # first adding poisson generator one for all neuron types different for the inhibitory and excitatory neurons
  name_list = ['M1_L23_CC', 'M1_L5A_CC', 'M1_L5A_CT', 'M1_L5B_CC', 'M1_L5B_CS', 'M1_L6_CT']
  if layer_name in name_list:
    PSG = nest.Create('poisson_generator', 1, params={'rate':  30.0, "start": float(ini_time)}) #, 'label': layer_name})
    nest.Connect(pre=PSG, post=nest.GetNodes(layer_gid)[0], syn_spec={'weight': 50.0, 'delay': 1.5})
  elif layer_name == 'M1_L5B_PT':
    PSG = nest.Create('poisson_generator', 1, params={'rate': 90.0,"start": float(ini_time)}) #, 'label': layer_name})           # specific PSG for L5B_PT neurons
    nest.Connect(pre=PSG, post=nest.GetNodes(layer_gid)[0], syn_spec={'weight': 50.0, 'delay': 1.5})
  elif layer_name == 'M1_L5A_CS':
    PSG = nest.Create('poisson_generator', 1, params={'rate': 70.0,"start": float(ini_time)}) #, 'label': layer_name})           # specific PSG for L5B_PT neurons
    nest.Connect(pre=PSG, post=nest.GetNodes(layer_gid)[0], syn_spec={'weight': 50.0, 'delay': 1.5})
  else:
    print(layer_name)
    PSG = nest.Create('poisson_generator', 1, params={'rate': 40.0, "start": float(ini_time_ini)}) #, 'label': layer_name})
    nest.Connect(pre=PSG, post=nest.GetNodes(layer_gid)[0], syn_spec={'weight': 50.0, 'delay': 1.5})

  # Add detector for all neuron types
  detector = nest.Create("spike_detector", params= params)
  nest.Connect(pre=nest.GetNodes(layer_gid)[0], post=detector)

  # add multimeter just for to record V_m and conductance (inhibitory and excitatory) of a single cell each cell population
  nest_mm = nest.Create('multimeter', params = {"interval": 0.1, "record_from": ["V_m", "g_ex", "g_in"], "withgid": True, "to_file": True,
                                                'label': layer_name, 'withtime': True, "start": float(ignore_time),  'use_gid_in_filename': True})  #
  nest.Connect(nest_mm, [nest.GetNodes(layer_gid)[0][7]])


  # voltmeter_con=nest.Create("voltmeter", params = {'withgid':True, 'withtime':True})
  voltmeter_single=nest.Create("voltmeter", params={'to_file': False, 'label': layer_name, "start": float(ignore_time), "withgid":True,  "withtime":True})   # 'use_gid_in_filename': True,   "start": float(ignore_time),
  # nest.Connect(pre=voltmeter_con, post= nest.GetNodes(layer_gid)[0])
  nest.Connect(voltmeter_single, [nest.GetNodes(layer_gid)[0][7]])

  return detector, nest_mm, voltmeter_single, PSG

#########################################
def average_fr(detector, simDuration, n):
  return nest.GetStatus(detector, 'n_events')[0] / (float(simDuration) * float(n) / 1000.)

###########################
def count_layer(layer_gid):
  return len(nest.GetNodes(layer_gid)[0])



########################################################################################################################
#################
#  main script  #
#################
# 1) reads parameters
print('Reading the simulation parameters using "baseSimParameters.py" file')
sim_params = read_sim()
print('Reading the M1 parameters using "baseCTXM1Parameters.py" file')
ctx_M1_params = read_ctx_M1()

# 2) initialize nest
print('Nest Initializations')
initialize_nest(sim_params)

# 3) instantiates regions
print('create and connect the M1 layers')
start_time = time.time()
ctx_M1_layers =instantiate_ctx_M1(ctx_M1_params, sim_params['scalefactor'])
with open('./log/logPSG/' + 'performance.txt', 'a') as file:
  file.write('M1_Construction_Time: ' + str(time.time() - start_time) + '\n')
  # _=get_connection_summary(ctx_params, ctx_layers)

# 4) get summary of connections
#get_connection_summary(ctx_M1_params, ctx_M1_layers)

# 5) detectors
poisson_gens = {}
detectors = {}
multimeters = {}
voltmeters = {}
start_time = time.time()
for layer_name in ctx_M1_layers.keys():
  detectors[layer_name], multimeters[layer_name], voltmeters[layer_name], poisson_gens[layer_name] = layer_spike_detector(ctx_M1_layers[layer_name], layer_name, sim_params['initial_ignore'])
with open('./log/logPSG/' + 'performance.txt', 'a') as file:
  file.write('Detectors_Elapse_Time: ' + str(time.time() - start_time ) + '\n')

# 6) randomizing the membarne potential
import numpy
for layer_name in ctx_M1_layers.keys():
  Vth = -50.
  Vrest = -70.
  for neuron in nest.GetNodes(ctx_M1_layers[layer_name])[0]:
    nest.SetStatus( [neuron], {"V_m": Vrest + (Vth - Vrest) * numpy.random.rand()} )


# 6) simulation
simulation_time = sim_params['simDuration']+ sim_params['initial_ignore']
print ('Simulation Started:')
start_time=time.time()
nest.Simulate(simulation_time)
with open('./log/logPSG/'+'performance.txt', 'a') as file:
  file.write('Simulation_Elapse_Time: '+str(time.time()-start_time)+'\n')

# 7) results
for layer_name in ctx_M1_layers.keys():
  rate = average_fr(detectors[layer_name], sim_params['simDuration'], count_layer(ctx_M1_layers[layer_name]))
  print('Layer '+layer_name+" fires at "+str(rate)+" Hz")
  with open( './log/logPSG/' + 'report.txt', 'a' ) as file:
    file.write('Layer '+layer_name+" fires at "+str(rate)+" Hz" + '\n' )
    #file.write(str( rate ) + " Hz" + '\n' )

print ('Simulation successfully finished!')
print ('Congradulation!!!!!!')

#################################
# to plot some graphs
#nest.raster_plot.from_device(detectors['M1_L1_ENGC'], hist=True, hist_binwidth=0.1)
#nest.voltage_trace.from_device(voltmeter)
#################################

'''
###############
# setting for figs size and font
font = {'family' : 'normal', 'weight' : 'normal', 'size'   : 42}
plt.rcParams["figure.figsize"] = [64,48]
plt.rc('font', **font)
plt.rc('xtick', labelsize=40) 
plt.rc('ytick', labelsize=40) 
###############
# plotting the garphs
i=1
for layer_name in ctx_M1_layers.keys():
  plt.figure()
  nest.voltage_trace.from_device(voltmeters[layer_name], title=layer_name)
  plt.show()
  save_results_to = '/home/morteza/Desktop/reports/190301/3-1930/mp/'
  plt.savefig(save_results_to + str(i) +'.jpeg')
  i=i+1

###############
i=1
for layer_name in ctx_M1_layers.keys():
  plt.figure()
  events = nest.GetStatus(multimeters[layer_name])[0]["events"]
  t = events["times"]
  plt.plot(t, events["g_ex"], t, events["g_in"])
  plt.xlabel("time (ms)")
  plt.ylabel("synaptic conductance (nS)")
  plt.title(layer_name)
  plt.legend(("g_exc", "g_inh"))
  save_results_to = '/home/morteza/Desktop/reports/190301/3-1930/con/'
  plt.savefig(save_results_to + str(i) +'.jpeg')
  i=i+1
  
###############
i=1
for layer_name in ctx_M1_layers.keys():
  nest.raster_plot.from_device(detectors[layer_name], title = layer_name, hist=False)
  font = {'family' : 'normal', 'weight' : 'normal', 'size'   : 42}
  plt.rcParams["figure.figsize"] = [64,48]
  plt.rc('font', **font)
  plt.rc('xtick', labelsize=40) 
  plt.rc('ytick', labelsize=40)  
  #save_results_to = '/home/morteza/Desktop/reports/190301/3-1930/raster/'
  #plt.savefig(save_results_to + str(i) +'.jpeg', bbox_inches='tight')
  i=i+1

###############

'''
'''
if __name__ == '__main__':
    main()
'''

'''
##############################################
# 6) deleting all empty files
import  os
path = "/home/morteza/PycharmProjects/postk_wb/code/ctx/M1/log/logPSG"
file_num = 0
for root,dirs,files in os.walk(path):
    for name in files:
        filename = os.path.join(root,name)
        if os.stat(filename).st_size == 0:
            print(" Removing ",filename)
            os.remove(filename)
            file_num = file_num +1

'''