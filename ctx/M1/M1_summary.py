#!/usr/bin/env python
# -*- coding: utf-8 -*-

# This is the all program for simulation of "primary motor cortex" including 6 layers
# eventhough L4 is somehow neglectable for M1

# this is the program which uses the "baseCTXM1Params.py" file as a initial values for the M1 and also
# 190512 "M1_internal_connection.pickle" file as a source for the connection matrix to reveal all connection parameters of M1

import fetch_params
import ini_all
import nest_routine
import nest
# import nest.topology as ntop
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
        file_params = run_path( 'baseCTXM1Params.py', init_globals=globals() )   #Here initila parameters are changing to find out best condition for the resting state
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
    # ntop.ConnectLayers( pre_SubSubRegion, post_SubSubRegion, conndict )


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


#########################################
def get_connection_summary(region_params, layers):
    import matplotlib.pyplot as plt
    from scipy import stats
    M1_internal_connection = np.load( '../../ctx/M1/' + region_params ['M1'] ['connection_info'] ['M1toM1'] )
    columns = ['pre_nodes_num', 'post_nodes_num', 'p_center', 'sigma', 'analy_conn_num', 'conn_num', 'weight_per_neuron', 'weight_per_connection']
    for post_l in layers.keys():
      rows_in = []
      cell_text_in = []
      conn_num_in_excitatory = 0.
      conn_num_in_inhibitory = 0.
      E_weight_per_conn_in = 0.0
      E_weight_per_neuron_in = 0.0
      I_weight_per_conn_in = 0.0
      I_weight_per_neuron_in = 0.0
      for pre_l in layers.keys():
        print( pre_l, post_l )
        p_center_in = M1_internal_connection [pre_l] [post_l] ['p_center']
        sigma_in = M1_internal_connection [pre_l] [post_l] ['sigma'] / 1000.
        # analytical_total_conn_num = 0.0
        pre_l_nodes = nest.GetNodes( layers [pre_l] ) [0]
        post_l_nodes = nest.GetNodes( layers [post_l] ) [0]
        analogical_total_conn_num = 0.0
        if p_center_in != 0. and sigma_in != 0.:
          rows_in.append( pre_l )
          print( "layer " + post_l + " in-degree output" )
          connectome = nest.GetConnections( pre_l_nodes, post_l_nodes )
          connection_weights = list( nest.GetStatus( connectome, "weight" ) )
          conn_num_in = len( connectome )
          weight_per_connection = 0
          if len( connection_weights ) > 0:
            weight_per_connection = sum( connection_weights ) / len( connection_weights )
          weight_per_neuron = 0.0
          weight_per_neuron = sum( connection_weights ) / len( post_l_nodes )
          analytical_total_conn_num = len( pre_l_nodes ) * p_center_in * np.sqrt(sigma_in * 2. * np.pi ) * p_center_in * np.sqrt( sigma_in * 2. * np.pi )
          cell_text_in.append([len( pre_l_nodes ), len( post_l_nodes ), M1_internal_connection [pre_l] [post_l] ['p_center'],
             M1_internal_connection [pre_l] [post_l] ['sigma'], analytical_total_conn_num, conn_num_in / len( post_l_nodes ), weight_per_neuron, weight_per_connection] )
          if pre_l in ['M1_L23_CC', 'M1_L5A_CC', 'M1_L5A_CS', 'M1_L5A_CT', 'M1_L5B_CC', 'M1_L5B_CS', 'M1_L5B_PT', 'M1_L6_CT']:
            print( pre_l )
            conn_num_in_excitatory += conn_num_in
            E_weight_per_conn_in += weight_per_connection
            E_weight_per_neuron_in += weight_per_neuron
            # E_weight_neuron_nb_in += 1
          else:
            conn_num_in_inhibitory += conn_num_in
            I_weight_per_conn_in += weight_per_connection
            I_weight_per_neuron_in += weight_per_neuron
      if len( rows_in ) > 0:
        columns [5] = 'conn_num/post_neuron_nb'
        rows_in.append( 'Total Excitatory' )
        cell_text_in.append(['/', '/', '/', '/', '/', conn_num_in_excitatory / len( post_l_nodes ), E_weight_per_neuron_in, E_weight_per_conn_in] )
        rows_in.append( 'Total Inhibitory' )
        cell_text_in.append(['/', '/', '/', '/', '/', conn_num_in_inhibitory / len( post_l_nodes ), I_weight_per_neuron_in, I_weight_per_conn_in] )
        rows_in.append( 'Total' )
        cell_text_in.append( ['/', '/', '/', '/', '/', conn_num_in_excitatory / len( post_l_nodes ) + conn_num_in_inhibitory / len(post_l_nodes ), E_weight_per_neuron_in + I_weight_per_neuron_in, E_weight_per_conn_in + I_weight_per_conn_in] )
        rows_in.append( 'I/E' )
        if conn_num_in_excitatory > 0. and E_weight_per_neuron_in != 0 and E_weight_per_conn_in != 0:
          cell_text_in.append( ['/', '/', '/', '/', '/', conn_num_in_inhibitory / conn_num_in_excitatory, I_weight_per_neuron_in / E_weight_per_neuron_in, I_weight_per_conn_in / E_weight_per_conn_in] )
        else:
          cell_text_in.append( ['/', '/', '/', '/', '/', '/', 0.0, 0.0] )

        # fig = plt.figure( figsize=(20, 10) )
        # ax = plt.subplot( 111, frame_on=False )
        # ax.xaxis.set_visible( False )
        # ax.yaxis.set_visible( False )
        # ax.title.set_text( post_l + ' cell in degree connections' )
        # the_table_p = plt.table( cellText=cell_text_in, colWidths=[0.12] * 90, rowLabels=rows_in, colLabels=columns, loc='center' )
        # the_table_p.auto_set_font_size( False )
        # the_table_p.set_fontsize(10)
        # the_table_p.scale( 1.1, 1.1 )
        #plt.savefig( 'log/Conn_mat_detail/' + post_l + '_cell_in_degree_connections.jpg' )

    for pre_l in layers.keys():
      rows_out = []
      cell_text_out = []
      conn_num_out_excitatory = 0.
      conn_num_out_inhibitory = 0.
      E_weight_per_conn_out = 0.0
      E_weight_per_neuron_out = 0.0
      I_weight_per_conn_out = 0.0
      I_weight_per_neuron_out = 0.0
      for post_l in layers.keys():
        print( pre_l, post_l )
        p_center_out = M1_internal_connection [pre_l] [post_l] ['p_center']
        sigma_out = M1_internal_connection [pre_l] [post_l] ['sigma'] / 1000.
        # analytical_total_conn_num = 0.0
        pre_l_nodes = nest.GetNodes( layers [pre_l] ) [0]
        post_l_nodes = nest.GetNodes( layers [post_l] ) [0]
        # out-degree
        # analogical_total_conn_num=0.0
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
          analytical_total_conn_num = len( post_l_nodes ) * p_center_out * np.sqrt(sigma_out * 2. * np.pi ) * p_center_out * np.sqrt( sigma_out * 2. * np.pi )
          cell_text_out.append([len( pre_l_nodes ), len( post_l_nodes ), M1_internal_connection [pre_l] [post_l] ['p_center'],
             M1_internal_connection [pre_l] [post_l] ['sigma'], analytical_total_conn_num, conn_num_out / len( pre_l_nodes ), weight_per_neuron, weight_per_connection] )
          if pre_l in ['M1_L23_CC', 'M1_L5A_CC', 'M1_L5A_CS', 'M1_L5A_CT', 'M1_L5B_CC', 'M1_L5B_CS', 'M1_L5B_PT', 'M1_L6_CT']:
            conn_num_out_excitatory += conn_num_out
            E_weight_per_conn_out += weight_per_connection
            E_weight_per_neuron_out += weight_per_neuron
          else:
            print( pre_l )
            conn_num_out_inhibitory += conn_num_out
            I_weight_per_conn_out += weight_per_connection
            I_weight_per_neuron_out += weight_per_neuron
      if len( rows_out ) > 0:
        columns [5] = 'conn_num/pre_neuron_nb'
        rows_out.append( 'Total Excitatory' )
        cell_text_out.append(['/', '/', '/', '/', '/', conn_num_out_excitatory / len( pre_l_nodes ), E_weight_per_neuron_out, E_weight_per_conn_out] )
        rows_out.append( 'Total Inhibitory' )
        cell_text_out.append(['/', '/', '/', '/', '/', conn_num_out_inhibitory / len( pre_l_nodes ), I_weight_per_neuron_out, I_weight_per_conn_out] )
        rows_out.append( 'Total' )
        cell_text_out.append( ['/', '/', '/', '/', '/', conn_num_out_excitatory / len( pre_l_nodes ) + conn_num_out_inhibitory / len(pre_l_nodes ), E_weight_per_neuron_out + I_weight_per_neuron_out, E_weight_per_conn_out + I_weight_per_conn_out] )
        rows_out.append( 'I/E' )
        if conn_num_out_excitatory > 0. and E_weight_per_neuron_out != 0 and E_weight_per_conn_out != 0:
          cell_text_out.append( ['/', '/', '/', '/', '/', conn_num_out_inhibitory / conn_num_out_excitatory, I_weight_per_neuron_out / E_weight_per_neuron_out, I_weight_per_conn_out / E_weight_per_conn_out] )
        else:
          cell_text_out.append( ['/', '/', '/', '/', '/', '/', 0.0, 0.0] )

      # fig = plt.figure( figsize=(20, 10) )
      # ax = plt.subplot( 111, frame_on=False )
      # ax.xaxis.set_visible( False )
      # ax.yaxis.set_visible( False )
      # the_table_p = plt.table( cellText=cell_text_out, colWidths=[0.12] * 90, rowLabels=rows_out, colLabels=columns, loc='center' )
      # the_table_p.auto_set_font_size( False )
      # the_table_p.set_fontsize(10)
      # the_table_p.scale( 1.1, 1.1 )
      # ax.title.set_text( pre_l + ' cell out degree connections' )
      #plt.savefig( 'log/Conn_mat_detail/' + pre_l + '_cell_out_degree_connections.jpg' )
      # table_in = plt.figure()



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


# 5) out degree connection figure
# all_neuron_GIDS = []
# for pre_l in ctx_M1_layers.keys():
#   for post_l in ctx_M1_layers.keys():
#     all_neuron_GIDS.extend(list(nest.GetNodes(ctx_M1_layers[post_l])[0]))
#   center_n = ntop.FindCenterElement(ctx_M1_layers[pre_l])
#   connectome_n = nest.GetConnections(center_n, all_neuron_GIDS)
#   conn_list = np.zeros((len(connectome_n), 9))     # make 9 column zero file
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
#   np.savetxt('log/M1_projections/conn_out_exp_%s.csv' % (pre_l),conn_list, delimiter=",", fmt='%.5f')

proj_numbers = {}
for pre_l in ctx_M1_layers.keys():
  all_neuron_GIDS = {}
  connectome_n = {}
  proj_numbers [pre_l] = {}
  for post_l in ctx_M1_layers.keys():
      all_neuron_GIDS[post_l] =list(nest.GetNodes(ctx_M1_layers[post_l])[0])
      center_n = ntop.FindCenterElement(ctx_M1_layers[pre_l])
      connectome_n[post_l] = nest.GetConnections(center_n, all_neuron_GIDS[post_l])
      proj_numbers[pre_l][post_l] = len(connectome_n[post_l])

  all_num = 0
  for post_l in ctx_M1_layers.keys():
    all_num = all_num + proj_numbers[pre_l][post_l]

  conn_list = np.zeros((all_num, 6)) # make 3 column zero file
  cnt = 0
  for post_l in ctx_M1_layers.keys():
    for ii in range(len(connectome_n[post_l])):
      pre_neuron_GID = connectome_n[post_l] [ii] [0]
      post_neuron_GID = connectome_n[post_l][ii][1]
      conn_list[ii+ cnt, 0:3] = np.asarray(ntop.GetPosition([post_neuron_GID])[0])
      conn_list[ii+ cnt, 3:6] = np.asarray( ntop.GetPosition([pre_neuron_GID])[0])

    cnt = cnt + len( connectome_n [post_l] )

  np.savetxt('log/M1_projections/conn_out_exp_%s.csv' % (pre_l),conn_list, delimiter=",", fmt='%.5f')


# 3D plot of the results
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d import proj3d
import matplotlib
matplotlib.use("TKAGG")
import matplotlib.pyplot as plt

plt.rc('font', size=40)   #size=22
plt.rcParams["figure.figsize"] = [80,60]
plt.rc('font', family = 'serif')

for pre_l in ctx_M1_layers.keys():
  conn_load = np.loadtxt('log/M1_projections/conn_out_exp_%s.csv' % (pre_l), delimiter=',' )
  fig = plt.figure()
  ax = fig.gca( projection='3d' )
  # making transparent grids and frames
  #ax.xaxis.set_pane_color( (1.0, 1.0, 1.0, 0.0) )
  #ax.yaxis.set_pane_color( (1.0, 1.0, 1.0, 0.0) )
  #ax.zaxis.set_pane_color( (1.0, 1.0, 1.0, 0.0) )
  #ax.xaxis._axinfo ["grid"] ['color'] = (1, 1, 1, 0)
  #ax.yaxis._axinfo ["grid"] ['color'] = (1, 1, 1, 0)
  #ax.zaxis._axinfo ["grid"] ['color'] = (1, 1, 1, 0)
  ax.set_frame_on( False )

  ax.grid( True )
  ax.xaxis.pane.set_edgecolor( 'black' )
  ax.yaxis.pane.set_edgecolor( 'black' )
  ax.xaxis.pane.fill = False
  ax.yaxis.pane.fill = False
  ax.zaxis.pane.fill = True

  ax.xaxis._axinfo ['tick'] ['inward_factor'] = 0
  ax.xaxis._axinfo ['tick'] ['outward_factor'] = 0.2
  ax.yaxis._axinfo ['tick'] ['inward_factor'] = 0
  ax.yaxis._axinfo ['tick'] ['outward_factor'] = 0.2
  ax.zaxis._axinfo ['tick'] ['inward_factor'] = 0
  ax.zaxis._axinfo ['tick'] ['outward_factor'] = 0.2

  # remove tick labels
  # ax.w_xaxis.set_ticklabels( [] )
  # ax.w_yaxis.set_ticklabels( [] )
  # ax.w_zaxis.set_ticklabels( [] )

  #ax.set_axis_off()
  ax.set_zlabel( '\n\n\nThickness (mm) \nL6   L5B   L5A  L23 L1', fontsize=60, linespacing=2.6)   #fontsize=35
  ax.set_ylabel( '\n\n3.5 mm', fontsize=60, linespacing=2)    #fontsize=35
  #ax.set_xlabel( '\n4 mm', fontsize=56, linespacing=2.1 )   #fontsize=28
  ax.xaxis._axinfo ['label'] ['space_factor'] = 7.0
  ax.set_zlim( 1.4 , 0.0)
  ax.set_xlim( -1.75, 1.75 )
  ax.set_ylim( -1.75, 1.75 )
  ax.set_xticks( [-1.75,0,1.75] )
  plt.rcParams ['xtick.major.pad'] = '28'
  plt.rcParams ['ytick.major.pad'] = '18'
  #plt.rcParams ['ztick.major.pad'] = '18'
  ax.view_init( 5,175)
  jj = 0
  color_list = ['g', 'b', 'm', 'y', 'r', 'c', 'g', 'b', 'm', 'y', 'r', 'c', 'g', 'b', 'm', 'y', 'r', 'c']
  k=0
  M1_z_scale = {'L1_': 0.07, 'L23': 0.292, 'L5A': 0.544, 'L5B':0.879, 'L6_': 1.257 }

  for post_l in ctx_M1_layers.keys():
      if proj_numbers[pre_l][post_l] != 0:
          x= [None] * proj_numbers[pre_l][post_l]
          y = [None] * proj_numbers [pre_l] [post_l]
          z = [None] * proj_numbers [pre_l] [post_l]
          for ii in range(proj_numbers[pre_l][post_l]):
            x[ii] = conn_load[ii+jj,0]
            y[ii] = conn_load[ii+jj,1]
            z[ii] = conn_load[ii+jj,2] + M1_z_scale[post_l[3:6]]


          ax.scatter(x, y, z, color =color_list [k], s = 15, label=post_l )
          ax.legend(loc=2)
          k =k+1
          jj = jj + proj_numbers[pre_l][post_l]

  ax.title.set_text( pre_l + '_projections' )
  plt.savefig( 'log/M1_projections/' + pre_l + '_projections (scatter).png')

'''
# 3D plot of the results
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d import proj3d
import matplotlib
matplotlib.use("TKAGG")
import matplotlib.pyplot as plt

plt.rc('font', size=40)   #size=22
plt.rcParams["figure.figsize"] = [80,60]
plt.rc('font', family = 'serif')
# making 3D with lines

for pre_l in ctx_M1_layers.keys():
  conn_load = np.loadtxt('log/M1_projections/conn_out_exp_%s.csv' % (pre_l), delimiter=',' )
  fig = plt.figure()
  ax = fig.gca( projection='3d' )
  # making transparent grids and frames
  ax.set_frame_on( False )

  ax.grid( True )
  ax.xaxis.pane.set_edgecolor( 'black' )
  ax.yaxis.pane.set_edgecolor( 'black' )
  ax.xaxis.pane.fill = False
  ax.yaxis.pane.fill = False
  ax.zaxis.pane.fill = True

  ax.xaxis._axinfo ['tick'] ['inward_factor'] = 0
  ax.xaxis._axinfo ['tick'] ['outward_factor'] = 0.2
  ax.yaxis._axinfo ['tick'] ['inward_factor'] = 0
  ax.yaxis._axinfo ['tick'] ['outward_factor'] = 0.2
  ax.zaxis._axinfo ['tick'] ['inward_factor'] = 0
  ax.zaxis._axinfo ['tick'] ['outward_factor'] = 0.2


  #remove tick labels
  # ax.w_xaxis.set_ticklabels( [] )
  # ax.w_yaxis.set_ticklabels( [] )
  # ax.w_zaxis.set_ticklabels( [] )

  #ax.set_axis_off()
  ax.set_zlabel( '\n\n\nThickness (mm) \nL6   L5B   L5A  L23 L1', fontsize=60, linespacing=2.6)   #fontsize=35
  ax.set_ylabel( '\n\n3.5 mm', fontsize=60, linespacing=2)    #fontsize=35
  #ax.set_xlabel( '\n4 mm', fontsize=56, linespacing=2.1 )   #fontsize=28
  ax.xaxis._axinfo ['label'] ['space_factor'] = 7.0
  ax.set_zlim( 1.4 , 0.0)
  ax.set_xlim( -1.75, 1.75 )
  ax.set_ylim( -1.75, 1.75 )
  ax.set_xticks( [-1.75,0,1.75] )
  plt.rcParams ['xtick.major.pad'] = '28'
  plt.rcParams ['ytick.major.pad'] = '18'
  ax.view_init( 5,175)
  jj = 0
  color_list = ['g', 'b', 'm', 'y', 'r', 'c', 'g', 'b', 'm', 'y', 'r', 'c', 'g', 'b', 'm', 'y', 'r', 'c']
  k=0
  M1_z_scale = {'L1_': 0.07, 'L23': 0.292, 'L5A': 0.544, 'L5B':0.879, 'L6_': 1.257 }
  
  for post_l in ctx_M1_layers.keys():
      if proj_numbers[pre_l][post_l] != 0:
          x= [None] * proj_numbers[pre_l][post_l]
          y = [None] * proj_numbers [pre_l] [post_l]
          z = [None] * proj_numbers [pre_l] [post_l]
          for ii in range(proj_numbers[pre_l][post_l]):
            x[ii] = conn_load[ii+jj,0]
            y[ii] = conn_load[ii+jj,1]
            z[ii] = conn_load[ii+jj,2] + M1_z_scale[post_l[3:6]]
            ax.plot3D((conn_load[0,3], x[ii]),  (conn_load[0,4], y[ii]), (conn_load[0,5]+ M1_z_scale[pre_l[3:6]], z[ii]), color =color_list [k], label=post_l, linewidth = 0.3)


          ax.legend(loc=2)
          k =k+1
          jj = jj + proj_numbers[pre_l][post_l]

  ax.title.set_text( pre_l + '_projections' )
  plt.savefig( 'log/M1_projections/' + pre_l + '_projections.png' )
'''

'''
# if I want to make a movie of graph
# for ii in xrange(0,360,1):
#         ax.view_init(elev=10., azim=ii)
#         savefig("movie%d.png" % ii)
'''





