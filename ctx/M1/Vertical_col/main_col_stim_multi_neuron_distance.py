#!/usr/bin/env python
# -*- coding: utf-8 -*-

# This is the program to prepare necessary file to check LSPS on M1 layers
# compare to file vLSPS_main_pre_M1 this try to use more functions to speed up run time

# For this code I use one multimeter for each neuron
# and number of trials for rest state and stimulated state are equal
# some functions were addred to reduce the calculations of outputs
# any number of inside and outside neurons could be recorded

import nest
import numpy as np
import nest.topology as ntop
print (nest.version())
nest.ResetKernel()
import configparser
import shelve
import pickle
import math
import time
import collections
import nest.raster_plot
import nest.voltage_trace
import matplotlib.pyplot as plt
import pylab

nest.ResetKernel()

#read configure file
#config = configparser.ConfigParser()
#config.read('simu.conf')       ## ????

###############################################################
###############################################################

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
        file_params = run_path('baseCTXM1Params.py', init_globals=globals())
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
  nest.SetKernelStatus({"data_path": 'log'})
  if sim_params['dt'] != '0.1':
    nest.SetKernelStatus({'resolution': float(sim_params['dt'])})


###############################################################
def gen_neuron_postions_ctx(layer_dep, layer_thickness, nbneuron, M1_layer_size, scalefactor, pop_name):
  neuron_per_grid = math.pow((nbneuron / layer_thickness), 1.0 / 3)        # ????
  Sub_Region_Architecture = [0, 0, 0]
  Sub_Region_Architecture [0] = int(np.round( neuron_per_grid * M1_layer_size[0] * scalefactor[0]))
  Sub_Region_Architecture [1] = int(np.round( neuron_per_grid * M1_layer_size[1] * scalefactor[1]))
  Sub_Region_Architecture [2] = int(np.round( neuron_per_grid * layer_thickness))
  Neuron_pos_x = np.linspace(-0.5*scalefactor[0], 0.5*scalefactor[0], num=Sub_Region_Architecture[0], endpoint=True)
  Neuron_pos_y = np.linspace(-0.5*scalefactor[1], 0.5*scalefactor[1], num=Sub_Region_Architecture[1], endpoint=True)
  Neuron_pos_z = np.linspace(layer_dep, (layer_dep+layer_thickness), num=Sub_Region_Architecture[2], endpoint=True)
  Neuron_pos = []
  for i in range( Sub_Region_Architecture [0] ):
    for j in range( Sub_Region_Architecture [1] ):
      for k in range( Sub_Region_Architecture [2] ):
        Neuron_pos.append( [Neuron_pos_x [i], Neuron_pos_y [j], Neuron_pos_z [k]] )
  #np.savez( 'ctx' + pop_name, Neuron_pos=Neuron_pos )
  return Neuron_pos

###############################################################
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
                'kernel': {
                  'gaussian2D': {'p_center': float( conn_dict ['p_center'] ), 'sigma_x': sigma_x, 'sigma_y': sigma_y}},
                'weights': float( conn_dict ['weight'] ),
                'delays': float( conn_dict ['delay'] ),
                'allow_autapses': False,
                'allow_multapses': False}
  if conn_dict['p_center'] != 0.0 and sigma_x != 0.0 and conn_dict['weight'] != 0.0:
    ntop.ConnectLayers( pre_SubSubRegion, post_SubSubRegion, conndict )

############################################################
def save_layers_position(layer_name, layer_gid, positions):
    np.array( nest.GetNodes( layer_gid ) [0] )
    gid_and_positions = np.column_stack( (np.array( nest.GetNodes( layer_gid ) [0] ), positions) )
    np.savetxt( 'log/' + layer_name + '.txt', gid_and_positions, fmt='%1.3f' )


###############################################################
def create_layers_ctx_M1(extent, center, positions, elements):
  newlayer = ntop.CreateLayer( {'extent': extent, 'center': center, 'positions': positions, 'elements': elements} )
  return newlayer

###############################################################
def get_input_column_layers_ctx_M1(ctx_M1_layers, layer_name, input_layer_name, radius, target_radius, num_sele_neurons):
    circle_center = [[0.0, 0.0]]
    # choose some neurons in different distances from center
    target_gids = {}
    inside_gids = []
    outside_gids = []
    for n in range(len(target_radius)):
        target_gids [n] = []

    layer_gids = nest.GetNodes(ctx_M1_layers[layer_name])[0]
    neuron_positions = ntop.GetPosition(layer_gids)

    if layer_name == input_layer_name:
        for nn in range(len(layer_gids)):
            if np.linalg.norm([(neuron_positions[nn][0] - circle_center[0][0]), (neuron_positions[nn][1] - circle_center[0][1])]) <= radius:
                inside_gids.append(layer_gids[nn])
            else:
                outside_gids.append(layer_gids[nn])

        print ('all', layer_name ,'neuron numbers were:', len(layer_gids), 'and only', len(inside_gids), 'neurons will be stimulated inside the column using poisson generator....')


    i=0
    for rad in target_radius:
        if rad < 0.1:
            j=0
            for nn in range(len(layer_gids)):
                if (np.linalg.norm([(neuron_positions[nn][0]-circle_center[0][0]),(neuron_positions[nn][1]-circle_center[0][1])]) >= (rad-0.03)) and (np.linalg.norm([(neuron_positions[nn][0]-circle_center[0][0]),(neuron_positions[nn][1]-circle_center[0][1])]) <= (rad+0.03)):
                    target_gids[i].append(layer_gids[nn])
                    j=j+1

                    if j == num_sele_neurons:
                        break

            i=i+1
        else:
            j = 0
            for nn in range(len(layer_gids)):
                if (np.linalg.norm( [(neuron_positions [nn] [0] - circle_center [0] [0]), (neuron_positions [nn] [1] - circle_center [0] [1])] ) >= (rad - 0.025)) and (np.linalg.norm( [(neuron_positions [nn] [0] - circle_center [0] [0]), (neuron_positions [nn] [1] - circle_center [0] [1])] ) <= (rad + 0.025)):
                    target_gids [i].append( layer_gids [nn] )
                    j = j + 1

                    if j == num_sele_neurons:
                        break

            i = i + 1


    return inside_gids , outside_gids, target_gids

###############################################################
def copy_neuron_model(elements, neuron_info, new_model_name):
    configuration = {}
    # Membrane potential in mV
    configuration['V_m'] = 0.0
    # Leak reversal Potential (aka resting potential) in mV
    configuration['E_L'] = -70.0
    # Membrane Capacitance in pF
    configuration['C_m'] = 250.0
    # Refractory period in ms
    configuration['t_ref'] = float(neuron_info['absolute_refractory_period'])
    # Threshold Potential in mV
    configuration['V_th'] = float(neuron_info['spike_threshold'])
    # Reset Potential in mV
    configuration['V_reset'] = float(neuron_info['reset_value'])
    # Excitatory reversal Potential in mV
    configuration['E_ex'] = float(neuron_info['E_ex'])
    # Inhibitory reversal Potential in mV
    configuration['E_in'] = float(neuron_info['E_in'])
    # Leak Conductance in nS
    configuration['g_L'] = 250./float(neuron_info['membrane_time_constant'])
    # Time constant of the excitatory synaptic exponential function in ms
    configuration['tau_syn_ex'] = float(neuron_info['tau_syn_ex'])
    # Time constant of the inhibitory synaptic exponential function in ms
    configuration['tau_syn_in'] = float(neuron_info['tau_syn_in'])
    # Constant Current in pA
    configuration['I_e'] = float(neuron_info['I_ex'])
    nest.CopyModel(elements, new_model_name, configuration)
    return new_model_name
###############################################################
def instantiate_ctx_M1(ctx_M1_params, scalefactor):
    region_name = 'M1'
    index = 0

    pos_Exc = np.zeros( (0, 4) )
    pos_inh = np.zeros( (0, 4) )
    M1_layer_Name = ctx_M1_params ['M1'] ['structure_info'] ['Layer_Name']
    M1_layer_size = ctx_M1_params ['M1'] ['structure_info'] ['region_size']
    M1_layer_size = np.array( M1_layer_size )         #may this is used later
    M1_layer_thickness = ctx_M1_params ['M1'] ['structure_info'] ['layer_thickness']
    M1_layer_thickness_ST = np.cumsum( M1_layer_thickness)
    M1_layer_thickness_ST = np.insert( M1_layer_thickness_ST, 0, 0.0 )
    M1_layer_depth = ctx_M1_params ['M1'] ['structure_info'] ['layer_depth']
    topo_extend = [M1_layer_size [0] * int( scalefactor [0] ) + 1., M1_layer_size [1] * int( scalefactor [1] ) + 1., M1_layer_size [2] + 1.]
    #topo_extend = np.array(topo_extend)
    topo_center = np.array( [0.0, 0.0, 0.0])
    SubSubRegion_Excitatory = []
    SubSubRegion_Inhibitory = []
    ctx_M1_layers = {}
    for l in range( len( M1_layer_Name)):
        print( '###########################################' )
        print( 'start to create layer: ' + M1_layer_Name [l] )
        topo_center [2] = M1_layer_depth [l] + 0.5 * M1_layer_thickness [l]
        ctx_M1_params [region_name] ['neuro_info'] [M1_layer_Name [l]] = collections.OrderedDict(sorted( ctx_M1_params [region_name] ['neuro_info'] [M1_layer_Name [l]].items(), key=lambda t: t [0] ) )
        n_type_index = 0
        for n_type in ctx_M1_params ['M1'] ['neuro_info'] [M1_layer_Name [l]].keys():
            n_type_index = ctx_M1_params ['M1'] ['neuro_info'] [M1_layer_Name [l]] [n_type] ['n_type_index']
            print( 'n_type_index:', n_type_index )
            print( n_type )
            n_type_info = ctx_M1_params ['M1'] ['neuro_info'] [M1_layer_Name [l]] [n_type]
            neuronmodel = copy_neuron_model(ctx_M1_params [region_name] ['neuro_info'] [M1_layer_Name [l]] [n_type] ['neuron_model'], n_type_info, region_name + '_' + M1_layer_Name [l] + '_' + n_type )
            elements = neuronmodel
            neuron_info = ctx_M1_params ['M1'] ['neuro_info'] [M1_layer_Name [l]] [n_type]
            Neuron_pos = gen_neuron_postions_ctx( M1_layer_depth [l], M1_layer_thickness [l], n_type_info ['Cellcount_mm2'], M1_layer_size, scalefactor, 'Neuron_pos_' + region_name + '_' + M1_layer_Name [l] + '_' + n_type )
            if n_type_info ['EorI'] == "E":
                nest.SetDefaults( elements, {"I_e": float( neuron_info ['I_ex'] ), "V_th": float( neuron_info ['spike_threshold'] ), "V_reset": float( neuron_info ['reset_value'] ), "t_ref": float( neuron_info ['absolute_refractory_period'] )} )
                SubSubRegion_Excitatory.append(create_layers_ctx_M1( topo_extend, topo_center, Neuron_pos, neuronmodel ) )
                ctx_M1_layers [region_name + '_' + M1_layer_Name [l] + '_' + n_type] = SubSubRegion_Excitatory[-1]
                Neuron_GIDs = None
                Neuron_GIDs = nest.GetNodes(SubSubRegion_Excitatory[-1])[0]
                print( 'Neuron_num:' + str( len( Neuron_GIDs ) ) )
                save_layers_position( region_name + '_' + M1_layer_Name [l] + '_' + n_type, SubSubRegion_Excitatory [-1], Neuron_pos)
            elif n_type_info ['EorI'] == "I":
                nest.SetDefaults( elements, {"I_e": float( neuron_info ['I_ex'] ), "V_th": float( neuron_info ['spike_threshold'] ), "V_reset": float( neuron_info ['reset_value'] ), "t_ref": float( neuron_info ['absolute_refractory_period'] )} )
                SubSubRegion_Inhibitory.append(create_layers_ctx_M1( topo_extend, topo_center, Neuron_pos, neuronmodel ) )
                ctx_M1_layers [region_name + '_' + M1_layer_Name [l] + '_' + n_type] = SubSubRegion_Inhibitory [-1]
                Neuron_GIDs=None
                Neuron_GIDs = nest.GetNodes( SubSubRegion_Inhibitory [-1] ) [0]
                print( 'Neuron_num:' + str( len( Neuron_GIDs ) ) )
                save_layers_position( region_name + '_' + M1_layer_Name [l] + '_' + n_type, SubSubRegion_Inhibitory [-1], Neuron_pos)
            else:
                print( 'Unknow E or I' )

    M1_internal_connection = np.load( ctx_M1_params ['M1'] ['connection_info'] ['M1toM1'] )
    for pre_layer_name in ctx_M1_layers.keys():
        for post_layer_name in ctx_M1_layers.keys():
            print( 'start to connect ' + pre_layer_name + ' with ' + post_layer_name )
            connect_layers_ctx_M1( ctx_M1_layers [pre_layer_name], ctx_M1_layers [post_layer_name], M1_internal_connection [pre_layer_name] [post_layer_name] )

    return ctx_M1_layers, SubSubRegion_Excitatory, SubSubRegion_Inhibitory

###############################################################
def generate_resting_state(layer_gid, layer_name, ignore_time, params={"withgid": True, "withtime": True, "to_file": False}):
  params.update({'label': layer_name, "start": float(ignore_time)})
  ini_time = ignore_time - 150
  ini_time_ini = ignore_time - 300
  # first adding poisson generator one for all neuron types different for the inhibitory and excitatory neurons
  name_list = ['M1_L23_CC', 'M1_L5A_CC', 'M1_L5A_CT', 'M1_L5B_CC', 'M1_L5B_CS', 'M1_L6_CT']
  if layer_name in name_list:
      PSG = nest.Create( 'poisson_generator', 1, params={'rate': 30.0, "start": float( ini_time )} )  # , 'label': layer_name})
      nest.Connect( pre=PSG, post=nest.GetNodes( layer_gid ) [0], syn_spec={'weight': 50.0, 'delay': 1.5} )
  elif layer_name == 'M1_L5B_PT':
      PSG = nest.Create( 'poisson_generator', 1, params={'rate': 90.0, "start": float(ini_time )} )  # , 'label': layer_name})           # specific PSG for L5B_PT neurons
      nest.Connect( pre=PSG, post=nest.GetNodes( layer_gid ) [0], syn_spec={'weight': 50.0, 'delay': 1.5} )
  elif layer_name == 'M1_L5A_CS':
      PSG = nest.Create( 'poisson_generator', 1, params={'rate': 70.0, "start": float(ini_time )} )  # , 'label': layer_name})           # specific PSG for L5B_PT neurons
      nest.Connect( pre=PSG, post=nest.GetNodes( layer_gid ) [0], syn_spec={'weight': 50.0, 'delay': 1.5} )
  else:
      PSG = nest.Create( 'poisson_generator', 1, params={'rate': 45.0, "start": float( ini_time_ini )} )  # , 'label': layer_name})
      nest.Connect( pre=PSG, post=nest.GetNodes( layer_gid ) [0], syn_spec={'weight': 50.0, 'delay': 1.5} )
  '''
  if layer_name in ['M1_L23_CC']:
    PSG = nest.Create('poisson_generator', 1, params={'rate': 800.0, "start": float(ini_time)})
    nest.Connect(pre=PSG, post=nest.GetNodes(layer_gid)[0], syn_spec={'weight': 4.0, 'delay': 1.5})
  elif layer_name in ['M1_L5A_CC']:
    PSG = nest.Create('poisson_generator', 1, params={'rate': 800.0, "start": float(ini_time)})
    nest.Connect(pre=PSG, post=nest.GetNodes(layer_gid)[0], syn_spec={'weight': 4.0, 'delay': 1.5})
  elif layer_name in ['M1_L5A_CS']:
    PSG = nest.Create('poisson_generator', 1, params={'rate': 950.0, "start": float(ini_time)})
    nest.Connect(pre=PSG, post=nest.GetNodes(layer_gid)[0], syn_spec={'weight': 4.0, 'delay': 1.5})
  elif layer_name in ['M1_L5A_CT']:
    PSG = nest.Create('poisson_generator', 1, params={'rate': 800.0, "start": float(ini_time)})
    nest.Connect(pre=PSG, post=nest.GetNodes(layer_gid)[0], syn_spec={'weight': 4.0, 'delay': 1.5})
  elif layer_name in ['M1_L5B_CC']:
    PSG = nest.Create('poisson_generator', 1, params={'rate': 950.0, "start": float(ini_time)})
    nest.Connect(pre=PSG, post=nest.GetNodes(layer_gid)[0], syn_spec={'weight': 4.0, 'delay': 1.5})
  elif layer_name in ['M1_L5B_CS']:
    PSG = nest.Create('poisson_generator', 1, params={'rate': 950.0, "start": float(ini_time)})  # , 'label': layer_name})           # specific PSG for L5B_PT neurons
    nest.Connect(pre=PSG, post=nest.GetNodes(layer_gid)[0], syn_spec={'weight': 4.0, 'delay': 1.5})
  elif layer_name in ['M1_L5B_PT']:
    PSG = nest.Create('poisson_generator', 1, params={'rate': 1100.0, "start": float(ini_time)})
    nest.Connect(pre=PSG, post=nest.GetNodes(layer_gid)[0], syn_spec={'weight': 4.0, 'delay': 1.5})
  elif layer_name in ['M1_L6_CT']:
    PSG = nest.Create('poisson_generator', 1, params={'rate': 1150.0, "start": float(ini_time)})
    nest.Connect(pre=PSG, post=nest.GetNodes(layer_gid)[0], syn_spec={'weight': 4.0, 'delay': 1.5})

  elif layer_name in ['M1_L1_SBC']:
    PSG = nest.Create('poisson_generator', 1, params={'rate': 1200.0, "start": float(ini_time_ini)})
    nest.Connect(pre=PSG, post=nest.GetNodes(layer_gid)[0], syn_spec={'weight': 4.5, 'delay': 1.5})
  elif layer_name in ['M1_L1_ENGC']:
    PSG = nest.Create('poisson_generator', 1, params={'rate': 1200.0, "start": float(ini_time_ini)})
    nest.Connect(pre=PSG, post=nest.GetNodes(layer_gid)[0], syn_spec={'weight': 4.5, 'delay': 1.5})
  elif layer_name in ['M1_L23_PV']:
    PSG = nest.Create('poisson_generator', 1, params={'rate': 1400.0, "start": float(ini_time_ini)})
    nest.Connect(pre=PSG, post=nest.GetNodes(layer_gid)[0], syn_spec={'weight': 4.5, 'delay': 1.5})
  elif layer_name in ['M1_L23_SST']:
    PSG = nest.Create('poisson_generator', 1, params={'rate': 650.0, "start": float(ini_time_ini)})
    nest.Connect(pre=PSG, post=nest.GetNodes(layer_gid)[0], syn_spec={'weight': 4.5, 'delay': 1.5})
  elif layer_name in ['M1_L23_VIP']:
    PSG = nest.Create('poisson_generator', 1, params={'rate': 1400.0, "start": float(ini_time_ini)})
    nest.Connect(pre=PSG, post=nest.GetNodes(layer_gid)[0], syn_spec={'weight': 4.5, 'delay': 1.5})
  elif layer_name in ['M1_L5A_PV']:
    PSG = nest.Create('poisson_generator', 1, params={'rate': 1350.0, "start": float(ini_time_ini)})
    nest.Connect(pre=PSG, post=nest.GetNodes(layer_gid)[0], syn_spec={'weight': 4.5, 'delay': 1.5})
  elif layer_name in ['M1_L5A_SST']:
    PSG = nest.Create('poisson_generator', 1, params={'rate': 700.0, "start": float(ini_time_ini)})
    nest.Connect(pre=PSG, post=nest.GetNodes(layer_gid)[0], syn_spec={'weight': 4.5, 'delay': 1.5})
  elif layer_name in ['M1_L5B_PV']:
    PSG = nest.Create('poisson_generator', 1, params={'rate': 1400.0, "start": float(ini_time_ini)})
    nest.Connect(pre=PSG, post=nest.GetNodes(layer_gid)[0], syn_spec={'weight': 4.0, 'delay': 1.5})
  elif layer_name in ['M1_L5B_SST']:
    PSG = nest.Create('poisson_generator', 1, params={'rate': 710.0, "start": float(ini_time_ini)})
    nest.Connect(pre=PSG, post=nest.GetNodes(layer_gid)[0], syn_spec={'weight': 4.0, 'delay': 1.5})
  elif layer_name in ['M1_L6_PV']:
    PSG = nest.Create('poisson_generator', 1, params={'rate': 1550.0, "start": float(ini_time_ini)})
    nest.Connect(pre=PSG, post=nest.GetNodes(layer_gid)[0], syn_spec={'weight': 4.0, 'delay': 1.5})
  elif layer_name in ['M1_L6_SST']:
    PSG = nest.Create('poisson_generator', 1, params={'rate': 710.0, "start": float(ini_time_ini)})
    nest.Connect(pre=PSG, post=nest.GetNodes(layer_gid)[0], syn_spec={'weight': 4.0, 'delay': 1.5})
  '''

#################################################################
def layer_poisson_generator(layer_gid, layer_name, input_layer_name, params={"withgid": True, "withtime": True, "to_file": False}):
    #def layer_spike_detector(layer_gid, layer_name, params={"withgid": True, "withtime": True, "to_file": True, 'fbuffer_size': 8192}):
    params.update({'label': layer_name})

    # stimulation by using PSG
    if layer_name == input_layer_name:
        PSG_arrow = nest.Create('poisson_generator', 1, params={'rate':  10000.0}) #, 'label': layer_name})
        nest.Connect(pre=PSG_arrow, post=layer_gid, syn_spec={'weight': 100.0, 'delay': 1.5})
        return PSG_arrow

###############################################################
def layer_spike_detector(target_gids, layer_name, record_onset, distance_num, params = {"withgid": True, "withtime": True, "to_file": False}):
    print( 'spike detector for ' + layer_name )
    params.update( {'label': layer_name, "start": float(record_onset)} )

    # Add detector for all neuron types
    #detector_in = nest.Create( "spike_detector", params=params )
    #nest.Connect( pre=layer_gids_inside, post=detector_in )

    #detector_out = nest.Create( "spike_detector", params=params )
    #nest.Connect( pre=layer_gids_outside, post=detector_out)

    detector_targets={}
    for n in range(distance_num):
        detector_targets[n] = nest.Create( "spike_detector", params=params )
        nest.Connect( pre=target_gids[n], post=detector_targets[n] )
        ran_neurons_num = len(target_gids[n])
    '''
    # generating some random numbers to choose random neuron to record
    inside_random = np.random.choice(len(layer_gids_inside), int(ran_neurons_num), replace=False)
    outside_random = np.random.choice(len(layer_gids_outside), int(ran_neurons_num), replace=False)

    nest_mm_in = {}
    nest_mm_out = {}
    # add multimeter just for to record V_m and conductance (inhibitory and excitatory) of a single cell each cell population
    for num in range( int( ran_neurons_num ) ):
        nest_mm_in[num] = nest.Create( 'multimeter', params={"interval": 0.1, "record_from": ["V_m", "g_ex", "g_in"], "withgid": True, "to_file": False, 'label': layer_name, 'withtime': True, "start": float(record_onset), 'use_gid_in_filename': True} )
        nest.Connect(nest_mm_in[num], [layer_gids_inside[inside_random[num]]])

    for num in range( int( ran_neurons_num ) ):
        nest_mm_out[num] = nest.Create( 'multimeter', params={"interval": 0.1, "record_from": ["V_m", "g_ex", "g_in"], "withgid": True, "to_file": False, 'label': layer_name, 'withtime': True, "start": float(record_onset), 'use_gid_in_filename': True} )  #
        nest.Connect(nest_mm_out[num], [layer_gids_outside[outside_random[num]]])

    voltmeter_single = nest.Create( "voltmeter", params={'to_file': False, 'label': layer_name, "start": float( ignore_time ), "withgid": True, "withtime": True} )  # 'use_gid_in_filename': True,   "start": float(ignore_time),
    nest.Connect(voltmeter_single, [layer_gids_inside[2]] )

    #return detector_in, detector_out, detector_targets, nest_mm_in, nest_mm_out, voltmeter_single
    '''
    return detector_targets

################################################################
def average_recorders_outputs(events, neurons_torecord):
    for num in range(neurons_torecord):
        if num == 0:
            temp_ins = events[num]['g_in']
            temp_exs = events[num]['g_ex']
            temp_in = events[num]['V_m']
        else:
            temp_ins = np.vstack( (temp_ins, events[num] ['g_in']) )
            temp_exs = np.vstack( (temp_exs, events[num] ['g_ex']) )
            temp_in = np.vstack( (temp_in, events[num] ['V_m']) )

    temp_ins = np.average(temp_ins, axis=0)
    temp_exs = np.average(temp_exs, axis=0)
    temp_in = np.average(temp_in, axis=0)

    return temp_ins, temp_exs, temp_in

################################################################
def average_over_trials(num_trials, var):
    var_ave_l = [None]*15000
    for n in range(15000):
        t1=0
        for num in range (num_trials):
            t1 = var[n+ (num_trials*15000) + (num*15000)] + t1

        var_ave_l[n] = t1/num_trials

    return np.asarray(var_ave_l)

###############################################################
def spike_sort_count_freq(num_trials, spike_t, initial_ignore, simulation_time, layer_gids, bin_list):
    spike_t = np.sort( spike_t )
    spike_t_rest = np.array([])
    if spike_t.size == 0:
        spike_t_rest = np.array([])

    kk = 0
    for t in spike_t:  # this is separating the rest and stimulated spikes
        if t <= (num_trials * simulation_time) + initial_ignore:
            kk = kk + 1
        if t > (num_trials * simulation_time) + initial_ignore:
            spike_t_rest = spike_t [0:kk]
            spike_t = spike_t [kk:]
            break

    np.set_printoptions( suppress=True )
    # here it bringing the range of two group of rest and stimulated to [0, (similation_time)* num_trials]
    if spike_t_rest.size != 0:
        spike_t_rest = np.around( spike_t_rest - initial_ignore, decimals=2 )
    if spike_t.size != 0:
        spike_t = np.around( spike_t - (initial_ignore + (num_trials * simulation_time)), decimals=2 )

    spike_t_rest = np.sort( spike_t_rest )
    spike_t = np.sort( spike_t )

    # find out the spikes of each trial in rest and stimulated time window
    spike_t_trials = {}
    if spike_t.size == 0:
        for n in range( num_trials ):
            spike_t_trials [n] = np.array( [] )
    else:
        m = 0
        key = 0
        for n in range( num_trials ):
            if n < num_trials - 1:
                ii = 0
                for t in spike_t:
                    if t > (simulation_time * (n + 1)):
                        spike_t_trials [n] = spike_t [m:ii]
                        m = ii
                        break

                    ii = ii + 1
                    if ii == len( spike_t ):
                        if key == 0:
                            spike_t_trials [n] = spike_t [m:]
                            key = key + 1
                        else:
                            spike_t_trials [n] = []

            elif n == num_trials - 1:
                if ii == len( spike_t ):
                    spike_t_trials [n] = []
                else:
                    spike_t_trials [n] = spike_t [m:]

    spike_t_rest_trials = {}
    if spike_t_rest.size == 0:
        for n in range( num_trials ):
            spike_t_rest_trials [n] = np.array( [] )
    else:
        m = 0
        key = 0
        for n in range( num_trials ):
            if n < num_trials - 1:
                ii = 0
                for t in spike_t_rest:
                    if t > (simulation_time * (n + 1)):
                        spike_t_rest_trials [n] = spike_t_rest [m:ii]
                        m = ii
                        break

                    ii = ii + 1
                    if ii == len( spike_t_rest ):
                        if key == 0:
                            spike_t_rest_trials [n] = spike_t_rest [m:]
                            key = key + 1
                        else:
                            spike_t_rest_trials [n] = []

            elif n == num_trials - 1:
                if ii == len( spike_t_rest ):
                    spike_t_rest_trials [n] = []
                else:
                    spike_t_rest_trials [n] = spike_t_rest [m:]

    for n in range( num_trials - 1 ):
        if spike_t_trials [n + 1] == []:
            break
        else:
            spike_t_trials [n + 1] = spike_t_trials [n + 1] - ((n + 1) * simulation_time)

    for n in range( num_trials - 1 ):
        if spike_t_rest_trials [n + 1] == []:
            break
        else:
            spike_t_rest_trials [n + 1] = spike_t_rest_trials [n + 1] - ((n + 1) * simulation_time)

    spike_count_rest_trials = {}
    freq_rest_trials = {}
    spike_count_trials = {}
    freq_trials = {}
    for n in range( num_trials ):
        spike_t_trials [n] = np.sort( spike_t_trials [n] )
        spike_t_rest_trials [n] = np.sort( spike_t_rest_trials [n] )

        if spike_t_trials [n] == []:
            spike_count_trials [n] = [0] * len( bin_list - 1 )

        if spike_t_rest_trials [n] == []:
            spike_count_rest_trials [n] = [0] * len( bin_list - 1 )

        spike_count_trials [n] = np.histogram( (spike_t_trials [n].tolist()), bins=bin_list ) [0]
        spike_count_rest_trials [n] = np.histogram( (spike_t_rest_trials [n].tolist()), bins=bin_list ) [0]

        if len( layer_gids ) == 0:
            freq_trials [n] = spike_count_trials [n] * (1000 / (bin * 5))
            freq_rest_trials [n] = spike_count_rest_trials [n] * (1000 / (bin * 5))
        else:
            freq_trials [n] = spike_count_trials [n] * (1000 / (bin * len( layer_gids )))
            freq_rest_trials [n] = spike_count_rest_trials [n] * (1000 / (bin * len( layer_gids )))


    return freq_trials, freq_rest_trials
############################
def calcualte_ave_sem_sd (num_trials, input_array):
    from scipy.stats import sem
    for num in range(num_trials):
        if num == 0:
            temp_input = input_array[num]
        else:
            temp_input = np.vstack( (temp_input, input_array[num]))

    AVE = np.average(temp_input, axis=0)
    SEM = sem(temp_input, axis=0 )
    SD = np.std(temp_input, axis=0)

    return AVE, SEM, SD

##############################################################################################################################################################################
##############################################################################################################################################################################
# Program initiation
##########################
# 1)read parameters
print('Reading the simulation parameters using "baseSimParameters.py" file')
sim_params = read_sim()
print('Reading the M1 parameters using "baseCTXM1Parameters.py" file')
ctx_M1_params = read_ctx_M1()

# 2) initialize nest
print('Nest Initializations')
initialize_nest(sim_params)

############################
# 3) defining some parameters
scalefactor = sim_params['scalefactor']
simulation_time = sim_params['simDuration']   #should be 1500 ms
initial_ignore = sim_params['initial_ignore']
###########################
# 4) making layers and doing all connections
print('create and connect the M1 layers')
start_time = time.time()
ctx_M1_layers, SubSubRegion_Excitatory, SubSubRegion_Inhibitory =instantiate_ctx_M1(ctx_M1_params, sim_params['scalefactor'])

###########################
# 5) generate the resting state firing
for layer_name in ctx_M1_layers.keys():
    generate_resting_state(ctx_M1_layers[layer_name], layer_name, sim_params['initial_ignore'])


# 6) randomizing the membarne potential
import numpy
for layer_name in ctx_M1_layers.keys():
  Vth = -50.
  Vrest = -70.
  for neuron in nest.GetNodes(ctx_M1_layers[layer_name])[0]:
    nest.SetStatus([neuron], {"V_m": Vrest + (Vth - Vrest) * numpy.random.rand()} )

##############################
# 6) set paramaters for the stimulation of VIP neurons inside the column\
record_onset = 499.9
record_intervals = 500.0
radius = 0.05
###############################
# 7) selecting the neurons for the stimulation located inside the coloumn
poisson_gens = []                 # in case of one trial     poisson_gens = {}
input_layer_name = 'M1_L23_PV'

# Adding distances from center to record effect of distance on activity and spatial extent of neurons
target_radius = [0.03, 0.06, 0.09, 0.12, 0.15, 0.18, 0.21, 0.24, 0.27, 0.30, 0.33, 0.36, 0.39, 0.42, 0.45]   # distances where some neuorns are selected to record
num_sele_neurons = 5

gids_inside = get_input_column_layers_ctx_M1(ctx_M1_layers, input_layer_name, input_layer_name, radius, target_radius, num_sele_neurons)[0]
gids_outside = get_input_column_layers_ctx_M1(ctx_M1_layers, input_layer_name, input_layer_name, radius, target_radius, num_sele_neurons)[1]
#poisson_gens[layer_name]= layer_poisson_generator(gids_inside, layer_name, stim_onset, stim_offset, sim_params['initial_ignore'])   # this was good for one trial
poisson_gens.append(layer_poisson_generator(gids_inside, input_layer_name, input_layer_name))

###############################
# 8) recording from other neurons in circle and out side of circle
#neurons_torecord = {}
#inside_neurons_torecord = {}
#outside_neurons_torecord = {}
#multimeters_in = {}
#multimeters_out = {}
#voltmeters = {}
#detectors_in = {}
#detectors_out = {}
detectors_targets= {}
#layer_gids_inside = {}
#layer_gids_outside = {}
#layer_gids_all = {}
target_gids = {}

for layer_name in ctx_M1_layers.keys():
    #layer_gids_inside[layer_name], layer_gids_outside[layer_name], target_gids[layer_name], distance_num = get_input_column_layers_ctx_M1(ctx_M1_layers, layer_name, input_layer_name, radius, target_radius, num_sele_neurons)
    target_gids [layer_name] = get_input_column_layers_ctx_M1( ctx_M1_layers, layer_name, input_layer_name, radius, target_radius, num_sele_neurons )[2]
    #neurons_torecord [layer_name] = len(target_gids[layer_name])
    #detectors_in[layer_name], detectors_out[layer_name], detectors_targets[layer_name], multimeters_in[layer_name], multimeters_out[layer_name], voltmeters[layer_name] = layer_spike_detector(layer_gids_inside[layer_name],
                                                                                                                                                           #layer_gids_outside[layer_name], target_gids[layer_name],
                                                                                                                                                            #layer_name, record_onset, sim_params['initial_ignore'], distance_num)
    detectors_targets [layer_name] = layer_spike_detector(target_gids [layer_name], layer_name, sim_params ['initial_ignore'], len(target_radius))


    #layer_gids_all [layer_name] = layer_gids_inside [layer_name] + layer_gids_outside [layer_name]

################################
# 9) simulation
distance_num = len(target_radius)
num_trials = 40
print ('Simulation Started:')
start_time=time.time()
for n in range(num_trials):
    if n == 0:
        #nest.SetStatus( poisson_gens [0], {'origin': nest.GetKernelStatus() ['time'], "start": 1000.0, 'stop': 1500.0} )
        #nest.Simulate(simulation_time + initial_ignore)
        nest.SetStatus( poisson_gens [0], {'origin': nest.GetKernelStatus() ['time'], "start": (num_trials*simulation_time) + (2*record_intervals), 'stop': (num_trials*simulation_time)+(3*record_intervals)} )
        nest.Simulate( (num_trials*simulation_time) + initial_ignore + simulation_time )
    elif n == num_trials -1:
        nest.SetStatus( poisson_gens [0], {'origin': nest.GetKernelStatus() ['time'], "start": record_intervals, 'stop': 2*record_intervals} )
        nest.Simulate( simulation_time+ 1.0 )
    else:
        nest.SetStatus( poisson_gens [0], {'origin': nest.GetKernelStatus() ['time'], "start": record_intervals, 'stop': 2*record_intervals} )
        nest.Simulate(simulation_time)

#with open('./log/'+'x_performance.txt', 'a') as file:
  #file.write('Simulation_Elapse_Time: '+str(time.time()-start_time)+'\n')




############################################################################
# 8) calculating the average graph of all trials for all recorded neurons
# first averaging the multimeter data including membrane potential and conductance (inhibitory and excitatory)
events_in = {}
events_out = {}
mps_in = {}
mps_out = {}
g_ins_in = {}
g_ins_out = {}
g_exs_in = {}
g_exs_out = {}
spike_t_in = {}
spike_t_out = {}
spike_t_all ={}
spike_t_targets ={}
for layer_name in ctx_M1_layers.keys():
    '''
    for num in range( inside_neurons_torecord ):
        events_in [num] = nest.GetStatus( multimeters_in [layer_name] [num] ) [0] ["events"]

    for num in range( outside_neurons_torecord ):
        events_out [num] = nest.GetStatus( multimeters_out [layer_name] [num] ) [0] ["events"]

    spike_t_in[layer_name] = nest.GetStatus( detectors_in [layer_name] ) [0] ['events'] ['times']  # recorded from all neurons inside
    spike_t_out[layer_name] = nest.GetStatus( detectors_out [layer_name] ) [0] ['events'] ['times']   # recorded from all neurons outside
    spike_t_all[layer_name] = np.asarray( ((spike_t_in[layer_name]).tolist()) + ((spike_t_out[layer_name]).tolist()))
    '''
    spike_t_targets [layer_name] = {}
    for n in range(distance_num):
        spike_t_targets[layer_name][n] = []
    for n in range(distance_num):
        spike_t_targets[layer_name][n] = nest.GetStatus(detectors_targets [layer_name][n] ) [0] ['events'] ['times']

    '''
    if inside_neurons_torecord > 1:
        g_ins_in[layer_name], g_exs_in[layer_name], mps_in[layer_name]= average_recorders_outputs(events_in, inside_neurons_torecord)   # average among the recorded neurons inside
    else:
        g_ins_in[layer_name] = events_in [0] ['g_in']
        g_exs_in[layer_name] = events_in [0] ['g_ex']
        mps_in[layer_name] = events_in [0] ['V_m']

    if outside_neurons_torecord > 1:
        g_ins_out[layer_name], g_exs_out[layer_name], mps_out[layer_name]= average_recorders_outputs(events_out, outside_neurons_torecord)   # average among the recorded neurons outside
    else:
        g_ins_out[layer_name] = events_out [0] ['g_in']
        g_exs_out[layer_name] = events_out [0] ['g_ex']
        mps_out[layer_name] = events_out [0] ['V_m']
    '''
#-----------------------------------------------------------------------------------------------------------------------
# now average among the trials
#tt = events_in[0] ['times']
#times = ((events_in[0]['times'])[0:15000])
mps_in_ave = {}
mps_out_ave = {}
g_ins_in_ave = {}
g_ins_out_ave = {}
g_exs_in_ave = {}
g_exs_out_ave = {}
'''
for layer_name in ctx_M1_layers.keys():
    mps_in_ave [layer_name] = average_over_trials(num_trials, mps_in[layer_name])
    mps_out_ave [layer_name] = average_over_trials(num_trials, mps_out[layer_name])
    g_ins_in_ave [layer_name] = average_over_trials(num_trials, g_ins_in[layer_name])
    g_ins_out_ave [layer_name] = average_over_trials(num_trials, g_ins_out[layer_name])
    g_exs_in_ave [layer_name] = average_over_trials(num_trials, g_exs_in[layer_name])
    g_exs_out_ave [layer_name] = average_over_trials(num_trials, g_exs_out[layer_name])
'''
#-----------------------------------------------------------------------------------------------------------------------
# calculate the firing rate of cells and finding the average among the cells and among the trials
# first it overlap all the events in one whole time window and then devide to number pf recorded cells
freq_in = {}
freq_out = {}
freq_rest = {}
freq_in_rest = {}
freq_out_rest = {}
freq_targets = {}
freq_targets_rest = {}
bin = 20.0
num_windows = simulation_time/bin
bin_list = [None] * int(num_windows+1)
for nn in range(int(num_windows+1)):
    bin_list[nn] = nn * 20

for layer_name in ctx_M1_layers.keys():
    '''
    print( 'number of inside and outside spikes for population of' + layer_name + 'cells are:', len( spike_t_in [layer_name]),  'and' , len( spike_t_out [layer_name] ) )
    freq_in[layer_name] = spike_sort_count_freq(num_trials, spike_t_in[layer_name], sim_params['initial_ignore'], sim_params['simDuration'], layer_gids_inside[layer_name], bin_list)[0]
    freq_out[layer_name] = spike_sort_count_freq(num_trials, spike_t_out[layer_name], sim_params ['initial_ignore'], sim_params ['simDuration'], layer_gids_outside[layer_name],bin_list)[0]
    freq_in_rest [layer_name] = spike_sort_count_freq(num_trials, spike_t_in[layer_name], sim_params ['initial_ignore'], sim_params ['simDuration'], layer_gids_inside[layer_name], bin_list)[1]
    freq_out_rest [layer_name] = spike_sort_count_freq(num_trials, spike_t_out[layer_name], sim_params ['initial_ignore'], sim_params ['simDuration'], layer_gids_outside[layer_name], bin_list)[1]
    '''
    freq_targets [layer_name] = {}
    freq_targets_rest [layer_name] = {}
    for n in range( distance_num ):
        freq_targets [layer_name] [n] = []
        freq_targets_rest[layer_name] [n] = []
    for n in range(distance_num):
        print( layer_name )
        print( 'number of spikes in target are :', len(spike_t_targets[layer_name][n]), 'n=', n, 'gid numbers:', len(target_gids [layer_name] [n]) )
        freq_targets[layer_name][n] = spike_sort_count_freq(num_trials, spike_t_targets[layer_name][n], sim_params['initial_ignore'], sim_params['simDuration'], target_gids[layer_name][n], bin_list)[0]
        freq_targets_rest [layer_name] [n] = spike_sort_count_freq( num_trials, spike_t_targets [layer_name] [n], sim_params ['initial_ignore'], sim_params ['simDuration'], target_gids [layer_name] [n], bin_list ) [1]


#######################################################################################################################
# calculation of Standard deviation (sd) and standard error of mean (sem)
freq_in_sd ={}
freq_out_sd ={}
freq_in_rest_sd ={}
freq_out_rest_sd ={}
freq_in_sem ={}
freq_out_sem ={}
freq_in_rest_sem ={}
freq_out_rest_sem ={}
freq_in_ave ={}
freq_out_ave ={}
freq_in_rest_ave ={}
freq_out_rest_ave ={}
freq_targets_sd = {}
freq_targets_sem = {}
freq_targets_ave = {}
freq_targets_rest_sd = {}
freq_targets_rest_sem = {}
freq_targets_rest_ave = {}
amp_changes = {}
pnt_amps= {}

for layer_name in ctx_M1_layers.keys():
    '''
    freq_in_ave [layer_name] = calcualte_ave_sem_sd(num_trials, freq_in[layer_name])[0]
    freq_in_sem[layer_name] = calcualte_ave_sem_sd(num_trials, freq_in[layer_name])[1]
    freq_in_sd[layer_name] = calcualte_ave_sem_sd(num_trials, freq_in[layer_name])[2]

    freq_in_rest_ave [layer_name] = calcualte_ave_sem_sd( num_trials, freq_in_rest [layer_name] ) [0]
    freq_in_rest_sem [layer_name] = calcualte_ave_sem_sd( num_trials, freq_in_rest [layer_name] ) [1]
    freq_in_rest_sd [layer_name] = calcualte_ave_sem_sd( num_trials, freq_in_rest [layer_name] ) [2]

    freq_out_ave [layer_name] = calcualte_ave_sem_sd( num_trials, freq_out [layer_name] ) [0]
    freq_out_sem [layer_name] = calcualte_ave_sem_sd( num_trials, freq_out [layer_name] ) [1]
    freq_out_sd [layer_name] = calcualte_ave_sem_sd( num_trials, freq_out [layer_name] ) [2]

    freq_out_rest_ave [layer_name] = calcualte_ave_sem_sd( num_trials, freq_out_rest [layer_name] ) [0]
    freq_out_rest_sem [layer_name] = calcualte_ave_sem_sd( num_trials, freq_out_rest [layer_name] ) [1]
    freq_out_rest_sd [layer_name] = calcualte_ave_sem_sd( num_trials, freq_out_rest [layer_name] ) [2]
    '''
    freq_targets_ave [layer_name] = {}
    freq_targets_rest_ave [layer_name] = {}
    freq_targets_sem [layer_name] = {}
    freq_targets_rest_sem [layer_name] = {}
    freq_targets_sd [layer_name] = {}
    freq_targets_rest_sd [layer_name] = {}
    amp_changes[layer_name] =  {}
    pnt_amps[layer_name] = np.array(())
    for n in range( distance_num ):
        freq_targets_ave [layer_name] [n] = []
        freq_targets_rest_ave [layer_name] [n] = []
        freq_targets_sem [layer_name] [n] = []
        freq_targets_rest_sem [layer_name] [n] = []
        freq_targets_sd [layer_name] [n] = []
        freq_targets_rest_sd [layer_name] [n] = []
    for n in range(distance_num):
        freq_targets_ave [layer_name][n] = calcualte_ave_sem_sd( num_trials, freq_targets [layer_name][n] ) [0]
        freq_targets_sem [layer_name][n] = calcualte_ave_sem_sd( num_trials, freq_targets [layer_name][n] ) [1]
        freq_targets_sd [layer_name][n] = calcualte_ave_sem_sd( num_trials, freq_targets [layer_name][n] ) [2]

        freq_targets_rest_ave [layer_name] [n] = calcualte_ave_sem_sd( num_trials, freq_targets_rest [layer_name] [n] ) [0]
        freq_targets_rest_sem [layer_name] [n] = calcualte_ave_sem_sd( num_trials, freq_targets_rest [layer_name] [n] ) [1]
        freq_targets_rest_sd [layer_name] [n] = calcualte_ave_sem_sd( num_trials, freq_targets_rest [layer_name] [n] ) [2]

        amp_changes[layer_name][n]= np.average(freq_targets_ave[layer_name][n][25:50])- np.average(freq_targets_rest_ave[layer_name][n][25:50])
        pnt_amps [layer_name] = numpy.append( pnt_amps [layer_name], amp_changes [layer_name] [n] )


pnt_amps.pop(input_layer_name, None)
pnt_amps.pop('M1_L5A_CC', None)
pnt_amps.pop('M1_L5A_CT', None)
pnt_amps.pop('M1_L5B_CC', None)
pnt_amps.pop('M1_L5B_CS', None)
pnt_amps_norm = {}
excluded_list = [input_layer_name, 'M1_L5A_CC', 'M1_L5A_CT', 'M1_L5B_CC', 'M1_L5B_CS']
for layer_name in ctx_M1_layers.keys():
    if layer_name not in excluded_list:
        if pnt_amps[layer_name][0] < 0:
            pnt_amps_norm[layer_name] = [float( i ) / (-pnt_amps [layer_name][0]) for i in pnt_amps [layer_name]]
        elif pnt_amps[layer_name][0] > 0:
            pnt_amps_norm [layer_name] = [float( i ) / pnt_amps [layer_name][0] for i in pnt_amps [layer_name]]




print ('Simulation successfully finished!')
print ('Congradulation!!!!!!')
###################################################################

###############
# setting for figs size and font
font = {'family' : 'normal', 'weight' : 'normal', 'size' : 42}
plt.rcParams["figure.figsize"] = [64,48]
plt.rc('font', **font)
plt.rc('xtick', labelsize=60)
plt.rc('ytick', labelsize=60)
###############
# plotting the garphs
######################################
'''
# Membrane potential of single neurons
i=1
for layer_name in ctx_M1_layers.keys():
  figure, ax = plt.subplots()
  ax.plot(times, mps_in_ave[layer_name], times, mps_out_ave [layer_name],linewidth=4.0)
  ax.spines['right'].set_visible(False)
  ax.spines['top'].set_visible(False)
  for line in ['left', 'bottom']:
    ax.spines[line].set_linewidth(5.0)
  
  ax.tick_params(axis='both',length = 10.0,width = 5.0)
  plt.xlabel("time (ms)")
  plt.ylabel("Membrane potential (mV)")
  plt.title(layer_name)
  plt.legend(("mp_in", "mp_out"))
  save_results_to = '/home/morteza/Desktop/reports/190513 (Ver col with std)/1-multi_rest_func/mp/'
  plt.savefig(save_results_to + str(i) +'.jpeg')
  i=i+1
#######################################
# Inhibitory and excitatory conductance
i=1
for layer_name in ctx_M1_layers.keys():
  figure, ax = plt.subplots()
  ax.plot(times, g_exs_in_ave[layer_name], times, g_ins_in_ave[layer_name], times, g_exs_out_ave[layer_name], times, g_ins_out_ave[layer_name], linewidth=3.0)
  ax.spines['right'].set_visible(False)
  ax.spines['top'].set_visible(False)
  for line in ['left', 'bottom']:
    ax.spines[line].set_linewidth(5.0)
  
  ax.tick_params(axis='both',length = 10.0,width = 5.0)
  plt.xlabel("time (ms)")
  plt.ylabel("Synaptic conductance (nS)")
  plt.title(layer_name)
  plt.legend(("g_exc_in", "g_inh_in", "g_exc_out", "g_inh_out"))
  save_results_to = '/home/morteza/Desktop/reports/190513 (Ver col with std)/1-multi_rest_func/con/'
  plt.savefig(save_results_to + str(i) +'.jpeg')
  i=i+1

######################################
# firing frequency 
del bin_list[-1]
i=1
for layer_name in ctx_M1_layers.keys():
  figure, ax = plt.subplots()
  ax.plot(bin_list, freq_in_ave[layer_name], color='g', linewidth=7.0)
  plt.fill_between(bin_list, freq_in_ave[layer_name] - freq_in_sem[layer_name],  freq_in_ave[layer_name] + freq_in_sem[layer_name], facecolor='lightgreen', alpha= 0.4)
  ax.plot(bin_list, freq_out_ave[layer_name],color='r', linewidth=7.0)
  plt.fill_between(bin_list, freq_out_ave[layer_name] - freq_out_sem[layer_name],  freq_out_ave[layer_name] + freq_out_sem[layer_name], facecolor='lightsalmon', alpha= 0.4)
  ax.plot(bin_list, freq_in_rest_ave[layer_name],color='b', linewidth=5.0)
  plt.fill_between(bin_list, freq_in_rest_ave[layer_name] - freq_in_rest_sem[layer_name],  freq_in_rest_ave[layer_name] + freq_in_rest_sem[layer_name], facecolor='skyblue', alpha= 0.4)
  ax.plot(bin_list, freq_out_rest_ave[layer_name],color='m', linewidth=5.0)
  plt.fill_between(bin_list, freq_out_rest_ave[layer_name] - freq_out_rest_sem[layer_name],  freq_out_rest_ave[layer_name] + freq_out_rest_sem[layer_name], facecolor='plum', alpha= 0.4)
  ax.spines['right'].set_visible(False)
  ax.spines['top'].set_visible(False)
  for line in ['left', 'bottom']:
    ax.spines[line].set_linewidth(5.0)
  
  ax.tick_params(axis='both',length = 10.0,width = 5.0)
  plt.xlabel("time (ms)")
  plt.ylabel("Average frequency (Hz)")
  plt.title(layer_name)
  plt.legend(("Inside_col", "Outside_col", "Resting activity inside", "Resting activity outside"))  
  save_results_to = '/home/morteza/Desktop/reports/190513 (Ver col with std)/1-multi_rest_func/freq/'
  plt.savefig(save_results_to + str(i) +'.jpeg')  
  i=i+1

#########################################
######################################
# firing frequency of different distances
del bin_list[-1]
i=1
for layer_name in ctx_M1_layers.keys():
  figure, ax1 = plt.subplots()
  ax1.plot(bin_list, freq_targets_ave[layer_name][0], color='g', linewidth=7.0)
  plt.fill_between(bin_list, freq_targets_ave[layer_name][0] - freq_targets_sem[layer_name][0],  freq_targets_ave[layer_name][0] + freq_targets_sem[layer_name][0], facecolor='lightgreen', alpha= 0.4)
  ax1.plot(bin_list, freq_targets_ave[layer_name][1], color='r', linewidth=7.0)
  plt.fill_between(bin_list, freq_targets_ave[layer_name][1] - freq_targets_sem[layer_name][1],  freq_targets_ave[layer_name][1] + freq_targets_sem[layer_name][1], facecolor='lightsalmon', alpha= 0.4)
  ax1.plot(bin_list, freq_targets_ave[layer_name][2], color='b', linewidth=7.0)
  plt.fill_between(bin_list, freq_targets_ave[layer_name][2] - freq_targets_sem[layer_name][2],  freq_targets_ave[layer_name][2] + freq_targets_sem[layer_name][2], facecolor='skyblue', alpha= 0.4)
  ax1.plot(bin_list, freq_targets_ave[layer_name][3], color='m', linewidth=7.0)
  plt.fill_between(bin_list, freq_targets_ave[layer_name][3] - freq_targets_sem[layer_name][3],  freq_targets_ave[layer_name][3] + freq_targets_sem[layer_name][3], facecolor='plum', alpha= 0.4)
  ax1.plot(bin_list, freq_targets_ave[layer_name][4], color='y', linewidth=7.0)
  plt.fill_between(bin_list, freq_targets_ave[layer_name][4] - freq_targets_sem[layer_name][4],  freq_targets_ave[layer_name][4] + freq_targets_sem[layer_name][4], facecolor='khaki', alpha= 0.4)
  ax1.plot(bin_list, freq_targets_ave[layer_name][5], color='g', linewidth=7.0)
  plt.fill_between(bin_list, freq_targets_ave[layer_name][5] - freq_targets_sem[layer_name][5],  freq_targets_ave[layer_name][5] + freq_targets_sem[layer_name][5], facecolor='lightgreen', alpha= 0.4)
  ax1.plot(bin_list, freq_targets_ave[layer_name][6], color='r', linewidth=7.0)
  plt.fill_between(bin_list, freq_targets_ave[layer_name][6] - freq_targets_sem[layer_name][6],  freq_targets_ave[layer_name][6] + freq_targets_sem[layer_name][6], facecolor='lightsalmon', alpha= 0.4)
  ax1.plot(bin_list, freq_targets_ave[layer_name][7], color='b', linewidth=7.0)
  plt.fill_between(bin_list, freq_targets_ave[layer_name][7] - freq_targets_sem[layer_name][7],  freq_targets_ave[layer_name][7] + freq_targets_sem[layer_name][7], facecolor='skyblue', alpha= 0.4)
  ax1.plot(bin_list, freq_targets_ave[layer_name][8], color='m', linewidth=7.0)
  plt.fill_between(bin_list, freq_targets_ave[layer_name][8] - freq_targets_sem[layer_name][8],  freq_targets_ave[layer_name][8] + freq_targets_sem[layer_name][8], facecolor='plum', alpha= 0.4)
  ax1.plot(bin_list, freq_targets_ave[layer_name][9], color='y', linewidth=7.0)
  plt.fill_between(bin_list, freq_targets_ave[layer_name][9] - freq_targets_sem[layer_name][9],  freq_targets_ave[layer_name][9] + freq_targets_sem[layer_name][9], facecolor='khaki', alpha= 0.4)
  
  ax1.spines['right'].set_visible(False)
  ax1.spines['top'].set_visible(False)
  for line in ['left', 'bottom']:
    ax1.spines[line].set_linewidth(5.0)
    
  ax1.tick_params(axis='both',length = 10.0,width = 5.0)
  plt.xlabel("time (ms)")
  plt.ylabel("Average frequency (Hz)")
  plt.title('Response of' + layer_name + ' neurons')
  plt.legend(("30 um", "60 um", "90 um", "120 um" , "150 um", "180 um", "210 um", "240 um", "270 um", "300 um"))  
  save_results_to = '/home/morteza/Desktop/reports/190604 (responses at different distances)/freq/'
  plt.savefig(save_results_to + str(i) +'.jpeg')
  
  figure, ax2 = plt.subplots()
  ax2.plot(bin_list, freq_targets_rest_ave[layer_name][0],color='g', linewidth=5.0)
  plt.fill_between(bin_list, freq_targets_rest_ave[layer_name][0] - freq_targets_rest_sem[layer_name][0],  freq_targets_rest_ave[layer_name][0] + freq_targets_rest_sem[layer_name][0], facecolor='lightgreen', alpha= 0.4)
  ax2.plot(bin_list, freq_targets_rest_ave[layer_name][1],color='r', linewidth=5.0)
  plt.fill_between(bin_list, freq_targets_rest_ave[layer_name][1] - freq_targets_rest_sem[layer_name][1],  freq_targets_rest_ave[layer_name][1] + freq_targets_rest_sem[layer_name][1], facecolor='lightsalmon', alpha= 0.4)
  ax2.plot(bin_list, freq_targets_rest_ave[layer_name][2],color='b', linewidth=5.0)
  plt.fill_between(bin_list, freq_targets_rest_ave[layer_name][2] - freq_targets_rest_sem[layer_name][2],  freq_targets_rest_ave[layer_name][2] + freq_targets_rest_sem[layer_name][2], facecolor='skyblue', alpha= 0.4)
  ax2.plot(bin_list, freq_targets_rest_ave[layer_name][3],color='m', linewidth=5.0)
  plt.fill_between(bin_list, freq_targets_rest_ave[layer_name][3] - freq_targets_rest_sem[layer_name][3],  freq_targets_rest_ave[layer_name][3] + freq_targets_rest_sem[layer_name][3], facecolor='plum', alpha= 0.4)
  ax2.plot(bin_list, freq_targets_rest_ave[layer_name][4],color='y', linewidth=5.0)
  plt.fill_between(bin_list, freq_targets_rest_ave[layer_name][4] - freq_targets_rest_sem[layer_name][4],  freq_targets_rest_ave[layer_name][4] + freq_targets_rest_sem[layer_name][4], facecolor='khaki', alpha= 0.4)
  ax2.plot(bin_list, freq_targets_rest_ave[layer_name][5],color='g', linewidth=5.0)
  plt.fill_between(bin_list, freq_targets_rest_ave[layer_name][5] - freq_targets_rest_sem[layer_name][5],  freq_targets_rest_ave[layer_name][5] + freq_targets_rest_sem[layer_name][5], facecolor='paleturquoise', alpha= 0.4)
  ax2.plot(bin_list, freq_targets_rest_ave[layer_name][6],color='r', linewidth=5.0)
  plt.fill_between(bin_list, freq_targets_rest_ave[layer_name][6] - freq_targets_rest_sem[layer_name][6],  freq_targets_rest_ave[layer_name][6] + freq_targets_rest_sem[layer_name][6], facecolor='lightcoral', alpha= 0.4)
  ax2.plot(bin_list, freq_targets_rest_ave[layer_name][7],color='b', linewidth=5.0)
  plt.fill_between(bin_list, freq_targets_rest_ave[layer_name][7] - freq_targets_rest_sem[layer_name][7],  freq_targets_rest_ave[layer_name][7] + freq_targets_rest_sem[layer_name][7], facecolor='lightblue', alpha= 0.4)
  ax2.plot(bin_list, freq_targets_rest_ave[layer_name][8],color='m', linewidth=5.0)
  plt.fill_between(bin_list, freq_targets_rest_ave[layer_name][8] - freq_targets_rest_sem[layer_name][8],  freq_targets_rest_ave[layer_name][8] + freq_targets_rest_sem[layer_name][8], facecolor='pink', alpha= 0.4)
  ax2.plot(bin_list, freq_targets_rest_ave[layer_name][9],color='y', linewidth=5.0)
  plt.fill_between(bin_list, freq_targets_rest_ave[layer_name][9] - freq_targets_rest_sem[layer_name][9],  freq_targets_rest_ave[layer_name][9] + freq_targets_rest_sem[layer_name][9], facecolor='burlywood', alpha= 0.4)
  
  ax2.spines['right'].set_visible(False)
  ax2.spines['top'].set_visible(False)
  for line in ['left', 'bottom']:
    ax2.spines[line].set_linewidth(5.0)
  
  ax2.tick_params(axis='both',length = 10.0,width = 5.0)
  
  
  plt.xlabel("time (ms)")
  plt.ylabel("Average frequency (Hz)")
  plt.title('Response of ' + layer_name + ' control neurons')
  plt.legend(("30 um", "60 um", "90 um", "120 um" , "150 um", "180 um", "210 um", "240 um", "270 um", "300 um"))  
  save_results_to = '/home/morteza/Desktop/reports/190604 (responses at different distances)/freq(control)/'
  plt.savefig(save_results_to + str(i) +'.jpeg')  
  i=i+1

#original graph
x = ['L1_ENGC','L1_SBC', 'L23_CC', 'L23_PV', 'L23_SST', 'L23_VIP', 'L5A_CC', 'L5A_CS', 'L5A_CT', 'L5A_PV', 'L5A_SST', 'L5B_CC', 'L5B_CS', 'L5B_PT', 'L5B_PV', 'L5B_SST','L6_CT', 'L6_PV', 'L6_SST']
for item in excluded_list:
    x.remove(item[3:])
i=1
figure, ax3 = plt.subplots()
for layer_name in ctx_M1_layers.keys():
    if layer_name not in excluded_list:
        #figure, ax3 = plt.subplots()
        dis = [30, 60, 90, 120, 150, 180, 210, 240, 270, 300, 330, 360, 390, 420, 450]
        plt.plot(dis, pnt_amps[layer_name], lw=5, marker = 'o', markersize=10)
        plt.xlabel("Distance (um)", fontsize = 70)
        plt.ylabel("Change in frequency (Hz)", fontsize= 70)
        plt.legend(x, loc='lower left', ncol=4, fontsize=40, fancybox=True, shadow=True, bbox_to_anchor = (0.42, 0.01))
    i=i+1
plt.axhline(y=0.0, color='k', lw = 4, linestyle='-')
plt.xlim(30, 450)
plt.title('Response of neurons to ' + input_layer_name + ' stimulation', fontsize=70)
save_results_to = '/home/morteza/Desktop/reports/190917/'
plt.savefig(save_results_to + input_layer_name +'.jpeg')


# normalized graph
x = ['L1_ENGC','L1_SBC', 'L23_CC', 'L23_PV', 'L23_SST', 'L23_VIP', 'L5A_CC', 'L5A_CS', 'L5A_CT', 'L5A_PV', 'L5A_SST', 'L5B_CC', 'L5B_CS', 'L5B_PT', 'L5B_PV', 'L5B_SST','L6_CT', 'L6_PV', 'L6_SST']
for item in excluded_list:
    x.remove(item[3:])
i=1
figure, ax3 = plt.subplots()
for layer_name in ctx_M1_layers.keys():
    if layer_name not in excluded_list:
        #figure, ax3 = plt.subplots()
        dis = [30, 60, 90, 120, 150, 180, 210, 240, 270, 300, 330, 360, 390, 420, 450]
        plt.plot(dis, pnt_amps_norm[layer_name], lw=5, marker = 'o', markersize=10)
        plt.xlabel("Distance (um)", fontsize = 70)
        plt.ylabel("Change in frequency (Hz)", fontsize= 70)
        plt.legend(x, loc='lower left', ncol=4, fontsize=40, fancybox=True, shadow=True, bbox_to_anchor = (0.42, 0.01))
    i=i+1
plt.axhline(y=0.0, color='k', lw = 4, linestyle='-')
plt.ylim(-3, 3)
plt.xlim(30, 450)
plt.title('Response of neurons to ' + input_layer_name + ' stimulation', fontsize=70)
save_results_to = '/home/morteza/Desktop/reports/190917/'
plt.savefig(save_results_to + input_layer_name +'_norm.jpeg')
'''
