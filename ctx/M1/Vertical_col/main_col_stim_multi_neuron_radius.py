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
def get_input_column_layers_ctx_M1(ctx_M1_layers, target_layer_name, layer_name, step_radius):
    circle_center = [[0.0, 0.0]]
    inside_gids = {}
    outside_gids = {}
    for n in range(len(step_radius)):
        inside_gids [n] = []
        outside_gids [n] = []

    i=0
    for rad in step_radius:
        layer_gids = nest.GetNodes(ctx_M1_layers[layer_name])[0]
        neuron_positions = ntop.GetPosition(layer_gids)
        for nn in range(len(layer_gids)):
            if np.linalg.norm([(neuron_positions[nn][0] - circle_center[0][0]), (neuron_positions[nn][1] - circle_center[0][1])]) <= rad:
                inside_gids[i].append(layer_gids[nn])
            else:
                outside_gids[i].append(layer_gids[nn])

        i=i+1

    if layer_name == target_layer_name:
        print ('all', layer_name ,'neuron numbers are', len(layer_gids), 'and for column with radius', rad, 'um only', len(inside_gids), 'neurons will be stimulated inside the column using poisson generator.')
        k=0
        for i in list(inside_gids):
            if inside_gids[i] == []:
                inside_gids.pop(i)
                outside_gids.pop( i )
                k=k+1

        if k > 0:
            for n in range(len(step_radius)-k):
                inside_gids[n] = inside_gids.pop(n+k)
                outside_gids [n] = outside_gids.pop( n + k )

            step_radius = step_radius [k:]

    return inside_gids , outside_gids, step_radius

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
    #topo_extend = [M1_layer_size [0] * int( scalefactor [0] ) + 1., M1_layer_size [1] * int( scalefactor [1] ) + 1., 2.]
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
def layer_poisson_generator(layer_gid, layer_name, params={"withgid": True, "withtime": True, "to_file": False}):
    #def layer_spike_detector(layer_gid, layer_name, params={"withgid": True, "withtime": True, "to_file": True, 'fbuffer_size': 8192}):
    params.update({'label': layer_name})

    PSG_arrow={}
    # stimulation by using PSG
    #if layer_name == 'M1_L23_VIP':
    for i in range(len(layer_gid)):
        PSG_arrow [i] = nest.Create('poisson_generator', 1, params={'rate':  10000.0}) #, 'label': layer_name})
        nest.Connect(pre=PSG_arrow[i], post=layer_gid[i], syn_spec={'weight': 100.0, 'delay': 1.5})

    return PSG_arrow

###############################################################
def layer_spike_detector(layer_gids_inside, layer_gids_outside, layer_name, record_onset, ignore_time, ran_inside_neurons, ran_outside_neurons, params = {"withgid": True, "withtime": True, "to_file": False}):
    print( 'spike detector for ' + layer_name )
    params.update( {'label': layer_name, "start": float(record_onset)} )

    # Add detector for all neuron types
    detector_in = {}
    for n in range( len( layer_gids_inside ) ):
        detector_in[n] = nest.Create( "spike_detector", params=params )
        nest.Connect( pre=layer_gids_inside[n], post=detector_in[n] )

    detector_out = {}
    for n in range(len(layer_gids_outside)):
        detector_out[n] = nest.Create( "spike_detector", params=params )
        nest.Connect( pre=layer_gids_outside[n], post=detector_out[n])

    # generating some random numbers to choose random neuron to record
    nest_mm_in = {}
    nest_mm_out = {}
    for n in range( len( layer_gids_inside ) ):
        nest_mm_in[n] = {}
        nest_mm_out[n] = {}
        inside_random = np.random.choice(len(layer_gids_inside[n]), int(ran_inside_neurons[n]), replace=False)
        outside_random = np.random.choice(len(layer_gids_outside[n]), int(ran_outside_neurons[n]), replace=False)

        # add multimeter just for to record V_m and conductance (inhibitory and excitatory) of a single cell each cell population
        for num in range( int( ran_inside_neurons[n] ) ):
            nest_mm_in[n][num] = nest.Create( 'multimeter', params={"interval": 0.1, "record_from": ["V_m", "g_ex", "g_in"], "withgid": True, "to_file": False, 'label': layer_name, 'withtime': True, "start": float(record_onset), 'use_gid_in_filename': False} )
            nest.Connect(nest_mm_in[n][num], [layer_gids_inside[n][inside_random[num]]])

        for num in range( int( ran_outside_neurons[n] ) ):
            nest_mm_out[n][num] = nest.Create( 'multimeter', params={"interval": 0.1, "record_from": ["V_m", "g_ex", "g_in"], "withgid": True, "to_file": False, 'label': layer_name, 'withtime': True, "start": float(record_onset), 'use_gid_in_filename': False} )  #
            nest.Connect(nest_mm_out[n][num], [layer_gids_outside[n][outside_random[num]]])

        voltmeter_single = nest.Create( "voltmeter", params={'to_file': False, 'label': layer_name, "start": float( ignore_time ), "withgid": True, "withtime": True} )  # 'use_gid_in_filename': True,   "start": float(ignore_time),
        nest.Connect(voltmeter_single, [layer_gids_inside[n][2]] )

    return detector_in, detector_out, nest_mm_in, nest_mm_out, voltmeter_single

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
        spike_t_rest = np.around(spike_t_rest - initial_ignore, decimals=2)
    if spike_t.size != 0:
        spike_t = np.around(spike_t - (initial_ignore+(num_trials*simulation_time)), decimals=2)

    spike_t_rest = np.sort( spike_t_rest )
    spike_t = np.sort( spike_t )

    # find out the spikes of each trial in rest and stimulated time window
    spike_t_trials = {}
    if spike_t.size == 0:
        for n in range(num_trials):
            spike_t_trials[n]= np.array([])
    else:
        m=0
        key = 0
        for n in range(num_trials):
            if n < num_trials-1:
                ii=0
                for t in spike_t:
                    if t > (simulation_time * (n + 1)):
                        spike_t_trials[n] = spike_t[m:ii]
                        m=ii
                        break

                    ii = ii + 1
                    if ii == len(spike_t):
                        if key == 0:
                            spike_t_trials [n] = spike_t [m:]
                            key = key +1
                        else:
                            spike_t_trials[n] = []

            elif n == num_trials -1:
                if ii == len(spike_t):
                    spike_t_trials [n] = []
                else:
                    spike_t_trials [n] = spike_t [m:]

    spike_t_rest_trials = {}
    if spike_t_rest.size == 0:
        for n in range(num_trials):
            spike_t_rest_trials[n]= np.array([])
    else:
        m = 0
        key = 0
        for n in range( num_trials ):
            if n < num_trials - 1:
                ii = 0
                for t in spike_t_rest:
                    if t > (simulation_time * (n + 1)):
                        spike_t_rest_trials [n] = spike_t_rest[m:ii]
                        m = ii
                        break

                    ii = ii + 1
                    if ii == len (spike_t_rest):
                        if key == 0:
                            spike_t_rest_trials [n] = spike_t_rest [m:]
                            key = key +1
                        else:
                            spike_t_rest_trials[n] = []

            elif n == num_trials - 1:
                if ii == len (spike_t_rest):
                    spike_t_rest_trials [n] = []
                else:
                    spike_t_rest_trials [n] = spike_t_rest [m:]


    for n in range( num_trials - 1 ):
        if spike_t_trials[n+1] == []:
            break
        else:
            spike_t_trials[n+1] = spike_t_trials[n+1] - ((n+1)*simulation_time)

    for n in range( num_trials - 1 ):
        if spike_t_rest_trials[n+1] == []:
            break
        else:
            spike_t_rest_trials[n + 1] = spike_t_rest_trials [n + 1] - ((n + 1) * simulation_time)



    spike_count_rest_trials = {}
    freq_rest_trials = {}
    spike_count_trials = {}
    freq_trials = {}
    for n in range(num_trials):
        spike_t_trials[n] = np.sort( spike_t_trials[n] )
        spike_t_rest_trials[n] = np.sort( spike_t_rest_trials[n] )

        if spike_t_trials[n] == []:
            spike_count_trials[n] = [0]*len(bin_list-1)

        if spike_t_rest_trials[n] == []:
            spike_count_rest_trials[n] = [0]*len(bin_list-1)

        spike_count_trials[n] = np.histogram( (spike_t_trials[n].tolist()), bins=bin_list ) [0]
        spike_count_rest_trials[n] = np.histogram( (spike_t_rest_trials[n].tolist()), bins=bin_list ) [0]

        freq_trials[n] = spike_count_trials[n] * (1000 / (bin * len( layer_gids )))
        freq_rest_trials[n] = spike_count_rest_trials[n] * (1000 / (bin * len( layer_gids )))


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

###############################################################################################################################################################################
###############################################################################################################################################################################
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
simulation_time = sim_params['simDuration']    # Should be 1500 ms
initial_ignore = sim_params['initial_ignore']
###########################
# 4) making layers and doing all connections of M1
print('create and connect the M1 layers')
start_time = time.time()
ctx_M1_layers, SubSubRegion_Excitatory, SubSubRegion_Inhibitory =instantiate_ctx_M1(ctx_M1_params, sim_params['scalefactor'])
#with open('./log/' + 'x_performance.txt', 'a') as file:
  #file.write('M1_Construction_Time: ' + str(time.time() - start_time) + '\n')


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
# 6) set paramaters for the stimulation of target neurons inside the column\
record_onset = 499.9
record_intervals = 500.0
radius = 0.1
#step_radius = [0.02, 0.04, 0.06, 0.08, 0.1, 0.12, 0.14, 0.16, 0.18, 0.2]
step_radius = [0.08, 0.1, 0.12, 0.14]
###############################
# 7) selecting the target neurons for the stimulation located inside the column and connecting the PSG
poisson_gens = []                 # in case of one trial     poisson_gens = {}
target_layer_name = 'M1_L23_VIP'
target_gids_inside, target_gids_outside, step_radius = get_input_column_layers_ctx_M1(ctx_M1_layers, target_layer_name, target_layer_name,  step_radius)
#poisson_gens[layer_name]= layer_poisson_generator(target_gids_inside, layer_name, stim_onset, stim_offset, sim_params['initial_ignore'])   # this was good for one trial
poisson_gens.append(layer_poisson_generator(target_gids_inside, target_layer_name))

###############################
# 8) recording from other neurons inside and outside the circle
multimeters_in = {}
multimeters_out = {}
voltmeters = {}
detectors_in = {}
detectors_out = {}
layer_gids_inside = {}
layer_gids_outside = {}
layer_gids_all = {}
inside_neurons_torecord = {}
outside_neurons_torecord = {}

for layer_name in ctx_M1_layers.keys():
    layer_gids_inside[layer_name], layer_gids_outside[layer_name], step_radius = get_input_column_layers_ctx_M1(ctx_M1_layers, target_layer_name, layer_name, step_radius)
    inside_neurons_torecord[layer_name] = [None] * len( step_radius )
    outside_neurons_torecord[layer_name] = [None] * len( step_radius )
    for i in range( len( step_radius ) ):
        if len(layer_gids_inside[layer_name][i] ) < 5:
            inside_neurons_torecord[layer_name][i] = len(layer_gids_inside[layer_name][i])
        else:
            inside_neurons_torecord[layer_name][i] = 5
        outside_neurons_torecord[layer_name][i] = 5

        print('number of neurons for population of ' + layer_name + 'neurons inside and outside of the column with radius ' + str(step_radius [i] ) + 'um are:', str(inside_neurons_torecord[layer_name][i]), 'and', str(outside_neurons_torecord[layer_name][i]) )

    detectors_in[layer_name], detectors_out[layer_name], multimeters_in[layer_name], multimeters_out[layer_name], voltmeters[layer_name] = layer_spike_detector(layer_gids_inside[layer_name], layer_gids_outside[layer_name],
                                                                                                                                                               layer_name, record_onset, sim_params['initial_ignore'],
                                                                                                                                                                inside_neurons_torecord[layer_name], outside_neurons_torecord[layer_name])
    layer_gids_all [layer_name] ={}
    for i in range(len(step_radius)):
        layer_gids_all [layer_name] [i] = []
    for i in range(len(step_radius)):
        layer_gids_all [layer_name][i] = layer_gids_inside [layer_name][i] + layer_gids_outside [layer_name][i]

########################################################################################################################
########################################################################################################################
# 9) simulation
num_trials = 5
print ('Simulation Started:')
start_time=time.time()
for n in range(num_trials):
    if n == 0:
        for i in range( len( step_radius ) ):
            nest.SetStatus( poisson_gens [0][i], {'origin': nest.GetKernelStatus() ['time'], "start": (num_trials*simulation_time) + (2*record_intervals), 'stop': (num_trials*simulation_time)+(3*record_intervals)} )

        nest.Simulate( (num_trials*simulation_time) + initial_ignore + simulation_time )
    elif n == num_trials -1:
        for i in range( len( step_radius ) ):
            nest.SetStatus( poisson_gens[0][i], {'origin': nest.GetKernelStatus() ['time'], "start": record_intervals, 'stop': 2*record_intervals} )

        nest.Simulate( simulation_time+ 1.0 )
    else:
        for i in range( len( step_radius ) ):
            nest.SetStatus( poisson_gens[0][i], {'origin': nest.GetKernelStatus() ['time'], "start": record_intervals, 'stop': 2*record_intervals} )

        nest.Simulate(simulation_time)

#with open('./log/'+'x_performance.txt', 'a') as file:
  #file.write('Simulation_Elapse_Time: '+str(time.time()-start_time)+'\n')

#########################################################################
# 8) calculating the average graph of all trials for all recorded neurons
# first averaging the multimeter data including membrane potential and conductance (inhibitory and excitatory) among the recorded neurons
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
for layer_name in ctx_M1_layers.keys():
    mps_in[layer_name] = {}
    mps_out[layer_name] = {}
    g_ins_in[layer_name] = {}
    g_ins_out[layer_name] = {}
    g_exs_in[layer_name] = {}
    g_exs_out[layer_name] = {}
    spike_t_in[layer_name] = {}
    spike_t_out[layer_name] = {}
    spike_t_all[layer_name] = {}
    for i in range( len( step_radius)):
        spike_t_in[layer_name][i] = []
        spike_t_out[layer_name][i] = []
        spike_t_all[layer_name][i] = []
        mps_in [layer_name][i] = []
        mps_out [layer_name][i] = []
        g_ins_in [layer_name][i] = []
        g_ins_out [layer_name][i] =[]
        g_exs_in [layer_name][i] = []
        g_exs_out [layer_name][i] = []
    for i in range(len(step_radius)):
        events_in[i] = {}
        events_out[i] = {}
    
        for num in range( inside_neurons_torecord[layer_name][i] ):
            events_in[i] [num] = nest.GetStatus( multimeters_in [layer_name] [i][num] ) [0] ["events"]

        for num in range( outside_neurons_torecord[layer_name][i] ):
            events_out[i] [num] = nest.GetStatus( multimeters_out [layer_name][i] [num] ) [0] ["events"]

        spike_t_in [layer_name][i]= nest.GetStatus( detectors_in[layer_name][i]) [0] ['events'] ['times']    # recorded from all neurons inside
        spike_t_out [layer_name][i] = nest.GetStatus( detectors_out[layer_name][i]) [0] ['events'] ['times']  # recorded from all neurons outside
        spike_t_all [layer_name][i] = np.asarray( ((spike_t_in[layer_name][i]).tolist()) + ((spike_t_out[layer_name][i]).tolist()))

        if inside_neurons_torecord[layer_name][i] > 1:
            g_ins_in[layer_name][i], g_exs_in[layer_name][i], mps_in[layer_name][i]= average_recorders_outputs(events_in[i], inside_neurons_torecord[layer_name][i])  # average among the recorded neurons inside
        else:
            g_ins_in[layer_name][i] = events_in[i] [0] ['g_in']
            g_exs_in[layer_name][i] = events_in[i] [0] ['g_ex']
            mps_in[layer_name][i] = events_in[i] [0] ['V_m']

        if outside_neurons_torecord[layer_name][i] > 1:
            g_ins_out[layer_name][i], g_exs_out[layer_name][i], mps_out[layer_name][i]= average_recorders_outputs(events_out[i], outside_neurons_torecord[layer_name][i]) # average among the recorded neurons outside
        else:
            g_ins_out[layer_name][i] = events_out[i] [0] ['g_in']
            g_exs_out[layer_name][i] = events_out[i] [0] ['g_ex']
            mps_out[layer_name][i] = events_out[i] [0] ['V_m']
    
#-----------------------------------------------------------------------------------------------------------------------
# now average among the trials
for i in range(len(step_radius)):
    tt = events_in[i][0] ['times']
    times = ((events_in[i][0]['times'])[0:15000])
mps_in_ave = {}
mps_out_ave = {}
g_ins_in_ave = {}
g_ins_out_ave = {}
g_exs_in_ave = {}
g_exs_out_ave = {}
for layer_name in ctx_M1_layers.keys():
    mps_in_ave[layer_name] = {}
    mps_out_ave[layer_name] = {}
    g_ins_in_ave[layer_name] = {}
    g_ins_out_ave[layer_name] = {}
    g_exs_in_ave[layer_name] = {}
    g_exs_out_ave[layer_name] = {}
    for i in range( len( step_radius ) ):
        mps_in_ave[layer_name][i] = []
        mps_out_ave[layer_name][i] = []
        g_ins_in_ave[layer_name][i] = []
        g_ins_out_ave[layer_name][i] = []
        g_exs_in_ave[layer_name][i] =[]
        g_exs_out_ave[layer_name][i] = []
    for i in range( len( step_radius ) ):
        mps_in_ave [layer_name][i] = average_over_trials(num_trials, mps_in[layer_name][i])
        mps_out_ave [layer_name][i] = average_over_trials(num_trials, mps_out[layer_name][i])
        g_ins_in_ave [layer_name][i] = average_over_trials(num_trials, g_ins_in[layer_name][i])
        g_ins_out_ave [layer_name][i] = average_over_trials(num_trials, g_ins_out[layer_name][i])
        g_exs_in_ave [layer_name][i] = average_over_trials(num_trials, g_exs_in[layer_name][i])
        g_exs_out_ave [layer_name][i] = average_over_trials(num_trials, g_exs_out[layer_name][i])

#-----------------------------------------------------------------------------------------------------------------------
# calculate the firing rate of cells and finding the average among the cells and among the trials
# first it overlap all the events in one whole time window and then devide to number pf recorded cells
bin = 20.0
num_windows = simulation_time/bin
bin_list = [None] * int(num_windows+1)
for nn in range(int(num_windows+1)):
    bin_list[nn] = nn * 20

freq_in = {}
freq_out = {}
freq_rest = {}
freq_in_rest = {}
freq_out_rest = {}
for layer_name in ctx_M1_layers.keys():
    freq_in[layer_name] = {}
    freq_out[layer_name] = {}
    freq_rest[layer_name] = {}
    freq_in_rest[layer_name] = {}
    freq_out_rest[layer_name] = {}
    for i in range( len( step_radius ) ):
        freq_in [layer_name][i] = []
        freq_out [layer_name][i] = []
        freq_rest [layer_name][i] = []
        freq_in_rest [layer_name][i] = []
        freq_out_rest [layer_name][i] = []
    for i in range( len( step_radius ) ):
        print('number of spikes for population of '+ layer_name + ' neurons inside and outside of the column with radius ' + str(step_radius[i])+'um are:', len( spike_t_in[layer_name][i]), 'and', len( spike_t_out[layer_name][i]))
        freq_in[layer_name][i] = spike_sort_count_freq(num_trials, spike_t_in[layer_name][i], sim_params['initial_ignore'], sim_params['simDuration'], layer_gids_inside[layer_name][i], bin_list)[0]
        freq_out[layer_name][i] = spike_sort_count_freq(num_trials, spike_t_out[layer_name][i], sim_params ['initial_ignore'], sim_params ['simDuration'], layer_gids_outside[layer_name][i],bin_list)[0]
        freq_in_rest [layer_name][i] = spike_sort_count_freq(num_trials, spike_t_in[layer_name][i], sim_params ['initial_ignore'], sim_params ['simDuration'], layer_gids_inside[layer_name][i], bin_list)[1]
        freq_out_rest [layer_name][i] = spike_sort_count_freq(num_trials, spike_t_out[layer_name][i], sim_params ['initial_ignore'], sim_params ['simDuration'], layer_gids_outside[layer_name][i], bin_list)[1]

#######################################################################################################################
# then here the calculated ferquency of one averaged cell again be averaged among the number of trials
# calculation of Standard deviation (sd) and standard error of mean (sem)
layer_dic = {'M1_L1_ENGC':1,'M1_L1_SBC':2, 'M1_L23_CC':3, 'M1_L23_PV':4, 'M1_L23_SST':5, 'M1_L23_VIP':6, 'M1_L5A_CC':7, 'M1_L5A_CS':8, 'M1_L5A_CT':9, 'M1_L5A_PV':10, 'M1_L5A_SST':11,
             'M1_L5B_CC':12, 'M1_L5B_CS':13, 'M1_L5B_PT':14, 'M1_L5B_PV':15, 'M1_L5B_SST':16,'M1_L6_CT':17, 'M1_L6_PV':18, 'M1_L6_SST':19}

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

amp_changes_in = {}
amp_changes_out = {}
#amp_changes_in_percent = {}
#amp_changes_out_percent = {}
amp_changes_in_abs = {}
amp_changes_out_abs = {}
amp_changes_in_abs_norm = {}
amp_changes_out_abs_norm = {}
for rad in step_radius:
    amp_changes_in_abs[rad] = []
    amp_changes_out_abs[rad] = []
    amp_changes_in_abs_norm[rad] = []
    amp_changes_out_abs_norm[rad] = []

for layer_name in ctx_M1_layers.keys():
    freq_in_sd[layer_name] = {}
    freq_out_sd[layer_name]  = {}
    freq_in_rest_sd[layer_name]  = {}
    freq_out_rest_sd[layer_name]  = {}
    freq_in_sem[layer_name]  = {}
    freq_out_sem[layer_name]  = {}
    freq_in_rest_sem[layer_name]  = {}
    freq_out_rest_sem[layer_name]  = {}
    freq_in_ave[layer_name]  = {}
    freq_out_ave[layer_name]  = {}
    freq_in_rest_ave[layer_name]  = {}
    freq_out_rest_ave[layer_name]  = {}

    amp_changes_in[layer_name] = {}
    amp_changes_out[layer_name] = {}
    for i in range( len( step_radius ) ):
        freq_in_sd [layer_name][i] = []
        freq_out_sd [layer_name][i] = []
        freq_in_rest_sd [layer_name][i] = []
        freq_out_rest_sd [layer_name][i] = []
        freq_in_sem [layer_name][i] = []
        freq_out_sem [layer_name][i] = []
        freq_in_rest_sem [layer_name][i] = []
        freq_out_rest_sem [layer_name][i] = []
        freq_in_ave [layer_name][i] = []
        freq_out_ave [layer_name][i] = []
        freq_in_rest_ave [layer_name][i] = []
        freq_out_rest_ave [layer_name][i] = []

        amp_changes_in [layer_name][i] = []
        amp_changes_out [layer_name][i] = []
    for i in range( len( step_radius ) ):
        freq_in_ave [layer_name][i] = calcualte_ave_sem_sd(num_trials, freq_in[layer_name][i])[0]
        freq_in_sem[layer_name][i] = calcualte_ave_sem_sd(num_trials, freq_in[layer_name][i])[1]
        freq_in_sd[layer_name][i] = calcualte_ave_sem_sd(num_trials, freq_in[layer_name][i])[2]

        freq_in_rest_ave [layer_name][i] = calcualte_ave_sem_sd( num_trials, freq_in_rest [layer_name][i] ) [0]
        freq_in_rest_sem [layer_name][i] = calcualte_ave_sem_sd( num_trials, freq_in_rest [layer_name][i] ) [1]
        freq_in_rest_sd [layer_name][i] = calcualte_ave_sem_sd( num_trials, freq_in_rest [layer_name][i] ) [2]

        freq_out_ave [layer_name][i] = calcualte_ave_sem_sd( num_trials, freq_out [layer_name][i] ) [0]
        freq_out_sem [layer_name][i] = calcualte_ave_sem_sd( num_trials, freq_out [layer_name][i] ) [1]
        freq_out_sd [layer_name][i] = calcualte_ave_sem_sd( num_trials, freq_out [layer_name][i] ) [2]

        freq_out_rest_ave [layer_name][i] = calcualte_ave_sem_sd( num_trials, freq_out_rest [layer_name][i] ) [0]
        freq_out_rest_sem [layer_name][i] = calcualte_ave_sem_sd( num_trials, freq_out_rest [layer_name][i] ) [1]
        freq_out_rest_sd [layer_name][i] = calcualte_ave_sem_sd( num_trials, freq_out_rest [layer_name][i] ) [2]

i=0
for rad in step_radius:
    for layer_name in ctx_M1_layers.keys():
        amp_changes_in [layer_name][i]= np.average( freq_in_ave [layer_name][i] [25:50] ) - np.average(freq_in_rest_ave [layer_name][i] [25:50] )
        amp_changes_in_abs[rad].append(amp_changes_in [layer_name][i])

        amp_changes_out [layer_name][i] = np.average( freq_out_ave [layer_name][i] [25:50] ) - np.average(freq_out_rest_ave [layer_name][i] [25:50] )
        amp_changes_out_abs[rad].append(amp_changes_out [layer_name][i])

    i=i+1

for rad in step_radius:
    amp_changes_in_abs_norm[rad] = [x * 100 / amp_changes_in [target_layer_name] [0] for x in amp_changes_in_abs[rad]]
    amp_changes_out_abs_norm[rad]= [x * 100 / amp_changes_out [target_layer_name] [0] for x in amp_changes_out_abs[rad]]
'''
for i in range( len( step_radius ) ):
    amp_changes_in_abs_norm[i] = [x*100 / amp_changes_in[target_layer_name][i] for x in amp_changes_in_abs[i]]
    amp_changes_out_abs_norm[i] = [x*100 / amp_changes_out[target_layer_name][i] for x in amp_changes_in_abs[i]]


with open('./log/log_radius/' + 'stim_param.txt', 'a') as file:
  file.write('amp_changes_in_abs_norm : ' + amp_changes_in_abs_norm  + '\n' + 'amp_changes_out_abs_norm : ' + amp_changes_out_abs_norm  + '\n')

np.savetxt('log/log_radius/stim_param.txt', (amp_changes_in_abs_norm, radius), fmt='%1.1f', delimiter=',')

f= open('/home/morteza/PycharmProjects/postk_wb/code/ctx/M1/Vertical_col/log/log_radius/guru99.txt','w+')  # w for write, r for read, a for append and + for create if there is no such a file
for item in a:
    f.write( "%f\n" % item )

f = open( "/home/morteza/PycharmProjects/postk_wb/code/ctx/M1/Vertical_col/log/log_radius/guru99.txt", "a" )
f.write('the radius is: %f\n' %radius)
f.close()
for i in range(10):
     f.write("This is line %d\n" % (i+1))

f.close()
'''

for rad in step_radius:
    del amp_changes_in_abs [rad] [layer_dic [target_layer_name] - 1:layer_dic [target_layer_name]]
    del amp_changes_out_abs [rad] [layer_dic [target_layer_name] - 1:layer_dic [target_layer_name]]
    del amp_changes_in_abs_norm[rad][layer_dic[target_layer_name]-1:layer_dic[target_layer_name]]
    del amp_changes_out_abs_norm[rad][layer_dic[target_layer_name]-1:layer_dic[target_layer_name]]

print ('Simulation successfully finished!')
print ('Congradulation!!!!!!')
###################################################################

###############
# setting for figs size and font
font = {'family' : 'normal', 'weight' : 'normal', 'size' : 42}
plt.rcParams["figure.figsize"] = [64,58]
plt.rc('font', **font)
plt.rc('xtick', labelsize=70)
plt.rc('ytick', labelsize=70)
###############
'''
# plotting the garphs
######################################
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
  save_results_to = '/home/morteza/Desktop/reports/190814 (Excitatory ver col results)/mp/'
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
  save_results_to = '/home/morteza/Desktop/reports/190814 (Excitatory ver col results)/con/'
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
  save_results_to = '/home/morteza/Desktop/reports/190814 (Excitatory ver col results)/freq/'
  #plt.savefig(save_results_to + str(i) +'.jpeg')  
  i=i+1
  
'''
'''
# plot changes of amplitude for stimulations
figure, ax3 = plt.subplots()
for rad in step_radius:
    figure, ax3 = plt.subplots()
    x = ['L1_ENGC','L1_SBC', 'L23_CC', 'L23_PV', 'L23_SST', 'L23_VIP', 'L5A_CC', 'L5A_CS', 'L5A_CT', 'L5A_PV', 'L5A_SST', 'L5B_CC', 'L5B_CS', 'L5B_PT', 'L5B_PV', 'L5B_SST','L6_CT', 'L6_PV', 'L6_SST']
    index = np.arange(len(x))
    del x[layer_dic[target_layer_name]-1:layer_dic[target_layer_name]]
    plt.rc('xtick', labelsize=10)
    #plt.plot(x, amp_changes_in_abs[rad], lw=2, marker = 'o', markersize=14, label="rad=%1.2f"%rad)
    plt.bar(x, amp_changes_in_abs[rad])  #, lw=2, marker = 'o', markersize=14, label="rad=%1.2f"%rad)
    plt.xticks(index, label, fontsize=5, rotation=30)
    plt.xlabel("Neuron types")
    plt.ylabel("relative change in firing rate")
    plt.legend(loc='upper center', ncol=4) #loc='best')
    plt.title('Response of inside neurons populations to ' + target_layer_name + ' stimulation')
    save_results_to = '/home/morteza/Desktop/reports/190827/'
    #plt.savefig(save_results_to + 'inside_cells'+'.jpeg')

figure, ax3 = plt.subplots()
for rad in step_radius:
    figure, ax3 = plt.subplots()
    x = ['L1_ENGC','L1_SBC', 'L23_CC', 'L23_PV', 'L23_SST', 'L23_VIP', 'L5A_CC', 'L5A_CS', 'L5A_CT', 'L5A_PV', 'L5A_SST', 'L5B_CC', 'L5B_CS', 'L5B_PT', 'L5B_PV', 'L5B_SST','L6_CT', 'L6_PV', 'L6_SST']
    #del x[layer_dic[target_layer_name]-1:layer_dic[target_layer_name]]
    plt.rc('xtick', labelsize=15)
    plt.plot(x, amp_changes_out_abs[rad], lw=2, marker = 'o', markersize=14, label="rad=%1.2f"%rad)#(rad,))
    plt.xlabel("Neuron types")
    plt.ylabel("relative change in firing rate")
    plt.legend(loc='upper center', ncol=4) #loc='best')
    plt.title('Response of outside neurons populations to ' + target_layer_name + ' stimulation')
    save_results_to = '/home/morteza/Desktop/reports/190827/'
    plt.savefig(save_results_to + 'outside_cells'+'.jpeg')
'''





# here I just plot some example graphs for representation
# L5B-PT
'''
figure, ax = plt.subplots()
ax.plot(bin_list, freq_in_ave['M1_L5B_PT'], color='g', linewidth=7.0)
plt.fill_between(bin_list, freq_in_ave['M1_L5B_PT'] - freq_in_sem['M1_L5B_PT'],  freq_in_ave['M1_L5B_PT'] + freq_in_sem['M1_L5B_PT'], facecolor='lightgreen', alpha= 0.4)
ax.plot(bin_list, freq_out_ave['M1_L5B_PT'],color='r', linewidth=7.0)
plt.fill_between(bin_list, freq_out_ave['M1_L5B_PT'] - freq_out_sem['M1_L5B_PT'],  freq_out_ave['M1_L5B_PT'] + freq_out_sem['M1_L5B_PT'], facecolor='lightsalmon', alpha= 0.4)
ax.plot(bin_list, freq_in_rest_ave['M1_L5B_PT'],color='b', linewidth=2.0)
plt.fill_between(bin_list, freq_in_rest_ave['M1_L5B_PT'] - freq_in_rest_sem['M1_L5B_PT'],  freq_in_rest_ave['M1_L5B_PT'] + freq_in_rest_sem['M1_L5B_PT'], facecolor='skyblue', alpha= 0.4)
ax.plot(bin_list, freq_out_rest_ave['M1_L5B_PT'],color='m', linewidth=2.0)
plt.fill_between(bin_list, freq_out_rest_ave['M1_L5B_PT'] - freq_out_rest_sem['M1_L5B_PT'],  freq_out_rest_ave['M1_L5B_PT'] + freq_out_rest_sem['M1_L5B_PT'], facecolor='plum', alpha= 0.4)
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
for line in ['left', 'bottom']:
    ax.spines[line].set_linewidth(5.0)

ax.tick_params(axis='both',length = 10.0,width = 5.0)
plt.xlabel("time (ms)")
plt.ylabel("Average frequency (Hz)")
plt.title('L5B-PT')
plt.legend(("Inside neurons", "Outside neurons", "Control neurons inside", "Control neurons outside"))  
save_results_to = '/home/morteza/Desktop/reports/190529 (Inhibitory ver col results)/'
#plt.savefig(save_results_to + str(i) +'.jpeg')



figure, ax = plt.subplots()
ax.plot(bin_list, freq_in_ave['M1_L5B_SST'], color='g', linewidth=7.0)
plt.fill_between(bin_list, freq_in_ave['M1_L5B_SST'] - freq_in_sem['M1_L5B_SST'],  freq_in_ave['M1_L5B_SST'] + freq_in_sem['M1_L5B_SST'], facecolor='lightgreen', alpha= 0.4)
ax.plot(bin_list, freq_in_rest_ave['M1_L5B_SST'], color='b', linewidth=2.0)
plt.fill_between(bin_list, freq_in_rest_ave['M1_L5B_SST'] - freq_in_rest_sem['M1_L5B_SST'],  freq_in_rest_ave['M1_L5B_SST'] + freq_in_rest_sem['M1_L5B_SST'], facecolor='skyblue', alpha= 0.4)
ax.plot(bin_list, freq_in_ave['M1_L6_SST'],color='r', linewidth=7.0)
plt.fill_between(bin_list, freq_in_ave['M1_L6_SST'] - freq_in_sem['M1_L6_SST'],  freq_in_ave['M1_L6_SST'] + freq_in_sem['M1_L6_SST'], facecolor='lightsalmon', alpha= 0.4)
ax.plot(bin_list, freq_in_rest_ave['M1_L6_SST'],color='m', linewidth=2.0)
plt.fill_between(bin_list, freq_in_rest_ave['M1_L6_SST'] - freq_in_rest_sem['M1_L6_SST'],  freq_in_rest_ave['M1_L6_SST'] + freq_in_rest_sem['M1_L6_SST'], facecolor='plum', alpha= 0.4)
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
for line in ['left', 'bottom']:
    ax.spines[line].set_linewidth(5.0)

ax.tick_params(axis='both',length = 10.0,width = 5.0)
plt.xlabel("time (ms)")
plt.ylabel("Average frequency (Hz)")
plt.title('L1-ENGC stimulated network')
plt.legend(("L5B-SST", "L5B-SST (control)", "L6-SST", "L6-SST (control)"))  
save_results_to = '/home/morteza/Desktop/reports/190529 (Inhibitory ver col results)/'
#plt.savefig(save_results_to + str(i) +'.jpeg')

'''

'''
import numpy as np
import matplotlib.pyplot as plt
bar_graph_data = {}
bar_graph_data['ENGC_inside'] = [0.0, -3.9955555555555553, -2.5784090909090911, -5.3199999999999994, 1.7920000000000016, -6.0599999999999996, -0.44000000000000011, -2.1013333333333337, -2.5013333333333385, -0.11524475524475528, 0.084571428571428686, -1.8057142857142843, -0.04454054054054056, -0.36088888888888881, 2.9111111111111114]
bar_graph_data['ENGC_outside'] = [-1.9219047619047616, -2.3624054982817868, -1.1986536248561568, -2.586031746031745, 0.19447272727272846, -3.2134603174603189, 0.1932474226804124, -0.98303030303030248, -3.8769696969696952, 0.43443269908386184, 0.20161038961038935, -3.6181818181818244, -0.043045045045045027, -0.31205714285714303, 1.3436426116838529]
bar_graph_data['SBC_inside'] = [-0.2200000000000002, 0.0 , 0.23454545454545395, -7.2359999999999998, 2.1856000000000009, -4.1039999999999992, -0.028333333333333321, -0.34666666666666757, 1.8399999999999999, -0.024615384615384595, -0.027428571428571358, 1.0537142857142854, -0.0080000000000000071, -0.040888888888888753, 0.82111111111110979]
bar_graph_data['SBC_outside'] = [-0.038253968253967763, 0.026048109965636002, 0.01453970080552347, -0.14184126984127055, 0.11600000000000499, -0.095873015873016776, -0.075515463917525505, -0.02545454545454362, 0.18230303030302863, -0.029006342494714454, -0.0058181818181819445, 0.21085714285714019, 9.0090090090033659e-05, -0.026742857142857179, -0.020103092783504195]
bar_graph_data['PV_inside'] = [1.5300000000000002, 1.1444444444444439, -2.6906818181818184, 0.0, -15.2288, -4.8399999999999999, -0.09000000000000008, -1.3066666666666666, 0.42133333333333667, 0.059860139860139938, 0.027428571428571469, 0.17828571428571394, -0.024432432432432427, -0.096000000000000085, 1.606666666666662]
bar_graph_data['PV_outside'] = [0.18857142857142861, 0.17903780068728548, -0.35205408515535108, -1.2027301587301586, -1.7259636363636321, -0.57371428571428673, 0.21737113402061814, -0.30836363636363551, -1.8986666666666672, 0.23627906976744195, 0.13662337662337642, -1.6280519480519473, -0.0087507507507508242, -0.056114285714286005, 0.24975945017182433]
bar_graph_data['SST_inside'] = [-3.23, -4.0622222222222222, -2.7065909090909095, -10.092000000000001, 0.0, -6.3039999999999994, -0.031666666666666621, -1.1626666666666665, 0.93866666666667342, 0.01734265734265733, 0.032000000000000084, 1.4125714285714324, -0.028972972972972966, -0.26488888888888884, 2.6877777777777787]
bar_graph_data['SST_outside'] = [-0.49222222222222145, -0.49003436426116842, -0.40449367088607646, -1.6111746031746019, 0.10770909090909164, -0.98304761904762117, 0.22268041237113456, -0.34254545454545315, -1.8736969696969652, 0.25317829457364316, 0.13651948051948026, -1.6536623376623396, -0.013555555555555543, -0.083771428571428608, 0.41151202749141014]
bar_graph_data['VIP_inside'] = [2.1900000000000004, 1.4311111111111119, -2.5536363636363637, 4.7760000000000016, -16.121599999999997, 0.0 , 4.4650000000000007, 5.9733333333333345, -29.071999999999999, 2.4576223776223776, 1.8879999999999995, -30.427428571428575, 0.53859459459459447, 3.1893333333333334, -14.675555555555556]
bar_graph_data['VIP_outside'] = [0.25619047619047652, 0.19560137457044746, 0.01902186421173635, 0.92539682539682389, -2.0572363636363633, -0.48863492063492231, 0.89402061855670079, 0.96145454545454534, -3.1284848484848453, 0.64776603241719477, 0.49823376623376614, -2.7547012987013026, 0.069753753753753756, 0.73811428571428594, -1.185979381443298]

# make bar graph

figure, ax = plt.subplots()
width = 0.18
x = ['L1_ENGC','L1_SBC', 'L23_CC', 'L23_PV', 'L23_SST', 'L23_VIP', 'L5A_CS', 'L5A_PV', 'L5A_SST', 'L5B_PT', 'L5B_PV', 'L5B_SST','L6_CT', 'L6_PV', 'L6_SST']
index = np.arange(len(x))
plt.rc('xtick', labelsize=10)

P1 = plt.bar(index-2*width, bar_graph_data['ENGC_inside'], width)
P2 = plt.bar( index - 1 * width, bar_graph_data ['SBC_inside'], width )
P3 = plt.bar( index , bar_graph_data ['PV_inside'], width )
P4 = plt.bar( index + 1 * width, bar_graph_data ['SST_inside'], width )
P5 = plt.bar( index + 2 * width, bar_graph_data ['VIP_inside'], width )
ax.set_xticks(index+width)
plt.xticks(index, x, fontsize=40, rotation=90)
plt.xlabel("Neuron types", fontsize = 70)
plt.ylabel("relative change in firing rate (Hz)", fontsize = 70)
plt.legend((P1[0], P2[0], P3[0], P4[0], P5[0]), ('L1_ENGC','L1_SBC','L23_PV','L23_SST','L23_VIP'), loc='lower left', ncol=3) #loc='best')
plt.title('Response of inside neurons to interneurons stimulation', fontsize = 70)

ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
for line in ['left', 'bottom']:
    ax.spines[line].set_linewidth(5.0)
ax.tick_params(axis='both',length = 10.0,width = 5.0)
save_results_to = '/home/morteza/Desktop/reports/190902/'
plt.savefig(save_results_to + 'inside'+'.jpeg')



figure, ax = plt.subplots()
width = 0.3
x = ['L1_ENGC','L1_SBC', 'L23_CC', 'L23_PV', 'L23_SST', 'L23_VIP', 'L5A_CS', 'L5A_PV', 'L5A_SST', 'L5B_PT', 'L5B_PV', 'L5B_SST','L6_CT', 'L6_PV', 'L6_SST']
index = np.arange(len(x))
plt.rc('xtick', labelsize=10)

P1 = plt.bar(index-width, bar_graph_data['ENGC_inside'], width)
P2 = plt.bar( index , bar_graph_data ['ENGC_outside'], width )

ax.set_xticks(index+width)
plt.xticks(index, x, fontsize=40, rotation=90)
plt.xlabel("Neuron types", fontsize = 70)
plt.ylabel("relative change in firing rate (Hz)", fontsize = 70)
plt.legend((P1[0], P2[0]), ('inside responses','Outside responses'), loc='lower right') #loc='best')
plt.title('Response of neurons to L1_ENGC neurons stimulation', fontsize = 70)

ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
for line in ['left', 'bottom']:
    ax.spines[line].set_linewidth(5.0)
ax.tick_params(axis='both',length = 10.0,width = 5.0)
save_results_to = '/home/morteza/Desktop/reports/190902/'
plt.savefig(save_results_to + 'inside_outside'+'.jpeg')
'''