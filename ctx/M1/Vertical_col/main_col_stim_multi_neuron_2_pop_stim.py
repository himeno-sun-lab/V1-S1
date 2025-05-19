#!/usr/bin/env python
# -*- coding: utf-8 -*-


# For this code I use one multimeter for each neuron
# and number of trials for rest state and stimulated state are equal
# some functions were added to reduce the calculations of outputs
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
def get_input_column_layers_ctx_M1(ctx_M1_layers, target_layer_name, layer_name, radius):
    circle_center = [[0.0, 0.0]]
    inside_gids = []
    outside_gids = []
    layer_gids = nest.GetNodes(ctx_M1_layers[layer_name])[0]
    neuron_positions = ntop.GetPosition(layer_gids)
    for nn in range(len(layer_gids)):
        if np.linalg.norm([(neuron_positions[nn][0] - circle_center[0][0]), (neuron_positions[nn][1] - circle_center[0][1])]) <= radius:
            inside_gids.append(layer_gids[nn])
        else:
            outside_gids.append(layer_gids[nn])

    if layer_name == target_layer_name:
        print ('all', layer_name ,'neuron numbers were: ', len(layer_gids), 'and only', len(inside_gids), 'neurons were stimulated inside the column using poisson generator....')

    return inside_gids , outside_gids,

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
  '''
  name_list = ['M1_L23_CC', 'M1_L5A_CC', 'M1_L5A_CT', 'M1_L5B_CC', 'M1_L5B_CS', 'M1_L6_CT']
  if layer_name in name_list:
      PSG = nest.Create( 'poisson_generator', 1, params={'rate': 30.0, "start": float( ini_time )} )  # , 'label': layer_name})
      print( layer_name )
      nest.Connect( pre=PSG, post=nest.GetNodes( layer_gid ) [0], syn_spec={'weight': 50.0, 'delay': 1.5} )
  elif layer_name == 'M1_L5B_PT':
      print( layer_name )
      PSG = nest.Create( 'poisson_generator', 1, params={'rate': 90.0, "start": float(ini_time )} )  # , 'label': layer_name})           # specific PSG for L5B_PT neurons
      nest.Connect( pre=PSG, post=nest.GetNodes( layer_gid ) [0], syn_spec={'weight': 50.0, 'delay': 1.5} )
  elif layer_name == 'M1_L5A_CS':
      print( layer_name )
      PSG = nest.Create( 'poisson_generator', 1, params={'rate': 70.0, "start": float(ini_time )} )  # , 'label': layer_name})           # specific PSG for L5B_PT neurons
      nest.Connect( pre=PSG, post=nest.GetNodes( layer_gid ) [0], syn_spec={'weight': 50.0, 'delay': 1.5} )
  else:
      print( layer_name )
      PSG = nest.Create( 'poisson_generator', 1,params={'rate': 45.0, "start": float( ini_time_ini )} )  # , 'label': layer_name})
      nest.Connect( pre=PSG, post=nest.GetNodes( layer_gid ) [0], syn_spec={'weight': 50.0, 'delay': 1.5} )
  '''
  if layer_name in ['M1_L23_CC']:
      PSG = nest.Create( 'poisson_generator', 1, params={'rate': 800.0, "start": float( ini_time - 40. )} )
      nest.Connect( pre=PSG, post=nest.GetNodes( layer_gid ) [0], syn_spec={'weight': 4.0, 'delay': 1.5} )
  elif layer_name in ['M1_L5A_CC']:
      PSG = nest.Create( 'poisson_generator', 1, params={'rate': 900.0, "start": float( ini_time - 20. )} )
      nest.Connect( pre=PSG, post=nest.GetNodes( layer_gid ) [0], syn_spec={'weight': 4.5, 'delay': 1.5} )
  elif layer_name in ['M1_L5A_CS']:
      PSG = nest.Create( 'poisson_generator', 1, params={'rate': 900.0, "start": float( ini_time + 30. )} )
      nest.Connect( pre=PSG, post=nest.GetNodes( layer_gid ) [0], syn_spec={'weight': 4.5, 'delay': 1.5} )
  elif layer_name in ['M1_L5A_CT']:
      PSG = nest.Create( 'poisson_generator', 1, params={'rate': 900.0, "start": float( ini_time + 10. )} )
      nest.Connect( pre=PSG, post=nest.GetNodes( layer_gid ) [0], syn_spec={'weight': 4.5, 'delay': 1.5} )
  elif layer_name in ['M1_L5B_CC']:
      PSG = nest.Create( 'poisson_generator', 1, params={'rate': 1050.0, "start": float( ini_time - 30. )} )
      nest.Connect( pre=PSG, post=nest.GetNodes( layer_gid ) [0], syn_spec={'weight': 4.5, 'delay': 1.5} )
  elif layer_name in ['M1_L5B_CS']:
      PSG = nest.Create( 'poisson_generator', 1, params={'rate': 1050.0, "start": float(ini_time + 20. )} )
      nest.Connect( pre=PSG, post=nest.GetNodes( layer_gid ) [0], syn_spec={'weight': 4.5, 'delay': 1.5} )
  elif layer_name in ['M1_L5B_PT']:
      PSG = nest.Create( 'poisson_generator', 1, params={'rate': 1050.0, "start": float( ini_time - 10. )} )
      nest.Connect( pre=PSG, post=nest.GetNodes( layer_gid ) [0], syn_spec={'weight': 4.5, 'delay': 1.5} )
  elif layer_name in ['M1_L6_CT']:
      PSG = nest.Create( 'poisson_generator', 1, params={'rate': 1050.0, "start": float( ini_time )} )
      nest.Connect( pre=PSG, post=nest.GetNodes( layer_gid ) [0], syn_spec={'weight': 4.0, 'delay': 1.5} )

  elif layer_name in ['M1_L1_SBC']:
      PSG = nest.Create( 'poisson_generator', 1, params={'rate': 1100.0, "start": float( ini_time_ini - 50. )} )
      nest.Connect( pre=PSG, post=nest.GetNodes( layer_gid ) [0], syn_spec={'weight': 4.5, 'delay': 1.5} )
  elif layer_name in ['M1_L1_ENGC']:
      PSG = nest.Create( 'poisson_generator', 1, params={'rate': 1100.0, "start": float( ini_time_ini + 30. )} )
      nest.Connect( pre=PSG, post=nest.GetNodes( layer_gid ) [0], syn_spec={'weight': 4.5, 'delay': 1.5} )
  elif layer_name in ['M1_L23_PV']:
      PSG = nest.Create( 'poisson_generator', 1, params={'rate': 1350.0, "start": float( ini_time_ini - 40. )} )
      nest.Connect( pre=PSG, post=nest.GetNodes( layer_gid ) [0], syn_spec={'weight': 4.5, 'delay': 1.5} )
  elif layer_name in ['M1_L23_SST']:
      PSG = nest.Create( 'poisson_generator', 1, params={'rate': 650.0, "start": float( ini_time_ini + 40. )} )
      nest.Connect( pre=PSG, post=nest.GetNodes( layer_gid ) [0], syn_spec={'weight': 4.5, 'delay': 1.5} )
  elif layer_name in ['M1_L23_VIP']:
      PSG = nest.Create( 'poisson_generator', 1, params={'rate': 1400.0, "start": float( ini_time_ini + 20. )} )
      nest.Connect( pre=PSG, post=nest.GetNodes( layer_gid ) [0], syn_spec={'weight': 4.5, 'delay': 1.5} )
  elif layer_name in ['M1_L5A_PV']:
      PSG = nest.Create( 'poisson_generator', 1, params={'rate': 1300.0, "start": float( ini_time_ini - 30. )} )
      nest.Connect( pre=PSG, post=nest.GetNodes( layer_gid ) [0], syn_spec={'weight': 4.5, 'delay': 1.5} )
  elif layer_name in ['M1_L5A_SST']:
      PSG = nest.Create( 'poisson_generator', 1, params={'rate': 650.0, "start": float( ini_time_ini + 10. )} )
      nest.Connect( pre=PSG, post=nest.GetNodes( layer_gid ) [0], syn_spec={'weight': 4.5, 'delay': 1.5} )
  elif layer_name in ['M1_L5B_PV']:
      PSG = nest.Create( 'poisson_generator', 1, params={'rate': 1300.0, "start": float( ini_time_ini - 20. )} )
      nest.Connect( pre=PSG, post=nest.GetNodes( layer_gid ) [0], syn_spec={'weight': 4.5, 'delay': 1.5} )
  elif layer_name in ['M1_L5B_SST']:
      PSG = nest.Create( 'poisson_generator', 1, params={'rate': 700.0, "start": float( ini_time_ini + 50. )} )
      nest.Connect( pre=PSG, post=nest.GetNodes( layer_gid ) [0], syn_spec={'weight': 4.5, 'delay': 1.5} )
  elif layer_name in ['M1_L6_PV']:
      PSG = nest.Create( 'poisson_generator', 1, params={'rate': 1350.0, "start": float( ini_time_ini - 10. )} )
      nest.Connect( pre=PSG, post=nest.GetNodes( layer_gid ) [0], syn_spec={'weight': 4.5, 'delay': 1.5} )
  elif layer_name in ['M1_L6_SST']:
      PSG = nest.Create( 'poisson_generator', 1, params={'rate': 700.0, "start": float( ini_time_ini )} )
      nest.Connect( pre=PSG, post=nest.GetNodes( layer_gid ) [0], syn_spec={'weight': 4.5, 'delay': 1.5} )

#################################################################
def layer_poisson_generator(layer_gid, layer_name, params={"withgid": True, "withtime": True, "to_file": False}):
    #def layer_spike_detector(layer_gid, layer_name, params={"withgid": True, "withtime": True, "to_file": True, 'fbuffer_size': 8192}):
    params.update({'label': layer_name})

    # stimulation by using PSG
    PSG_arrow = nest.Create('poisson_generator', 1, params={'rate':  300.0}) #, 'label': layer_name})
    nest.Connect(pre=PSG_arrow, post=layer_gid, syn_spec={'weight': 10.0, 'delay': 1.5})

    return PSG_arrow

###############################################################
def layer_spike_detector(layer_gids_inside, layer_gids_outside, layer_name, record_onset, ignore_time, ran_inside_neurons, ran_outside_neurons, params = {"withgid": True, "withtime": True, "to_file": False}):
    print( 'spike detector for ' + layer_name )
    params.update( {'label': layer_name, "start": float(record_onset)} )

    # Add detector for all neuron types
    detector_in = nest.Create( "spike_detector", params=params )
    nest.Connect( pre=layer_gids_inside, post=detector_in )

    detector_out = nest.Create( "spike_detector", params=params )
    nest.Connect( pre=layer_gids_outside, post=detector_out)

    # generating some random numbers to choose random neuron to record
    inside_random = np.random.choice(len(layer_gids_inside), int(ran_inside_neurons), replace=False)
    outside_random = np.random.choice(len(layer_gids_outside), int(ran_outside_neurons), replace=False)

    nest_mm_in = {}
    nest_mm_out = {}
    # add multimeter just for to record V_m and conductance (inhibitory and excitatory) of a single cell each cell population
    for num in range( int( ran_inside_neurons ) ):
        nest_mm_in[num] = nest.Create( 'multimeter', params={"interval": 0.1, "record_from": ["V_m", "g_ex", "g_in"], "withgid": True, "to_file": False, 'label': layer_name, 'withtime': True, "start": float(record_onset), 'use_gid_in_filename': True} )
        nest.Connect(nest_mm_in[num], [layer_gids_inside[inside_random[num]]])

    for num in range( int( ran_outside_neurons ) ):
        nest_mm_out[num] = nest.Create( 'multimeter', params={"interval": 0.1, "record_from": ["V_m", "g_ex", "g_in"], "withgid": True, "to_file": False, 'label': layer_name, 'withtime': True, "start": float(record_onset), 'use_gid_in_filename': True} )  #
        nest.Connect(nest_mm_out[num], [layer_gids_outside[outside_random[num]]])

    voltmeter_single = nest.Create( "voltmeter", params={'to_file': False, 'label': layer_name, "start": float( ignore_time ), "withgid": True, "withtime": True} )  # 'use_gid_in_filename': True,   "start": float(ignore_time),
    nest.Connect(voltmeter_single, [layer_gids_inside[2]] )

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

########################################################################################################################
# This is the program to assume one column in center of simulated area and then stimulate the inside neurons of specific
# cell population and then record from:
# 1- Membrane potential on one cell inside and one cell outside of the col of all cell populations and plot it
# 2- Inhibitory & Excitatory conductance on one (easy to add to 2 or more cells) cell inside and one cell outside of the
# col of all cell populations and plot it
# 3- Firing rate of all cell populations inside and outside as a avarage of all trials and all cells including sem among
# the number of trials (not among the cells)
####################################################
# Program initiation
#######################################################
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
M1_layer_size = ctx_M1_params ['M1'] ['structure_info'] ['region_size']
M1_layer_Name = ctx_M1_params ['M1'] ['structure_info'] ['Layer_Name']
M1_layer_thickness = ctx_M1_params ['M1'] ['structure_info'] ['layer_thickness']
M1_layer_thickness_ST=np.cumsum(M1_layer_thickness)
M1_Layer_thickness_ST=np.insert(M1_layer_thickness_ST, 0, 0.0)
M1_layer_depth = ctx_M1_params ['M1'] ['structure_info'] ['layer_depth']

###########################
# 4) making layers and doing all connections
print('create and connect the M1 layers')
start_time = time.time()
ctx_M1_layers, SubSubRegion_Excitatory, SubSubRegion_Inhibitory =instantiate_ctx_M1(ctx_M1_params, sim_params['scalefactor'])
with open('./log/' + 'x_performance.txt', 'a') as file:
  file.write('M1_Construction_Time: ' + str(time.time() - start_time) + '\n')


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
radius = 0.16
###############################
# 7) selecting the VIP neurons for the stimulation located inside the coloumn
poisson_gens = []    # in case of one trial     poisson_gens = {}
target_layer_name = {'1': 'M1_L23_CC','2': 'M1_L23_SST'}
#target_layer_name = 'M1_L23_CC'
target_gids_inside, target_gids_outside = get_input_column_layers_ctx_M1(ctx_M1_layers, target_layer_name['1'], target_layer_name['1'],  radius)
#poisson_gens[layer_name]= layer_poisson_generator(target_gids_inside, layer_name, stim_onset, stim_offset, sim_params['initial_ignore'])   # this was good for one trial
poisson_gens.append(layer_poisson_generator(target_gids_inside, target_layer_name['1']))

target_gids_inside, target_gids_outside = get_input_column_layers_ctx_M1(ctx_M1_layers, target_layer_name['2'], target_layer_name['2'],  radius)
poisson_gens.append(layer_poisson_generator(target_gids_inside, target_layer_name['2']))

###############################
# 8) recording from other neurons in circle and out side of circle
inside_neurons_torecord =5
outside_neurons_torecord = 5
multimeters_in = {}
multimeters_out = {}
voltmeters = {}
detectors_in = {}
detectors_out = {}
layer_gids_inside = {}
layer_gids_outside = {}
layer_gids_all = {}
for layer_name in ctx_M1_layers.keys():
    layer_gids_inside[layer_name], layer_gids_outside[layer_name] = get_input_column_layers_ctx_M1(ctx_M1_layers, target_layer_name['1'], layer_name, radius)
    detectors_in[layer_name], detectors_out[layer_name], multimeters_in[layer_name], multimeters_out[layer_name], voltmeters[layer_name] = layer_spike_detector(layer_gids_inside[layer_name], layer_gids_outside[layer_name],
                                                                                                                                                               layer_name, record_onset, sim_params['initial_ignore'],
                                                                                                                                                                inside_neurons_torecord, outside_neurons_torecord)
    layer_gids_all [layer_name] = layer_gids_inside [layer_name] + layer_gids_outside [layer_name]

################################
# 9) simulation
num_trials = 2
simulation_time = sim_params['simDuration']
initial_ignore = sim_params['initial_ignore']


print ('Simulation Started:')
start_time=time.time()
for n in range(num_trials):
    if n == 0:
        #nest.SetStatus( poisson_gens [0], {'origin': nest.GetKernelStatus() ['time'], "start": 1000.0, 'stop': 1500.0} )
        #nest.Simulate(simulation_time + initial_ignore)
        print(n)
        nest.SetStatus( poisson_gens [0], {'origin': nest.GetKernelStatus() ['time'], "start": (num_trials*simulation_time) + (2*record_intervals), 'stop': (num_trials*simulation_time)+(3*record_intervals)} )
        nest.SetStatus( poisson_gens [1], {'origin': nest.GetKernelStatus() ['time'], "start": (num_trials * simulation_time) + (2 * record_intervals), 'stop': (num_trials * simulation_time) + (3 * record_intervals)} )
        nest.Simulate( (num_trials*simulation_time) + initial_ignore + simulation_time )
    elif n == num_trials -1:
        print(n)
        nest.SetStatus( poisson_gens [0], {'origin': nest.GetKernelStatus() ['time'], "start": record_intervals, 'stop': 2*record_intervals} )
        nest.SetStatus( poisson_gens [1], {'origin': nest.GetKernelStatus() ['time'], "start": record_intervals, 'stop': 2 * record_intervals} )
        nest.Simulate( simulation_time+ 1.0 )
    else:
        print(n)
        nest.SetStatus( poisson_gens [0], {'origin': nest.GetKernelStatus() ['time'], "start": record_intervals, 'stop': 2*record_intervals} )
        nest.SetStatus( poisson_gens [1], {'origin': nest.GetKernelStatus() ['time'], "start": record_intervals, 'stop': 2 * record_intervals} )
        nest.Simulate(simulation_time)

with open('./log/'+'x_performance.txt', 'a') as file:
  file.write('Simulation_Elapse_Time: '+str(time.time()-start_time)+'\n')




############################################################################
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
    for num in range( inside_neurons_torecord ):
        events_in [num] = nest.GetStatus( multimeters_in [layer_name] [num] ) [0] ["events"]

    for num in range( outside_neurons_torecord ):
        events_out [num] = nest.GetStatus( multimeters_out [layer_name] [num] ) [0] ["events"]

    spike_t_in[layer_name] = nest.GetStatus( detectors_in [layer_name] ) [0] ['events'] ['times']    # recorded from all neurons inside
    spike_t_out[layer_name] = nest.GetStatus( detectors_out [layer_name] ) [0] ['events'] ['times']  # recorded from all neurons outside
    spike_t_all[layer_name] = np.asarray( ((spike_t_in[layer_name]).tolist()) + ((spike_t_out[layer_name]).tolist()))

    if inside_neurons_torecord > 1:
        g_ins_in[layer_name], g_exs_in[layer_name], mps_in[layer_name]= average_recorders_outputs(events_in, inside_neurons_torecord)  # average among the recorded neurons inside
    else:
        g_ins_in[layer_name] = events_in [0] ['g_in']
        g_exs_in[layer_name] = events_in [0] ['g_ex']
        mps_in[layer_name] = events_in [0] ['V_m']

    if outside_neurons_torecord > 1:
        g_ins_out[layer_name], g_exs_out[layer_name], mps_out[layer_name]= average_recorders_outputs(events_out, outside_neurons_torecord) # average among the recorded neurons outside
    else:
        g_ins_out[layer_name] = events_out [0] ['g_in']
        g_exs_out[layer_name] = events_out [0] ['g_ex']
        mps_out[layer_name] = events_out [0] ['V_m']

#-----------------------------------------------------------------------------------------------------------------------
tt = events_in[0] ['times']
times = ((events_in[0]['times'])[0:15000])
mps_in_ave = {}
mps_out_ave = {}
g_ins_in_ave = {}
g_ins_out_ave = {}
g_exs_in_ave = {}
g_exs_out_ave = {}
for layer_name in ctx_M1_layers.keys():
    mps_in_ave [layer_name] = average_over_trials(num_trials, mps_in[layer_name])
    mps_out_ave [layer_name] = average_over_trials(num_trials, mps_out[layer_name])
    g_ins_in_ave [layer_name] = average_over_trials(num_trials, g_ins_in[layer_name])
    g_ins_out_ave [layer_name] = average_over_trials(num_trials, g_ins_out[layer_name])
    g_exs_in_ave [layer_name] = average_over_trials(num_trials, g_exs_in[layer_name])
    g_exs_out_ave [layer_name] = average_over_trials(num_trials, g_exs_out[layer_name])

#-----------------------------------------------------------------------------------------------------------------------
freq_in = {}
freq_out = {}
freq_rest = {}
freq_in_rest = {}
freq_out_rest = {}

bin = 20.0
num_windows = simulation_time/bin
bin_list = [None] * int(num_windows+1)
for nn in range(int(num_windows+1)):
    bin_list[nn] = nn * 20

for layer_name in ctx_M1_layers.keys():
    print( layer_name )
    print( 'number of spikes in are:', len( spike_t_in[layer_name] ) )
    print( 'number of spikes out are:', len( spike_t_out[layer_name] ) )
    freq_in[layer_name] = spike_sort_count_freq(num_trials, spike_t_in[layer_name], sim_params['initial_ignore'], sim_params['simDuration'], layer_gids_inside[layer_name], bin_list)[0]
    freq_out[layer_name] = spike_sort_count_freq(num_trials, spike_t_out[layer_name], sim_params ['initial_ignore'], sim_params ['simDuration'], layer_gids_outside[layer_name],bin_list)[0]
    freq_in_rest [layer_name] = spike_sort_count_freq(num_trials, spike_t_in[layer_name], sim_params ['initial_ignore'], sim_params ['simDuration'], layer_gids_inside[layer_name], bin_list)[1]
    freq_out_rest [layer_name] = spike_sort_count_freq(num_trials, spike_t_out[layer_name], sim_params ['initial_ignore'], sim_params ['simDuration'], layer_gids_outside[layer_name], bin_list)[1]

#######################################################################################################################
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
amp_changes_in_abs = []
amp_changes_out_abs = []
amp_changes_in_abs_norm = []
amp_changes_out_abs_norm = []


for layer_name in ctx_M1_layers.keys():
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

    amp_changes_in [layer_name]= np.average( freq_in_ave [layer_name] [25:50] ) - np.average(freq_in_rest_ave [layer_name] [25:50] )
    amp_changes_in_abs.append(amp_changes_in [layer_name])

    amp_changes_out [layer_name] = np.average( freq_out_ave [layer_name] [25:50] ) - np.average(freq_out_rest_ave [layer_name] [25:50] )
    amp_changes_out_abs.append(amp_changes_out [layer_name])

amp_changes_in_abs_norm = [x * 100 / amp_changes_in [target_layer_name['1']] for x in amp_changes_in_abs]
amp_changes_out_abs_norm= [x * 100 / amp_changes_out [target_layer_name['1']] for x in amp_changes_out_abs]

del amp_changes_in_abs_norm[layer_dic[target_layer_name['1']]-1:layer_dic[target_layer_name['1']]]
del amp_changes_out_abs_norm[layer_dic[target_layer_name['1']]-1:layer_dic[target_layer_name['1']]]
print ('Simulation successfully finished!')
print ('Congradulation!!!!!!')
###################################################################

###############
# setting for figs size and font
font = {'family' : 'normal', 'weight' : 'normal', 'size' : 42}
plt.rcParams["figure.figsize"] = [64,48]
plt.rc('font', **font)
plt.rc('xtick', labelsize=70)
plt.rc('ytick', labelsize=70)
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
  
  save_results_to = '/home/morteza/Desktop/reports/191029 (Exc ver col)/'
  #plt.savefig(save_results_to + str(i) +'.jpeg')
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
  save_results_to = '/home/morteza/Desktop/reports/191029 (Exc ver col)/'
  #plt.savefig(save_results_to + str(i) +'.jpeg')
  i=i+1

######################################
# firing frequency 
del bin_list[-1]
i=1
for layer_name in ctx_M1_layers.keys():
  figure, ax = plt.subplots()
  #ax.plot(bin_list, freq_in_ave[layer_name], color='g', linewidth=7.0)
  #plt.fill_between(bin_list, freq_in_ave[layer_name] - freq_in_sem[layer_name],  freq_in_ave[layer_name] + freq_in_sem[layer_name], facecolor='lightgreen', alpha= 0.4)
  ax.plot(bin_list, freq_out_ave[layer_name],color='r', linewidth=7.0)
  plt.fill_between(bin_list, freq_out_ave[layer_name] - freq_out_sem[layer_name],  freq_out_ave[layer_name] + freq_out_sem[layer_name], facecolor='lightsalmon', alpha= 0.4)
  #ax.plot(bin_list, freq_in_rest_ave[layer_name],color='b', linewidth=5.0)
  #plt.fill_between(bin_list, freq_in_rest_ave[layer_name] - freq_in_rest_sem[layer_name],  freq_in_rest_ave[layer_name] + freq_in_rest_sem[layer_name], facecolor='skyblue', alpha= 0.4)
  ax.plot(bin_list, freq_out_rest_ave[layer_name],color='m', linewidth=5.0)
  plt.fill_between(bin_list, freq_out_rest_ave[layer_name] - freq_out_rest_sem[layer_name],  freq_out_rest_ave[layer_name] + freq_out_rest_sem[layer_name], facecolor='plum', alpha= 0.4)
  ax.spines['right'].set_visible(False)
  ax.spines['top'].set_visible(False)
  for line in ['left', 'bottom']:
    ax.spines[line].set_linewidth(5.0)
  
  ax.tick_params(axis='both',length = 10.0,width = 5.0)
  plt.xlabel("time (ms)", fontsize = 70)
  plt.ylabel("Average frequency (Hz)", fontsize = 70)
  plt.title(layer_name, fontsize = 70)
  #plt.legend(("Inside_col", "Outside_col", "Resting activity inside", "Resting activity outside"))
  #plt.legend(("Inside_col", "Resting activity inside"))
  plt.legend(("Outside_col", "Resting activity Outside"))   
  save_results_to = '/home/morteza/Desktop/reports/191029 (Exc ver col)/'
  #plt.savefig(save_results_to + str(i) + '_inside' +'.jpeg')
  plt.savefig(save_results_to + str(i) + '_outside' +'.jpeg')  
  i=i+1
  

########################################################################################################################
# try to make graphs of all rasters at once
# graph for the selective raster plot of all pop in one graph #
del bin_list[-1]
plt.rcParams ["figure.figsize"] = [8, 64]
plt.rc('xtick', labelsize=20)
i = 1
for layer_name in ctx_M1_layers.keys():
    ax = plt.subplot( 19, 1, i )
    ax.tick_params( axis='both', length=5.0, width=2.0 )
    if i == 1:
        plt.title(f'response to {target_layer_name} stimulation inside col\n', fontsize=15)
    if i == 19:
        plt.xlabel( "time (ms)", fontsize=15 )
    else:
        plt.xticks( [] )

    plt.ylabel( layer_name, fontsize=13, rotation=20, labelpad=40 )
    plt.yticks( [] )
    plt.xlim( 0.0, 1500. )

    ax.spines ['left'].set_visible( False )
    ax.spines ['right'].set_visible( False )

    ax.plot(bin_list, freq_in_ave[layer_name], color='g', linewidth=3.0)
    plt.fill_between(bin_list, freq_in_ave[layer_name] - freq_in_sem[layer_name],  freq_in_ave[layer_name] + freq_in_sem[layer_name], facecolor='lightgreen', alpha= 0.4)
    ax.plot(bin_list, freq_in_rest_ave[layer_name],color='b', linewidth=3.0)
    plt.fill_between(bin_list, freq_in_rest_ave[layer_name] - freq_in_rest_sem[layer_name],  freq_in_rest_ave[layer_name] + freq_in_rest_sem[layer_name], facecolor='skyblue', alpha= 0.4)
    i += 1

save_results_to = '/home/morteza/Desktop/reports/191031 (final M1 tuning)/'
plt.savefig( save_results_to + target_layer_name['1'] + ' and '+ target_layer_name['2'] + 'stim inside' + '.jpeg', bbox_inches='tight' )

i = 1
for layer_name in ctx_M1_layers.keys():
    ax = plt.subplot( 19, 1, i )
    ax.tick_params( axis='both', length=5.0, width=2.0 )
    if i == 1:
        plt.title(f'response to {target_layer_name} stimulation outside col\n', fontsize=15)
    if i == 19:
        plt.xlabel( "time (ms)", fontsize=30 )
    else:
        plt.xticks( [] )

    plt.ylabel( layer_name, fontsize=13, rotation=20, labelpad=40 )
    plt.yticks( [] )
    plt.xlim( 0.0, 1500. )

    ax.spines ['left'].set_visible( False )
    ax.spines ['right'].set_visible( False )

    ax.plot(bin_list, freq_out_ave[layer_name],color='r', linewidth=3.0)
    plt.fill_between(bin_list, freq_out_ave[layer_name] - freq_out_sem[layer_name],  freq_out_ave[layer_name] + freq_out_sem[layer_name], facecolor='lightsalmon', alpha= 0.4)
    ax.plot(bin_list, freq_out_rest_ave[layer_name],color='m', linewidth=3.0)
    plt.fill_between(bin_list, freq_out_rest_ave[layer_name] - freq_out_rest_sem[layer_name],  freq_out_rest_ave[layer_name] + freq_out_rest_sem[layer_name], facecolor='plum', alpha= 0.4)
    i += 1

save_results_to = '/home/morteza/Desktop/reports/191031 (final M1 tuning)/'
plt.savefig( save_results_to + target_layer_name['1']+ ' and '+ target_layer_name['2'] + ' stim outside' + '.jpeg', bbox_inches='tight' )


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
save_results_to = '/home/morteza/Desktop/reports/191029 (Exc ver col)/'
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
save_results_to = '/home/morteza/Desktop/reports/191029 (Exc ver col)/'
#plt.savefig(save_results_to + str(i) +'.jpeg')
'''

'''
# plot changes of amplitude for stimulations
figure, ax3 = plt.subplots()
x = ['L1_ENGC','L1_SBC', 'L23_CC', 'L23_PV', 'L23_SST', 'L23_VIP', 'L5A_CC', 'L5A_CS', 'L5A_CT', 'L5A_PV', 'L5A_SST', 'L5B_CC', 'L5B_CS', 'L5B_PT', 'L5B_PV', 'L5B_SST','L6_CT', 'L6_PV', 'L6_SST']
#del x[layer_dic[target_layer_name]-1:layer_dic[target_layer_name]]
plt.rc('xtick', labelsize=12)
plt.plot(x, amp_changes_in_abs, lw=2, marker = 'o', markersize=14)#(rad,))
plt.xlabel("Neuron types")
plt.ylabel("relative change in firing rate")
plt.legend(loc='best') #loc='best')
plt.title('Response of inside neurons populations to ' + target_layer_name['1']+ ' and '+ target_layer_name['2'] + ' stimulation')
save_results_to = '/home/morteza/Desktop/reports/191029 (Exc ver col)/'
#plt.savefig(save_results_to + 'inside_cells'+'.jpeg')

figure, ax3 = plt.subplots()
x = ['L1_ENGC','L1_SBC', 'L23_CC', 'L23_PV', 'L23_SST', 'L23_VIP', 'L5A_CC', 'L5A_CS', 'L5A_CT', 'L5A_PV', 'L5A_SST', 'L5B_CC', 'L5B_CS', 'L5B_PT', 'L5B_PV', 'L5B_SST','L6_CT', 'L6_PV', 'L6_SST']
#del x[layer_dic[target_layer_name]-1:layer_dic[target_layer_name]]
plt.rc('xtick', labelsize=15)
plt.plot(x, amp_changes_out_abs, lw=2, marker = 'o', markersize=14)#(rad,))
plt.xlabel("Neuron types")
plt.ylabel("relative change in firing rate")
plt.legend(loc='upper center') #loc='best')
plt.title('Response of outside neurons populations to ' + target_layer_name['1']+ ' and '+ target_layer_name['2'] + ' stimulation')
save_results_to = '/home/morteza/Desktop/reports/191029 (Exc ver col)/'
#plt.savefig(save_results_to + 'outside_cells'+'.jpeg')
'''
'''
f = open( "/home/morteza/Desktop/reports/190902/L23_SST_inside.txt", "w" )
i=0
for layer_name in ctx_M1_layers.keys():
    f.write(', %f' %amp_changes_in_abs[i])
    i=i+1

f.close()
f = open( "/home/morteza/Desktop/reports/190902/L23_SST_outside.txt", "w" )
i=0
for layer_name in ctx_M1_layers.keys():
    f.write(', %f' %amp_changes_out_abs[i])
    i=i+1

f.close()
'''

'''
# import xlsxwriter module 
import xlsxwriter 

workbook = xlsxwriter.Workbook('/home/morteza/Desktop/reports/190902/L1_ENGC_outside.xlsx') 

# By default worksheet names in the spreadsheet will be  
# Sheet1, Sheet2 etc., but we can also specify a name. 
worksheet = workbook.add_worksheet("My sheet") 

# Some data we want to write to the worksheet. 
# Start from the first cell. Rows and 
# columns are zero indexed. 
row = 0
col = 0

# Iterate over the data and write it out row by row.
i=0
for layer_name in ctx_M1_layers.keys():
    worksheet.write(row, col, amp_changes_out_abs[i])
    i=i+1
    col += 1
     
workbook.close()
'''