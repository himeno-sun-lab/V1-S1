#!/usr/bin/env python
# -*- coding: utf-8 -*-

# This is the file which provides the output from 2 cell type of L5A_CS * L5B_PT neurons for Basal Ganglia
# It is using the main_col_stim_multi_neuron.py as a basic file to be developed
# no need to use trial state so it may use the raster report of ' ctx_primaryM1_multi_PSG.py"


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
import nest.raster_plot
import nest.voltage_trace
import matplotlib.pyplot as plt
import pylab
import numpy as np

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
def hex_corner(center,size,i, scale):
    angle_deg = 60 * i - 30
    angle_rad = np.pi / 180 * angle_deg

    return [(center[0] + size * np.cos(angle_rad))*scale[0],(center[1] + size * np.sin(angle_rad))*scale[1]]
###############################################################
def get_channel_centers(hex_center,ci,hex_radius, scale):
    center_aux = []
    #if bg_params['channels']:
        #if len(bg_params['circle_center'])==0: #must be done before bg instantiation.
    for i in np.arange(ci):
        x_y = hex_corner(hex_center,hex_radius,i, scale) #center, radius, vertex id # gives x,y of an hexagon vertexs.
        center_aux.append(x_y)
                            #bg_params['circle_center'].append(x_y)
        #np.savetxt('./log/centers.txt',center_aux) #save the centers.
        #print('generated centers: ',center_aux)

    return center_aux
###############################################################
def get_input_column_layers_ctx_M1(ctx_M1_layers, layer_name, radius, circles_center, num_of_col):

    inside_gids = {}  # number of columns
    outside_gids = {}
    for i in range(num_of_col):
        inside_gids[i]=[]
        outside_gids[i]=[]

    layer_gids = nest.GetNodes(ctx_M1_layers[layer_name])[0]
    neuron_positions = ntop.GetPosition(layer_gids)
    for nn in range(len(layer_gids)):
        if np.linalg.norm([(neuron_positions[nn][0] - circles_center[0][0]), (neuron_positions[nn][1] - circles_center[0][1])]) <= radius:
            inside_gids[0].append(layer_gids[nn])
        else:
            outside_gids[0].append(layer_gids[nn])

        if np.linalg.norm([(neuron_positions[nn][0] - circles_center[1][0]), (neuron_positions[nn][1] - circles_center[1][1])]) <= radius:
            inside_gids[1].append(layer_gids[nn])
        else:
            outside_gids[1].append(layer_gids[nn])

        if np.linalg.norm([(neuron_positions[nn][0] - circles_center[2][0]), (neuron_positions[nn][1] - circles_center[2][1])]) <= radius:
            inside_gids[2].append(layer_gids[nn])
        else:
            outside_gids[2].append(layer_gids[nn])

        if np.linalg.norm([(neuron_positions[nn][0] - circles_center[3][0]), (neuron_positions[nn][1] - circles_center[3][1])]) <= radius:
            inside_gids[3].append(layer_gids[nn])
        else:
            outside_gids[3].append(layer_gids[nn])

        if np.linalg.norm([(neuron_positions[nn][0] - circles_center[4][0]), (neuron_positions[nn][1] - circles_center[4][1])]) <= radius:
            inside_gids[4].append(layer_gids[nn])
        else:
            outside_gids[4].append(layer_gids[nn])

        if np.linalg.norm([(neuron_positions[nn][0] - circles_center[5][0]), (neuron_positions[nn][1] - circles_center[5][1])]) <= radius:
            inside_gids[5].append(layer_gids[nn])
        else:
            outside_gids[5].append(layer_gids[nn])

    return inside_gids , outside_gids
###############################################################
def get_output_column_layers_ctx(ctx_M1_layers,radius,pre_area):
    if pre_area == 'M1':
        gid_L5B = nest.GetNodes(ctx_M1_layers['M1_L5B_PT'])[0]
        gid_L5A = nest.GetNodes(ctx_M1_layers['M1_L5A_CS'])[0]
        pos_L5B = ntop.GetPosition(gid_L5B)
        pos_L5A = ntop.GetPosition(gid_L5A)

        #neuron_positions_M1_L5B_PT = np.load('ctx/Neuron_pos_' + 'M1_L5B_PT' + '.npz') #ntop.GetPosition(gid_M1_L5B_PT)
        #neuron_positions_M1_L5A_CS = np.load('ctx/Neuron_pos_' + 'M1_L5A_CS' + '.npz') #ntop.GetPosition(gid_M1_L5A_CS)
        #pos_L5B = neuron_positions_M1_L5B_PT['Neuron_pos']
        #pos_L5A = neuron_positions_M1_L5A_CS['Neuron_pos']
    print('gids and pos L5A ', len(gid_L5A), len(pos_L5A))
    print('gids and pos L5B ', len(gid_L5B), len(pos_L5B))

    circle_center = [[0.3, 0.0], [0.15,0.3], [-0.15,0.3], [-0.3,0.0], [-0.15, -0.3], [0.15, -0.3]]


    circle_1_gids, circle_2_gids, circle_3_gids, circle_4_gids, circle_5_gids, circle_6_gids = {},{},{},{},{},{}  # number of circles depend on centers
    circle_1_gids['5A'], circle_2_gids['5A'], circle_3_gids['5A'], circle_4_gids['5A'], circle_5_gids['5A'], circle_6_gids['5A'] = [], [], [], [], [],[]
    circle_1_gids ['5B'], circle_2_gids ['5B'], circle_3_gids ['5B'], circle_4_gids ['5B'], circle_5_gids ['5B'], circle_6_gids ['5B'] = [], [], [], [], [], []
    for nn in range(len(gid_L5A)):
        if np.linalg.norm([(pos_L5A[nn][0] - circle_center[0][0]), (pos_L5A[nn][1] - circle_center[0][1])]) <= radius:
            circle_1_gids['5A'].append(gid_L5A[nn])
        if np.linalg.norm([(pos_L5A[nn][0] - circle_center[1][0]), (pos_L5A[nn][1] - circle_center[1][1])]) <= radius:
            circle_2_gids['5A'].append(gid_L5A[nn])
        if np.linalg.norm([(pos_L5A[nn][0] - circle_center[2][0]), (pos_L5A[nn][1] - circle_center[2][1])]) <= radius:
            circle_3_gids['5A'].append(gid_L5A[nn])
        if np.linalg.norm([(pos_L5A[nn][0] - circle_center[3][0]), (pos_L5A[nn][1] - circle_center[3][1])]) <= radius:
            circle_4_gids['5A'].append(gid_L5A[nn])
        if np.linalg.norm([(pos_L5A[nn][0] - circle_center[4][0]), (pos_L5A[nn][1] - circle_center[4][1])]) <= radius:
            circle_5_gids['5A'].append(gid_L5A[nn])
        if np.linalg.norm([(pos_L5A[nn][0] - circle_center[5][0]), (pos_L5A[nn][1] - circle_center[5][1])]) <= radius:
            circle_6_gids['5A'].append(gid_L5A[nn])

    for nn in range(len(gid_L5B)):
        if np.linalg.norm([(pos_L5B[nn][0] - circle_center[0][0]), (pos_L5B[nn][1] - circle_center[0][1])]) <= radius:
            circle_1_gids['5B'].append(gid_L5B[nn])
        if np.linalg.norm([(pos_L5B[nn][0] - circle_center[1][0]), (pos_L5B[nn][1] - circle_center[1][1])]) <= radius:
            circle_2_gids['5B'].append(gid_L5B[nn])
        if np.linalg.norm([(pos_L5B[nn][0] - circle_center[2][0]), (pos_L5B[nn][1] - circle_center[2][1])]) <= radius:
            circle_3_gids['5B'].append(gid_L5B[nn])
        if np.linalg.norm([(pos_L5B[nn][0] - circle_center[3][0]), (pos_L5B[nn][1] - circle_center[3][1])]) <= radius:
            circle_4_gids['5B'].append(gid_L5B[nn])
        if np.linalg.norm([(pos_L5B[nn][0] - circle_center[4][0]), (pos_L5B[nn][1] - circle_center[4][1])]) <= radius:
            circle_5_gids['5B'].append(gid_L5B[nn])
        if np.linalg.norm([(pos_L5B[nn][0] - circle_center[5][0]), (pos_L5B[nn][1] - circle_center[5][1])]) <= radius:
            circle_6_gids['5B'].append(gid_L5B[nn])

    circle_gids=[circle_1_gids, circle_2_gids, circle_3_gids, circle_4_gids, circle_5_gids, circle_6_gids]
    return circle_gids

###############################################################
def instantiate_ctx_M1(ctx_M1_params, scalefactor):
    region_name = 'M1'
    index = 0
    pos_inh = np.zeros( (0, 4) )
    M1_layer_Name = ctx_M1_params ['M1'] ['structure_info'] ['Layer_Name']
    M1_layer_size = ctx_M1_params ['M1'] ['structure_info'] ['region_size']
    # M1_layer_size = np.array( M1_layer_size )         may this is used later
    M1_layer_thickness = ctx_M1_params ['M1'] ['structure_info'] ['layer_thickness']
    M1_layer_thickness_ST = np.cumsum( M1_layer_thickness)
    M1_layer_thickness_ST = np.insert( M1_layer_thickness_ST, 0, 0.0 )
    M1_layer_depth = ctx_M1_params ['M1'] ['structure_info'] ['layer_depth']
    topo_extend = [M1_layer_size [0] * int( scalefactor [0] ) + 1., M1_layer_size [1] * int( scalefactor [1] ) + 1., 2.]
    topo_extend = np.array(topo_extend)
    topo_center = np.array( [0.0, 0.0, 0.0])
    SubSubRegion_Excitatory = []
    SubSubRegion_Inhibitory = []
    ctx_M1_layers = {}
    for l in range( len( M1_layer_Name)):
        print( '###########################################' )
        print( 'start to create layer: ' + M1_layer_Name [l] )
        topo_center [2] = M1_layer_depth [l] + 0.5 * M1_layer_thickness [l]
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

    #M1_internal_connection = np.load(ctx_M1_params ['M1'] ['connection_info'] ['M1toM1'] )
    ctx_M1_internal_connection = np.load( ctx_M1_params ['M1'] ['connection_info'] ['M1toM1'] )
    for pre_layer_name in ctx_M1_layers.keys():
        for post_layer_name in ctx_M1_layers.keys():
            print( 'start to connect ' + pre_layer_name + ' with ' + post_layer_name )
            connect_layers_ctx_M1( ctx_M1_layers [pre_layer_name], ctx_M1_layers [post_layer_name], ctx_M1_internal_connection [pre_layer_name] [post_layer_name] )

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
      print( layer_name)
      nest.Connect( pre=PSG, post=nest.GetNodes( layer_gid ) [0], syn_spec={'weight': 50.0, 'delay': 1.5} )
  elif layer_name == 'M1_L5B_PT':
      print( layer_name )
      PSG = nest.Create( 'poisson_generator', 1, params={'rate': 60.0, "start": float(ini_time )} )  # , 'label': layer_name})           # specific PSG for L5B_PT neurons     90 changed to 60
      nest.Connect( pre=PSG, post=nest.GetNodes( layer_gid ) [0], syn_spec={'weight': 50.0, 'delay': 1.5} )
  elif layer_name == 'M1_L5A_CS':
      print( layer_name )
      PSG = nest.Create( 'poisson_generator', 1, params={'rate': 60.0, "start": float(ini_time )} )  # , 'label': layer_name})           # specific PSG for L5B_PT neurons      70 changed to 60
      nest.Connect( pre=PSG, post=nest.GetNodes( layer_gid ) [0], syn_spec={'weight': 50.0, 'delay': 1.5} )
  else:
      print( layer_name )
      PSG = nest.Create( 'poisson_generator', 1, params={'rate': 40.0, "start": float( ini_time_ini )} )  # , 'label': layer_name})
      nest.Connect( pre=PSG, post=nest.GetNodes( layer_gid ) [0], syn_spec={'weight': 50.0, 'delay': 1.5} )

  '''
  if layer_name in ['M1_L23_CC']:
      PSG = nest.Create( 'poisson_generator', 1, params={'rate': 800.0, "start": float( ini_time )} )
      nest.Connect( pre=PSG, post=nest.GetNodes( layer_gid ) [0], syn_spec={'weight': 4.0, 'delay': 1.5} )
  elif layer_name in ['M1_L5A_CC']:
      PSG = nest.Create( 'poisson_generator', 1, params={'rate': 800.0, "start": float( ini_time )} )
      nest.Connect( pre=PSG, post=nest.GetNodes( layer_gid ) [0], syn_spec={'weight': 4.0, 'delay': 1.5} )
  elif layer_name in ['M1_L5A_CS']:
      PSG = nest.Create( 'poisson_generator', 1, params={'rate': 950.0, "start": float( ini_time )} )
      nest.Connect( pre=PSG, post=nest.GetNodes( layer_gid ) [0], syn_spec={'weight': 4.0, 'delay': 1.5} )
  elif layer_name in ['M1_L5A_CT']:
      PSG = nest.Create( 'poisson_generator', 1, params={'rate': 800.0, "start": float( ini_time )} )
      nest.Connect( pre=PSG, post=nest.GetNodes( layer_gid ) [0], syn_spec={'weight': 4.0, 'delay': 1.5} )
  elif layer_name in ['M1_L5B_CC']:
      PSG = nest.Create( 'poisson_generator', 1, params={'rate': 950.0, "start": float( ini_time )} )
      nest.Connect( pre=PSG, post=nest.GetNodes( layer_gid ) [0], syn_spec={'weight': 4.0, 'delay': 1.5} )
  elif layer_name in ['M1_L5B_CS']:
      PSG = nest.Create( 'poisson_generator', 1, params={'rate': 950.0, "start": float(ini_time )} )  # , 'label': layer_name})           # specific PSG for L5B_PT neurons
      nest.Connect( pre=PSG, post=nest.GetNodes( layer_gid ) [0], syn_spec={'weight': 4.0, 'delay': 1.5} )
  elif layer_name in ['M1_L5B_PT']:
      PSG = nest.Create( 'poisson_generator', 1, params={'rate': 1100.0, "start": float( ini_time )} )
      nest.Connect( pre=PSG, post=nest.GetNodes( layer_gid ) [0], syn_spec={'weight': 4.0, 'delay': 1.5} )
  elif layer_name in ['M1_L6_CT']:
      PSG = nest.Create( 'poisson_generator', 1, params={'rate': 1150.0, "start": float( ini_time )} )
      nest.Connect( pre=PSG, post=nest.GetNodes( layer_gid ) [0], syn_spec={'weight': 4.0, 'delay': 1.5} )

  elif layer_name in ['M1_L1_SBC']:
      PSG = nest.Create( 'poisson_generator', 1, params={'rate': 1200.0, "start": float( ini_time_ini )} )
      nest.Connect( pre=PSG, post=nest.GetNodes( layer_gid ) [0], syn_spec={'weight': 4.5, 'delay': 1.5} )
  elif layer_name in ['M1_L1_ENGC']:
      PSG = nest.Create( 'poisson_generator', 1, params={'rate': 1200.0, "start": float( ini_time_ini )} )
      nest.Connect( pre=PSG, post=nest.GetNodes( layer_gid ) [0], syn_spec={'weight': 4.5, 'delay': 1.5} )
  elif layer_name in ['M1_L23_PV']:
      PSG = nest.Create( 'poisson_generator', 1, params={'rate': 1400.0, "start": float( ini_time_ini )} )
      nest.Connect( pre=PSG, post=nest.GetNodes( layer_gid ) [0], syn_spec={'weight': 4.5, 'delay': 1.5} )
  elif layer_name in ['M1_L23_SST']:
      PSG = nest.Create( 'poisson_generator', 1, params={'rate': 650.0, "start": float( ini_time_ini )} )
      nest.Connect( pre=PSG, post=nest.GetNodes( layer_gid ) [0], syn_spec={'weight': 4.5, 'delay': 1.5} )
  elif layer_name in ['M1_L23_VIP']:
      PSG = nest.Create( 'poisson_generator', 1, params={'rate': 1400.0, "start": float( ini_time_ini )} )
      nest.Connect( pre=PSG, post=nest.GetNodes( layer_gid ) [0], syn_spec={'weight': 4.5, 'delay': 1.5} )
  elif layer_name in ['M1_L5A_PV']:
      PSG = nest.Create( 'poisson_generator', 1, params={'rate': 1350.0, "start": float( ini_time_ini )} )
      nest.Connect( pre=PSG, post=nest.GetNodes( layer_gid ) [0], syn_spec={'weight': 4.5, 'delay': 1.5} )
  elif layer_name in ['M1_L5A_SST']:
      PSG = nest.Create( 'poisson_generator', 1, params={'rate': 700.0, "start": float( ini_time_ini )} )
      nest.Connect( pre=PSG, post=nest.GetNodes( layer_gid ) [0], syn_spec={'weight': 4.5, 'delay': 1.5} )
  elif layer_name in ['M1_L5B_PV']:
      PSG = nest.Create( 'poisson_generator', 1, params={'rate': 1400.0, "start": float( ini_time_ini )} )
      nest.Connect( pre=PSG, post=nest.GetNodes( layer_gid ) [0], syn_spec={'weight': 4.0, 'delay': 1.5} )
  elif layer_name in ['M1_L5B_SST']:
      PSG = nest.Create( 'poisson_generator', 1, params={'rate': 710.0, "start": float( ini_time_ini )} )
      nest.Connect( pre=PSG, post=nest.GetNodes( layer_gid ) [0], syn_spec={'weight': 4.0, 'delay': 1.5} )
  elif layer_name in ['M1_L6_PV']:
      PSG = nest.Create( 'poisson_generator', 1, params={'rate': 1550.0, "start": float( ini_time_ini )} )
      nest.Connect( pre=PSG, post=nest.GetNodes( layer_gid ) [0], syn_spec={'weight': 4.0, 'delay': 1.5} )
  elif layer_name in ['M1_L6_SST']:
      PSG = nest.Create( 'poisson_generator', 1, params={'rate': 710.0, "start": float( ini_time_ini )} )
      nest.Connect( pre=PSG, post=nest.GetNodes( layer_gid ) [0], syn_spec={'weight': 4.0, 'delay': 1.5} )
    '''
  detector = nest.Create( "spike_detector", params=params )
  nest.Connect(pre=nest.GetNodes( layer_gid ) [0], post=detector )

  return detector
#################################################################
def layer_poisson_generator(layer_gid, layer_name, params={"withgid": True, "withtime": True, "to_file": False}):
    #def layer_spike_detector(layer_gid, layer_name, params={"withgid": True, "withtime": True, "to_file": True, 'fbuffer_size': 8192}):
    params.update({'label': layer_name})

    # stimulation by using PSG
    # if we want to have indirect input through for example VIP
    #if layer_name == 'M1_L23_VIP':
        #PSG_arrow = nest.Create('poisson_generator', 1, params={'rate':  10000.0}) #, 'label': layer_name})
        #nest.Connect(pre=PSG_arrow, post=layer_gid, syn_spec={'weight': 100.0, 'delay': 1.5})
        #return PSG_arrow
    # if direct input is the goal
    if layer_name == 'M1_L5A_CS':
        PSG_arrow = nest.Create('poisson_generator', 1, params={'rate':  3000.0}) #, 'label': layer_name})
        nest.Connect(pre=PSG_arrow, post=layer_gid, syn_spec={'weight': 5.0, 'delay': 1.5})
        return PSG_arrow

    if layer_name == 'M1_L5B_PT':
        PSG_arrow = nest.Create('poisson_generator', 1, params={'rate':  3000.0}) #, 'label': layer_name})
        nest.Connect(pre=PSG_arrow, post=layer_gid, syn_spec={'weight': 5.0, 'delay': 1.5})
        return PSG_arrow

###############################################################
def layer_spike_detector(CS_gids, PT_gids, ignore_time, params = {"withgid": True, "withtime": True, "to_file": True}):
    params.update( {"start": float( ignore_time )} )
    # Add detector for stimulated neuron types inside the columns
    detector_5A = nest.Create( "spike_detector", params=params )
    nest.Connect( pre=CS_gids, post=detector_5A)

    detector_5B = nest.Create( "spike_detector", params=params )
    nest.Connect( pre=PT_gids, post=detector_5B)

    return detector_5A, detector_5B

########################################################################################################################
# Program initiation
########################################################################################################################
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


# 5) randomizing the membarne potential
import numpy
for layer_name in ctx_M1_layers.keys():
  Vth = -50.
  Vrest = -70.
  for neuron in nest.GetNodes(ctx_M1_layers[layer_name])[0]:
    nest.SetStatus([neuron], {"V_m": Vrest + (Vth - Vrest) * numpy.random.rand()} )


##############################
# 6) set paramaters for the the defining the structure of columns
radius = 0.125*sim_params['scalefactor'][0]
dis_from_center = 0.35*sim_params['scalefactor'][0]# to have separated physically separated columns limit for radius is r(max) = 0.125
###############################
# 7) selecting the neurons for the stimulation located inside the column
gids_inside = {}
gids_outside = {}
circles_center = [0,0]
num_of_columns = 6
# define 6 columns to stmiluate for the 'M1_L5A_CS' & 'M1_L5B_PT' cells separately
circle_centers = get_channel_centers(circles_center,num_of_columns ,dis_from_center, sim_params['scalefactor'])
layer_name = 'M1_L5A_CS'
gids_inside['M1_L5A_CS'], gids_outside['M1_L5A_CS'] = get_input_column_layers_ctx_M1(ctx_M1_layers, layer_name, radius, circle_centers, num_of_columns)
layer_name = 'M1_L5B_PT'
gids_inside['M1_L5B_PT'], gids_outside['M1_L5B_PT'] = get_input_column_layers_ctx_M1(ctx_M1_layers, layer_name, radius, circle_centers, num_of_columns)


# 8) generate the resting state firing
detectors = {}
#detector_5A={}
#detector_5B={}
detector_5A={}
detector_5B={}
for layer_name in ctx_M1_layers.keys():
    detectors[layer_name] = generate_resting_state(ctx_M1_layers[layer_name], layer_name, sim_params['initial_ignore'])
with open('./log/' + 'performance.txt', 'a') as file:
  file.write('Detectors_Elapse_Time: ' + str(time.time() - start_time ) + '\n')

gids_inside_CS = []
gids_inside_PT = []
for i in range(num_of_columns):
    gids_inside_CS= gids_inside['M1_L5A_CS'][i] + gids_inside_CS
    gids_inside_PT = gids_inside ['M1_L5B_PT'] [i] + gids_inside_PT

for i in range(num_of_columns):
    detector_5A[i] , detector_5B[i] = layer_spike_detector(tuple(gids_inside['M1_L5A_CS'][i]), tuple(gids_inside ['M1_L5B_PT'] [i]), sim_params['initial_ignore'])

################################
# 9) simulation
simulation_time = sim_params['simDuration']
initial_ignore = sim_params['initial_ignore']
print ('Simulation Started:')
#nest.Simulate(simulation_time)

poisson_gens = []
start_time=time.time()

poisson_gens.append(layer_poisson_generator(gids_inside['M1_L5A_CS'][0], 'M1_L5A_CS'))
poisson_gens.append(layer_poisson_generator(gids_inside['M1_L5B_PT'][0], 'M1_L5B_PT'))
nest.SetStatus( poisson_gens [0], {'origin': nest.GetKernelStatus() ['time'], "start": initial_ignore, 'stop': simulation_time/6 + initial_ignore} )
nest.SetStatus( poisson_gens [1], {'origin': nest.GetKernelStatus() ['time'], "start": initial_ignore, 'stop': simulation_time/6 + initial_ignore} )
nest.Simulate(simulation_time/6 + initial_ignore)

poisson_gens.append(layer_poisson_generator(gids_inside['M1_L5A_CS'][1], 'M1_L5A_CS'))
poisson_gens.append(layer_poisson_generator(gids_inside['M1_L5B_PT'][1], 'M1_L5B_PT'))
nest.SetStatus( poisson_gens [2], {'origin': nest.GetKernelStatus() ['time'], "start": 0.0, 'stop': simulation_time/6} )
nest.SetStatus( poisson_gens [3], {'origin': nest.GetKernelStatus() ['time'], "start": 0.0, 'stop': simulation_time/6} )
nest.Simulate(simulation_time/6)

poisson_gens.append(layer_poisson_generator(gids_inside['M1_L5A_CS'][2], 'M1_L5A_CS'))
poisson_gens.append(layer_poisson_generator(gids_inside['M1_L5B_PT'][2], 'M1_L5B_PT'))
nest.SetStatus( poisson_gens [4], {'origin': nest.GetKernelStatus() ['time'], "start": 0.0, 'stop': simulation_time/6} )
nest.SetStatus( poisson_gens [5], {'origin': nest.GetKernelStatus() ['time'], "start": 0.0, 'stop': simulation_time/6} )
nest.Simulate(simulation_time/6)

poisson_gens.append(layer_poisson_generator(gids_inside['M1_L5A_CS'][3], 'M1_L5A_CS'))
poisson_gens.append(layer_poisson_generator(gids_inside['M1_L5B_PT'][3], 'M1_L5B_PT'))
nest.SetStatus( poisson_gens [6], {'origin': nest.GetKernelStatus() ['time'], "start": 0.0, 'stop': simulation_time/6} )
nest.SetStatus( poisson_gens [7], {'origin': nest.GetKernelStatus() ['time'], "start": 0.0, 'stop': simulation_time/6} )
nest.Simulate(simulation_time/6)

poisson_gens.append(layer_poisson_generator(gids_inside['M1_L5A_CS'][4], 'M1_L5A_CS'))
poisson_gens.append(layer_poisson_generator(gids_inside['M1_L5B_PT'][4], 'M1_L5B_PT'))
nest.SetStatus( poisson_gens [8], {'origin': nest.GetKernelStatus() ['time'], "start": 0.0, 'stop': simulation_time/6} )
nest.SetStatus( poisson_gens [9], {'origin': nest.GetKernelStatus() ['time'], "start": 0.0, 'stop': simulation_time/6} )
nest.Simulate(simulation_time/6)

poisson_gens.append(layer_poisson_generator(gids_inside['M1_L5A_CS'][5], 'M1_L5A_CS'))
poisson_gens.append(layer_poisson_generator(gids_inside['M1_L5B_PT'][5], 'M1_L5B_PT'))
nest.SetStatus( poisson_gens [10], {'origin': nest.GetKernelStatus() ['time'], "start": 0.0, 'stop': simulation_time/6} )
nest.SetStatus( poisson_gens [11], {'origin': nest.GetKernelStatus() ['time'], "start": 0.0, 'stop': simulation_time/6} )
nest.Simulate(simulation_time/6)


with open('./log/'+'x_performance.txt', 'a') as file:
  file.write('Simulation_Elapse_Time: '+str(time.time()-start_time)+'\n')
print ('Simulation successfully finished!')
print ('Congradulation!!!!!!')



############################################################################
# 8) plotting the garphs
############################################################################
# setting for figs size and font
font = {'family' : 'normal', 'weight' : 'normal', 'size'   : 42}
plt.rcParams["figure.figsize"] = [64,48]
plt.rc('font', **font)
plt.rc('xtick', labelsize=40) 
plt.rc('ytick', labelsize=40)
##################################

# modify the raster plot representation
de_5A = {}
eve_5A = {}
eve_s_5A = {}
t_5A = {}
eve_5A_all = np.array(())
t_5A_all = np.array(())
de_5B = {}
eve_5B = {}
eve_s_5B = {}
t_5B = {}
eve_5B_all = np.array(())
t_5B_all = np.array(())
j = 0
n=0
for i in range(num_of_columns):
    de_5A[i] = nest.GetStatus(detector_5A[i],keys="events")[0]
    eve_5A[i] = de_5A[i]["senders"]
    t_5A[i] = de_5A[i]["times"]
    de_5B[i] = nest.GetStatus(detector_5B[i],keys="events")[0]
    eve_5B[i] = de_5B[i]["senders"]
    t_5B[i] = de_5B[i]["times"]

    eve_s_5A[i] = [None] * len(eve_5A[i])
    for m in range(len(eve_5A[i])):
        if eve_5A[i][m] > 10000:
            tar = eve_5A[i][m]
            j = j + 1
            eve_s_5A[m] = j
            eve_5A[i][m] = j
            for k in range(len(eve_5A[i])):
                if eve_5A[i][k] == tar:
                    eve_s_5A[k] = j
                    eve_5A[i][k] = j

    eve_5A_all= np.concatenate((eve_5A_all, eve_5A[i]), axis=None)
    t_5A_all= np.concatenate((t_5A_all, t_5A[i]), axis=None) 

    eve_s_5B[i] = [None] * len(eve_5B[i])
    for m in range(len(eve_5B[i])):
        if eve_5B[i][m] > 10000:
            tar = eve_5B[i][m]
            n = n + 1
            eve_s_5B[m] = n
            eve_5B[i][m] = n
            for k in range(len(eve_5B[i])):
                if eve_5B[i][k] == tar:
                    eve_s_5B[k] = n
                    eve_5B[i][k] = n

    eve_5B_all= np.concatenate((eve_5B_all, eve_5B[i]), axis=None)
    t_5B_all= np.concatenate((t_5B_all, t_5B[i]), axis=None)

# make a figures of six columns with different colors
figure, ax = plt.subplots()
ax.tick_params(axis='both',length = 5.0,width = 2.0)
plt.xlabel("time (ms)")
plt.ylabel("Neurons")
plt.title('All_columns_activity_5A_CS')
ax.plot(t_5A[0], eve_5A[0], "|", color='g')
ax.plot(t_5A[1], eve_5A[1], "|", color='r')
ax.plot(t_5A[2], eve_5A[2], "|", color='b')
ax.plot(t_5A[3], eve_5A[3], "|", color='g')
ax.plot(t_5A[4], eve_5A[4], "|", color='r')
ax.plot(t_5A[5], eve_5A[5], "|", color='b')
#pylab.show()
#save_results_to = '/home/morteza/Desktop/reports/190822/'
#plt.savefig(save_results_to + 'all_Columns_activity_5A_CS'+'.jpeg')

figure, ax = plt.subplots()
ax.tick_params(axis='both',length = 5.0,width = 2.0)
plt.xlabel("time (ms)")
plt.ylabel("Neurons")
plt.title('All_columns_activity_5B_PT')
ax.plot(t_5B[0], eve_5B[0], "|", color='g')
ax.plot(t_5B[1], eve_5B[1], "|", color='r')
ax.plot(t_5B[2], eve_5B[2], "|", color='b')
ax.plot(t_5B[3], eve_5B[3], "|", color='g')
ax.plot(t_5B[4], eve_5B[4], "|", color='r')
ax.plot(t_5B[5], eve_5B[5], "|", color='b')
#pylab.show()
save_results_to = '/home/morteza/Desktop/reports/190822/'
#plt.savefig(save_results_to + 'all_Columns_activity_5B_PT'+'.jpeg')

'''

# All columns in one step (5A_CS)
figure, ax = plt.subplots()
ax.tick_params(axis='both',length = 5.0,width = 2.0)
plt.xlabel("time (ms)")
plt.ylabel("Neurons")
plt.title('All_columns_activity')
pylab.plot(t_5A_all, eve_5A_all, "|")
#pylab.show()
save_results_to = '/home/morteza/Desktop/reports/190822/'
#plt.savefig(save_results_to + 'all_Columns_activity'+'.jpeg') 
'''