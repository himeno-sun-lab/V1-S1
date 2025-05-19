#!/usr/bin/env python
# -*- coding: utf-8 -*-

# This is the program to prepare necessary file to check LSPS on M1 layers
# compare to file vLSPS_main_pre_M1 this try to use more functions to speed up run time


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
        file_params = run_path('baseCTXM1Params_1903.py', init_globals=globals())
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
def get_input_column_layers_ctx_M1(ctx_M1_layers, layer_name, radius):
    # tmp progress
    # make circles for bg input
    radius_big = 0.5 / np.cos(np.pi / 4.) / 2.
    radius_small = 0.1
    circle_center = [[0.0, 0.0]]
    inside_gids = []
    outside_gids = []
    layer_gids = nest.GetNodes(ctx_M1_layers[layer_name])[0]
    neuron_positions = ntop.GetPosition(layer_gids)
    for nn in range(len(layer_gids)):
        #print (neuron_positions[nn])

        #print (circle_center[0])
        if np.linalg.norm([(neuron_positions[nn][0] - circle_center[0][0]), (neuron_positions[nn][1] - circle_center[0][1])]) <= radius:
            inside_gids.append(layer_gids[nn])
        else:
            outside_gids.append(layer_gids[nn])
    #circle_gids=[circle_1_gids, circle_2_gids, circle_3_gids, circle_4_gids, circle_5_gids, circle_6_gids, circle_7_gids, circle_8_gids]
    if layer_name == 'M1_L1_ENGC':
        print ('all neuron PV number was: ', len(layer_gids), 'and only', len(inside_gids), 'neurons were stimulated inside the column using poisson generator....')
    return inside_gids , outside_gids

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

    Neuron_GID = []
    index = 0
    record_gid = []

    pos_Exc = np.zeros( (0, 4) )
    pos_inh = np.zeros( (0, 4) )
    M1_layer_Name = ctx_M1_params ['M1'] ['structure_info'] ['Layer_Name']
    M1_layer_size = ctx_M1_params ['M1'] ['structure_info'] ['region_size']
    #M1_layer_size = [i * 1000 for i in M1_layer_size]
    # M1_layer_size = np.array( M1_layer_size )         may this is used later
    M1_layer_thickness = ctx_M1_params ['M1'] ['structure_info'] ['layer_thickness']
    #M1_layer_thickness = [i *1000 for i in M1_layer_thickness]
    M1_layer_thickness_ST = np.cumsum( M1_layer_thickness)
    M1_layer_thickness_ST = np.insert( M1_layer_thickness_ST, 0, 0.0 )
    M1_layer_depth = ctx_M1_params ['M1'] ['structure_info'] ['layer_depth']
    #M1_layer_depth = [i * 1000 for i in M1_layer_depth]
    sd_list = []
    detectors={}
    multimeter_exc = []
    multimeter_inh = []
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
            detector = nest.Create( "spike_detector", params={'to_file': True, "label": "sd_M1_%s_%s" % (M1_layer_Name [l], n_type), "use_gid_in_filename": True} )
            sd_list.append(detector)
            neuronmodel = copy_neuron_model(ctx_M1_params [region_name] ['neuro_info'] [M1_layer_Name [l]] [n_type] ['neuron_model'], n_type_info, region_name + '_' + M1_layer_Name [l] + '_' + n_type )
            #neuronmodel = ctx_M1_params ['M1'] ['neuro_info'] [M1_layer_Name [l]] [n_type] ['neuron_model']
            elements = neuronmodel
            neuron_info = ctx_M1_params ['M1'] ['neuro_info'] [M1_layer_Name [l]] [n_type]
            detectors["sd_M1_%s_%s" % (M1_layer_Name [l], n_type)] = detector
            Neuron_pos = gen_neuron_postions_ctx( M1_layer_depth [l], M1_layer_thickness [l], n_type_info ['Cellcount_mm2'], M1_layer_size, scalefactor, 'Neuron_pos_' + region_name + '_' + M1_layer_Name [l] + '_' + n_type )
            if n_type_info ['EorI'] == "E":
                nest.SetDefaults( elements, {"I_e": float( neuron_info ['I_ex'] ), "V_th": float( neuron_info ['spike_threshold'] ), "V_reset": float( neuron_info ['reset_value'] ), "t_ref": float( neuron_info ['absolute_refractory_period'] )} )
                SubSubRegion_Excitatory.append(create_layers_ctx_M1( topo_extend, topo_center, Neuron_pos, neuronmodel ) )
                ctx_M1_layers [region_name + '_' + M1_layer_Name [l] + '_' + n_type] = SubSubRegion_Excitatory[-1]
                Neuron_GIDs = None
                Neuron_GIDs = nest.GetNodes(SubSubRegion_Excitatory[-1])[0]
                print( 'Neuron_num:' + str( len( Neuron_GIDs ) ) )
                nest.Connect( Neuron_GIDs, detector)
                multimeter_exc.append(nest.Create( "multimeter", params={"interval": 0.1, "record_from": ["V_m", "g_ex", "g_in"], "withgid": True, "to_file": True, "label": "mm_vLSPS_%s_%s" %(M1_layer_Name [l], n_type)} ) )
                print( ntop.FindCenterElement( SubSubRegion_Excitatory [-1] ) )
                nest.Connect( multimeter_exc [-1], ntop.FindCenterElement( SubSubRegion_Excitatory [-1] ) )
                save_layers_position( region_name + '_' + M1_layer_Name [l] + '_' + n_type, SubSubRegion_Excitatory [-1], Neuron_pos )
            elif n_type_info ['EorI'] == "I":
                nest.SetDefaults( elements, {"I_e": float( neuron_info ['I_ex'] ), "V_th": float( neuron_info ['spike_threshold'] ), "V_reset": float( neuron_info ['reset_value'] ), "t_ref": float( neuron_info ['absolute_refractory_period'] )} )
                SubSubRegion_Inhibitory.append(create_layers_ctx_M1( topo_extend, topo_center, Neuron_pos, neuronmodel ) )
                ctx_M1_layers [region_name + '_' + M1_layer_Name [l] + '_' + n_type] = SubSubRegion_Inhibitory [-1]
                Neuron_GIDs=None
                Neuron_GIDs = nest.GetNodes( SubSubRegion_Inhibitory [-1] ) [0]
                print( 'Neuron_num:' + str( len( Neuron_GIDs ) ) )
                #nest.Connect( Neuron_GIDs, sd_list [-1] )
                nest.Connect( Neuron_GIDs, detector)
                multimeter_inh.append(nest.Create( "multimeter", params={"interval": 0.1, "record_from": ["V_m", "g_ex", "g_in"], "withgid": True, "to_file": True, "label": "mm_vLSPS_%s_%s" %(M1_layer_Name [l], n_type)} ) )
                print(ntop.FindCenterElement( SubSubRegion_Inhibitory [-1] ) )
                nest.Connect( multimeter_inh [-1], ntop.FindCenterElement( SubSubRegion_Inhibitory [-1] ) )
                save_layers_position( region_name + '_' + M1_layer_Name [l] + '_' + n_type, SubSubRegion_Inhibitory [-1], Neuron_pos )

            else:
                print( 'Unknow E or I' )

    np.savetxt( "pos_Exc.csv", pos_Exc, delimiter="," )
    np.savetxt( "pos_inh.csv", pos_inh, delimiter="," )

    M1_internal_connection = np.load( ctx_M1_params ['M1'] ['connection_info'] ['M1toM1'] )
    for pre_layer_name in ctx_M1_layers.keys():
        for post_layer_name in ctx_M1_layers.keys():
            print( 'start to connect ' + pre_layer_name + ' with ' + post_layer_name )
            connect_layers_ctx_M1( ctx_M1_layers [pre_layer_name], ctx_M1_layers [post_layer_name], M1_internal_connection [pre_layer_name] [post_layer_name] )
            #conn_num = len( nest.GetConnections( nest.GetNodes( ctx_M1_layers [pre_layer_name] ) [0], nest.GetNodes( ctx_M1_layers [post_layer_name] ) [0] ) )
            #ctx_layers_conn [pre_layer_name] [post_layer_name] = {'conn_num': conn_num, 'neuron_num': len(nest.GetNodes( ctx_M1_layers [pre_layer_name] ) [0] )}

    return ctx_M1_layers, SubSubRegion_Excitatory, SubSubRegion_Inhibitory, detectors

###############################################################
def generate_resting_state(layer_gid, layer_name, ignore_time, params={"withgid": True, "withtime": True, "to_file": False}):
#def layer_spike_detector(layer_gid, layer_name, params={"withgid": True, "withtime": True, "to_file": True, 'fbuffer_size': 8192}):
  params.update({'label': layer_name, "start": float(ignore_time)}) #  , "start": float(ignore_time)
  ini_time = ignore_time - 150
  ini_time_ini = ignore_time - 300
  # first adding poisson generator one for all neuron types different for the inhibitory and excitatory neurons
  name_list = ['M1_L23_CC', 'M1_L5A_CC', 'M1_L5A_CT', 'M1_L5B_CC', 'M1_L5B_CS', 'M1_L6_CT']
  if layer_name in name_list:
    PSG = nest.Create('poisson_generator', 1, params={'rate':  30.0, "start": float(ini_time)}) #, 'label': layer_name})
    print(layer_name)
    nest.Connect(pre=PSG, post=nest.GetNodes(layer_gid)[0], syn_spec={'weight': 50.0, 'delay': 1.5})
  elif layer_name == 'M1_L5B_PT':
    print(layer_name)
    PSG = nest.Create('poisson_generator', 1, params={'rate': 90.0,"start": float(ini_time)}) #, 'label': layer_name})           # specific PSG for L5B_PT neurons
    nest.Connect(pre=PSG, post=nest.GetNodes(layer_gid)[0], syn_spec={'weight': 50.0, 'delay': 1.5})
  elif layer_name == 'M1_L5A_CS':
    print(layer_name)
    PSG = nest.Create('poisson_generator', 1, params={'rate': 70.0,"start": float(ini_time)}) #, 'label': layer_name})           # specific PSG for L5B_PT neurons
    nest.Connect(pre=PSG, post=nest.GetNodes(layer_gid)[0], syn_spec={'weight': 50.0, 'delay': 1.5})
  else:
    print(layer_name)
    PSG = nest.Create('poisson_generator', 1, params={'rate': 40.0, "start": float(ini_time_ini)}) #, 'label': layer_name})
    nest.Connect(pre=PSG, post=nest.GetNodes(layer_gid)[0], syn_spec={'weight': 50.0, 'delay': 1.5})


#################################################################
def layer_poisson_generator(layer_gid, layer_name, stim_onset, stim_offset,ignore_time, params={"withgid": True, "withtime": True, "to_file": False}):
    #def layer_spike_detector(layer_gid, layer_name, params={"withgid": True, "withtime": True, "to_file": True, 'fbuffer_size': 8192}):
    params.update({'label': layer_name, "start": float(ignore_time)})

    # stimulation by using PSG
    if layer_name == 'M1_L1_ENGC':
        PSG = nest.Create('poisson_generator', 1, params={'rate':  10000.0, "start": float(stim_onset), 'stop': float(stim_offset)}) #, 'label': layer_name})
        nest.Connect(pre=PSG, post=layer_gid, syn_spec={'weight': 100.0, 'delay': 1.5})
        return PSG
    '''
    # Stimulationm by using dc_generator
    if layer_name == 'M1_L1_ENGC':
        stim_amplitude = 10000.0
        stim_duration = 500.0
        params = {'amplitude': stim_amplitude, 'start': float(stim_onset), 'stop': float(stim_onset + stim_duration)}
        DCG = nest.Create( "dc_generator", 1, params)
        nest.Connect(pre = DCG, post = layer_gid)
    '''
###############################################################
def layer_spike_detector(layer_gids_inside, layer_gids_outside, layer_name, record_onset, ignore_time, params = {"withgid": True, "withtime": True, "to_file": False}):
    # def layer_spike_detector(layer_gid, layer_name, params={"withgid": True, "withtime": True, "to_file": True, 'fbuffer_size': 8192}):
    print( 'spike detector for ' + layer_name )

    # Add detector for all neuron types
    detector_in = nest.Create( "spike_detector", params=params )
    nest.Connect( pre=layer_gids_inside, post=detector_in )

    detector_out = nest.Create( "spike_detector", params=params )
    nest.Connect( pre=layer_gids_outside, post=detector_out)

    # add multimeter just for to record V_m and conductance (inhibitory and excitatory) of a single cell each cell population
    nest_mm_in = nest.Create( 'multimeter', params={"interval": 0.1, "record_from": ["V_m", "g_ex", "g_in"], "withgid": True, "to_file": False, 'label': layer_name, 'withtime': True, "start": float( ignore_time), 'use_gid_in_filename': True} )  #
    #nest.Connect( nest_mm, [nest.GetNodes( layer_gid ) [0] [7]] )
    nest.Connect( nest_mm_in, [layer_gids_inside[7]] )
    nest_mm_out = nest.Create( 'multimeter', params={"interval": 0.1, "record_from": ["V_m", "g_ex", "g_in"], "withgid": True, "to_file": False, 'label': layer_name, 'withtime': True, "start": float( ignore_time ), 'use_gid_in_filename': True} )  #
    nest.Connect( nest_mm_out, [layer_gids_outside[7]])

    voltmeter_single = nest.Create( "voltmeter", params={'to_file': False, 'label': layer_name, "start": float( ignore_time ), "withgid": True, "withtime": True} )  # 'use_gid_in_filename': True,   "start": float(ignore_time),
    nest.Connect(voltmeter_single, [layer_gids_inside[7]] )

    return detector_in, detector_out, nest_mm_in, nest_mm_out, voltmeter_single

'''
###############################################################
def average_fr(detectors):
  #return nest.GetStatus(detector, 'n_events')[0] / (float(simDuration) * float(n) / 1000.0)
  n_before = 0
  n = 0
  n_after = 0
  for t in nest.GetStatus(detectors)[0]['events']['times']:
      if t < 500.0 :
          n_before = 1 + n_before
      elif t <= 1000.0 and t>=500.0:
          n =1 + n
      elif t > 1000.0:
          n_after = 1 + n_after

          return n_before, n, n_after,


###############################################################
'''

##########################
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
M1_layer_size = ctx_M1_params ['M1'] ['structure_info'] ['region_size']
#M1_layer_size = [i * 1000 for i in M1_layer_size]
# M1_layer_size = np.array( M1_layer_size )         may this is used later
M1_layer_Name = ctx_M1_params ['M1'] ['structure_info'] ['Layer_Name']
M1_layer_thickness = ctx_M1_params ['M1'] ['structure_info'] ['layer_thickness']
#M1_layer_thickness = [i *1000 for i in M1_layer_thickness]
M1_layer_thickness_ST=np.cumsum(M1_layer_thickness)
M1_Layer_thickness_ST=np.insert(M1_layer_thickness_ST, 0, 0.0)
M1_layer_depth = ctx_M1_params ['M1'] ['structure_info'] ['layer_depth']
#M1_layer_depth = [i * 1000 for i in M1_layer_depth]


###########################
# 4) making layers and doing all connections
print('create and connect the M1 layers')
start_time = time.time()
ctx_M1_layers, SubSubRegion_Excitatory, SubSubRegion_Inhibitory, detectors =instantiate_ctx_M1(ctx_M1_params, sim_params['scalefactor'])
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
# 6) set paramaters for the stimulation of ENGC neurons inside the column\
stim_onset = 1000.0
stim_offset = 1500.0
simulation_duration = sim_params['simDuration']
record_onset = 500.0

###############################
# 7) selecting the ENGC neurons for the stimulation located inside the coloumn
poisson_gens = {}
layer_name = 'M1_L1_ENGC'
ENGC_gids_inside, ENGC_gids_outside = get_input_column_layers_ctx_M1(ctx_M1_layers, layer_name, radius = 0.3)
poisson_gens[layer_name]= layer_poisson_generator(ENGC_gids_inside, layer_name, stim_onset, stim_offset, sim_params['initial_ignore'])


###############################
# 8) recording from other neurons in circle and out side of circle
multimeters_in = {}
multimeters_out = {}
voltmeters = {}
detectors_in = {}
detectors_out = {}
layer_gids_inside = {}
layer_gids_outside = {}
for layer_name in ctx_M1_layers.keys():
    layer_gids_inside[layer_name], layer_gids_outside[layer_name] = get_input_column_layers_ctx_M1(ctx_M1_layers, layer_name, radius=0.3)
    detectors_in[layer_name], detectors_out[layer_name], multimeters_in[layer_name], multimeters_out[layer_name], voltmeters[layer_name] = layer_spike_detector(layer_gids_inside[layer_name], layer_gids_outside[layer_name],
                                                                                                                                                                layer_name, record_onset, sim_params['initial_ignore'])

################################
# 9) simulation
simulation_time = sim_params['simDuration'] + sim_params['initial_ignore']
print ('Simulation Started:')
start_time=time.time()
nest.Simulate(simulation_time)
with open('./log/'+'x_performance.txt', 'a') as file:
  file.write('Simulation_Elapse_Time: '+str(time.time()-start_time)+'\n')

################################
'''
# 7) results
for layer_name in ctx_M1_layers.keys():
  #rate_on = average_fr(detectors_in[layer_name], 500.0, len(layer_gids_inside[layer_name]))
  #rate_off = average_fr(detectors_out[layer_name], 500.0, len(layer_gids_inside [layer_name]))

  n_in_before={}
  n_in = {}
  n_in_after = {}
  n_out_before = {}
  n_out = {}
  n_out_after = {}

  n_in_before[layer_name], n_in[layer_name], n_in_after[layer_name] = average_fr(detectors_in[layer_name])
  n_out_before[layer_name], n_out[layer_name], n_out_after[layer_name] = average_fr( detectors_out[layer_name] )
  # rate_off = average_fr(detectors_out[layer_name], 500.0, len(layer_gids_inside [layer_name]))
  
  print('Layer '+layer_name+"_inside produce "+ n_in_before + "and")
  with open( './log/' + 'report.txt', 'a' ) as file:
    file.write('Layer '+layer_name+" fires at "+str(rate)+" Hz" + '\n' )
'''
print ('Simulation successfully finished!')
print ('Congradulation!!!!!!')


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

# Membrane potential of single neurons
i=1
for layer_name in ctx_M1_layers.keys():
  plt.figure()
  nest.voltage_trace.from_device(multimeters_out[layer_name], title=layer_name)
  plt.show()
  save_results_to = '/home/morteza/Desktop/reports/190306/Vertical-column-results/2-1700_ENGC/mp_out/'
  plt.savefig(save_results_to + str(i) +'.jpeg')
  i=i+1

###############
i=1
for layer_name in ctx_M1_layers.keys():
  plt.figure()
  events_in = nest.GetStatus(multimeters_in[layer_name])[0]["events"]
  events_out = nest.GetStatus(multimeters_out[layer_name])[0]["events"]
  t = events_in["times"]
  plt.plot(t, events_in["g_ex"], t, events_in["g_in"], t, events_out["g_ex"], t, events_out["g_in"])
  plt.xlabel("time (ms)")
  plt.ylabel("synaptic conductance (nS)")
  plt.title(layer_name)
  plt.legend(("g_exc_in", "g_inh_in", "g_exc_out", "g_inh_out"))
  save_results_to = '/home/morteza/Desktop/reports/190306/Vertical-column-results/2-1700_ENGC/con/'
  plt.savefig(save_results_to + str(i) +'.jpeg')
  i=i+1

###############
i=1
for layer_name in ctx_M1_layers.keys():
  nest.raster_plot.from_device(detectors_out[layer_name], title = layer_name, hist=False)
  font = {'family' : 'normal', 'weight' : 'normal', 'size'   : 42}
  plt.rcParams["figure.figsize"] = [64,48]
  plt.rc('font', **font)
  plt.rc('xtick', labelsize=40) 
  plt.rc('ytick', labelsize=40)  
  save_results_to = '/home/morteza/Desktop/reports/190306/Vertical-column-results/2-1700_ENGC/raster_out/'
  plt.savefig(save_results_to + str(i) +'.jpeg', bbox_inches='tight')
  i=i+1

###############
'''








