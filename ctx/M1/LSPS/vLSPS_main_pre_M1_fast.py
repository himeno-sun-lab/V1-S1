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
  if sigma_x != 0:
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
###############################################################


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









##############################
# LSPS
##############################
# 5) config LSPS laser
stimu_duration = 1  # ms
stimu_interval = 400  # ms
#vLSPS_volume = 100  # micron
stim_amplitude=10000.
stim_start=100

vLSPS_grid_shape = [16, 16]
vLSPS_grid_size = [M1_layer_size[0] / 16., M1_layer_size[2] / 16.]
vLSPS_radii = 0.045 # micro
grid_laser=[]

vLSPS_grid_center_x=np.linspace(-M1_layer_size[0]/2.,M1_layer_size[0]/2. ,num=vLSPS_grid_shape[0]+1, endpoint=True)+vLSPS_grid_size[0]/2
vLSPS_grid_center_y=np.linspace(0, M1_layer_size[2],num=vLSPS_grid_shape[1]+1, endpoint=True)+vLSPS_grid_size[1]/2
vLSPS_grid_center=np.zeros((vLSPS_grid_shape[0], vLSPS_grid_shape[1], 2))

for i in range (vLSPS_grid_shape[0]):
    for j in range(vLSPS_grid_shape[1]):
        vLSPS_grid_center[i, j, :]=[vLSPS_grid_center_x[i], vLSPS_grid_center_y[j]]

grid_laser_index=0
for m in range(vLSPS_grid_shape[0]):
    for n in range(vLSPS_grid_shape[1]):
        vLSPS_t_start=(stimu_interval)*grid_laser_index+stim_start
        vLSPS_t_end=(stimu_interval)*grid_laser_index+stim_start+stimu_duration
        params = {'amplitude': stim_amplitude,'start': float(vLSPS_t_start),'stop': float(vLSPS_t_end)}
        grid_laser.append(nest.Create("dc_generator", 1, params))
        grid_laser_index+=1

print ('Gridding finished')


grid_laser_index=0
start_time = time.time()
for n in range( vLSPS_grid_shape[1]):
    for m in range(vLSPS_grid_shape[0]):
        vLSPS_stimu_neurons_GID=[]
        for sei in range (len(SubSubRegion_Excitatory)):
            neurons=nest.GetNodes(SubSubRegion_Excitatory[sei])[0]
            neuron_positions=ntop.GetPosition(neurons)
            for nn in range (len(neurons)):
                euclidean_distance = np.linalg.norm(([neuron_positions[nn][0], neuron_positions[nn][2]]- vLSPS_grid_center[m, n]))
                if euclidean_distance <= vLSPS_radii:
                    #print (neurons[nn])
                    vLSPS_stimu_neurons_GID.append(neurons[nn])
        nest.Connect(grid_laser[grid_laser_index], vLSPS_stimu_neurons_GID)
        grid_laser_index+=1
nest.Simulate((stimu_interval)*(vLSPS_grid_shape[0]*vLSPS_grid_shape[1]-1)+stim_start+stimu_duration+stimu_interval)
print ('LSPS finished')
with open('./log/' + 'x_performance.txt', 'a') as file:
  file.write('All LSPS time: ' + str(time.time() - start_time ) + '\n')

'''
##############################################
# 6) deleting all empty files
import  os
path = "/home/morteza/PycharmProjects/postk_wb/code/ctx/M1/LSPS/log"
file_num = 0
for root,dirs,files in os.walk(path):
    for name in files:
        filename = os.path.join(root,name)
        if os.stat(filename).st_size == 0:
            print(" Removing ",filename)
            os.remove(filename)
            file_num = file_num +1

with open('./log/' + 'x_performance.txt', 'a') as file:
  file.write('Number of deleted empty files is: ' + str(file_num) + '\n')
'''