#!/usr/bin/env python
# -*- coding: utf-8 -*-

# This is the program to prepare necessary file to check LSPS on M1 layers
# this file was adopted from "vLSPS_main_pre" (original vLSPS for S1) to work for M1

import nest
import numpy as np
import nest.topology as ntop
print (nest.version())
nest.ResetKernel()
import configparser
import shelve
#from simu_fun import *
#from vLSPS_fun import *
import pickle
import math

nest.ResetKernel()

#read configure file
config = configparser.ConfigParser()
config.read('simu.conf')
#local_num_threads=int(config['parallel_computing']['local_num_threads'])
#if local_num_threads!=1:
    #print ('Setlocal_num_threads')
    #nest.SetKernelStatus({"local_num_threads": local_num_threads})

###################################################################################
###################################################################################
from runpy import run_path
file_params = run_path('baseSimParams.py', init_globals=globals())
sim_params = file_params['simParams']
######################################
from runpy import run_path
file_params = run_path('baseCTXM1Params.py', init_globals=globals())
ctx_M1_params = file_params['ctxM1Params']
######################################
nest.set_verbosity("M_WARNING")
nest.SetKernelStatus({"overwrite_files": sim_params['overwrite_files']}) # should we erase previous traces when redoing a simulation?
nest.SetKernelStatus({'local_num_threads': int(sim_params['nbcpu'])})
nest.SetKernelStatus({"data_path": 'log'})
if sim_params['dt'] != '0.1':
    nest.SetKernelStatus({'resolution': float(sim_params['dt'])})

#############################################################################################
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

##############################################################
def connect_layers_ctx_M1(pre_SubSubRegion, post_SubSubRegion, conn_dict):
  sigma_x = conn_dict ['sigma'] / 1000.
  sigma_y = conn_dict ['sigma'] / 1000.
  weight_distribution = conn_dict ['weight_distribution']
  if weight_distribution == 'lognormal':
    conndict = {'connection_type': 'divergent',
                'mask': {'spherical': {'radius': 2.0}},
                'kernel': {
                  'gaussian2D': {'p_center': float( conn_dict ['p_center'] ), 'sigma_x': sigma_x,
                                 'sigma_y': sigma_y}},
                'weights': {'lognormal': {'mu': float( conn_dict ['weight'] ), 'sigma': 1.0}},
                'delays': float( conn_dict ['delay'] ),
                'allow_autapses': False,
                'allow_multapses': False}
  else:
    conndict = {'connection_type': 'divergent',
                'mask': {'spherical': {'radius': 2.0}},
                'kernel': {
                  'gaussian2D': {'p_center': float( conn_dict ['p_center'] ), 'sigma_x': sigma_x,
                                 'sigma_y': sigma_y}},
                'weights': float( conn_dict ['weight'] ),
                'delays': float( conn_dict ['delay'] ),
                'allow_autapses': False,
                'allow_multapses': False}
  if sigma_x != 0:
    ntop.ConnectLayers( pre_SubSubRegion, post_SubSubRegion, conndict )


###########################################################
def create_layers_ctx_M1(extent, center, positions, elements):
  newlayer = ntop.CreateLayer( {'extent': extent, 'center': center, 'positions': positions, 'elements': elements} )
  return newlayer

##########################################################################
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
##########################################################################


M1_layer_size = ctx_M1_params ['M1'] ['structure_info'] ['region_size']
# M1_layer_size = np.array( M1_layer_size )         may this is used later
M1_layer_Name = ctx_M1_params ['M1'] ['structure_info'] ['Layer_Name']
M1_layer_thickness = ctx_M1_params ['M1'] ['structure_info'] ['layer_thickness']
M1_layer_thickness_ST=np.cumsum(M1_layer_thickness)
M1_Layer_thickness_ST=np.insert(M1_layer_thickness_ST, 0, 0.0)
M1_layer_depth = ctx_M1_params ['M1'] ['structure_info'] ['layer_depth']
scalefactor = sim_params['scalefactor']

#config LSPS laser
stimu_duration = 1  # ms
stimu_interval = 400  # ms
#vLSPS_volume = 100  # micron
stim_amplitude=10000.
stim_start=100

vLSPS_grid_shape = [16, 16]
vLSPS_grid_size = [M1_layer_size[0] / 16., M1_layer_size[2] / 16.]
vLSPS_radii = 45 # micro
grid_laser=[]

vLSPS_grid_center_x=np.linspace(-M1_layer_size[0]/2.,M1_layer_size[0]/2. ,num=vLSPS_grid_shape[0]+1, endpoint=True)+vLSPS_grid_size[0]/2
vLSPS_grid_center_y=np.linspace(-M1_layer_size[2]/2., M1_layer_size[2]/2.,num=vLSPS_grid_shape[1]+1, endpoint=True)+vLSPS_grid_size[1]/2
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


SubSubRegion_Excitatory = []
SubSubRegion_Inhibitory = []
Neuron_GID = []
index = 0
M1_layer_Name = ctx_M1_params ['M1']['structure_info']['Layer_Name']
pos_Exc=np.zeros((0, 4))
pos_inh=np.zeros((0, 4))
sd_list=[]

multimeter_exc=[]
multimeter_inh=[]

topo_extend = [M1_layer_size [0] * int(scalefactor [0] ) + 1., M1_layer_size [1] * int( scalefactor [1] ) + 1., 2.]
topo_center = [0.0, 0.0, 0.0]
'''
topo_extend_m=np.array([1440., 400., 1366.])+1
topo_extend=[1441., 401., 1361.]
topo_extend_tmp=topo_extend_m-1
topo_center=np.array([0., 0., 0.])
topo_center=[0.0, 0.0, 0.0]
'''
ctx_M1_layers={}
region_name = 'M1'
record_gid=[]
for l in range(len(M1_layer_Name)):
    print('###########################################')
    print('start to create layer: ' + M1_layer_Name[l])
    #Neuron_pos_fileload = np.load(ctx_M1_params ['M1']['position_type_info'][M1_layer_Name[l]])
    #Neuron_pos = Neuron_pos_fileload['Neuron_pos']
    print('Network Architecture:')
    #print(Neuron_pos.shape)
    #Neuron_Types = np.unique(Neuron_pos[:, :, :, 3])
    # print (Neuron_Types.shape[0])
    n_type_index = 0
    #print (ctx_M1_params ['M1']['neuro_info'][M1_Layer_Name[l]].keys())
    #z_center=(M1_layer_thickness_ST[l + 1]-M1_layer_thickness_ST[l])/2.+M1_layer_thickness_ST[l]
    #topo_center[2]=z_center-topo_extend_tmp[2]/2.
    topo_center [2] = M1_layer_depth [l] + 0.5 * M1_layer_thickness [l]
    for n_type in ctx_M1_params ['M1']['neuro_info'][M1_layer_Name[l]].keys():
        n_type_index=ctx_M1_params ['M1']['neuro_info'][M1_layer_Name[l]][n_type]['n_type_index']
        print('n_type_index:', n_type_index)
        print(n_type)
        n_type_info = ctx_M1_params ['M1']['neuro_info'][M1_layer_Name[l]][n_type]
        sd_list.append(nest.Create("spike_detector", params={'to_file': True, "label": "sd_M1_%s_%s"%(M1_layer_Name[l], n_type), "use_gid_in_filename": True}))
        neuronmodel = copy_neuron_model(
            ctx_M1_params [region_name] ['neuro_info'] [M1_layer_Name [l]] [n_type] ['neuron_model'], n_type_info, region_name + '_' + M1_layer_Name [l] + '_' + n_type )
        #neuronmodel=ctx_M1_params ['M1']['neuro_info'][M1_layer_Name[l]][n_type]['neuron_model']
        elements=neuronmodel
        neuron_info=ctx_M1_params ['M1']['neuro_info'][M1_layer_Name[l]][n_type]
        Neuron_pos = gen_neuron_postions_ctx( M1_layer_depth [l], M1_layer_thickness [l], n_type_info ['Cellcount_mm2'], M1_layer_size, scalefactor, 'Neuron_pos_' + region_name + '_' + M1_layer_Name [l] + '_' + n_type )
        if n_type_info['EorI'] == "E":
            #pos=None
            #pos = Neuron_pos[np.where(Neuron_pos[:, :, :, 3] == n_type_index)]
            nest.SetDefaults(elements, {"I_e": float(neuron_info['I_ex']), "V_th": float(neuron_info['spike_threshold']), "V_reset": float(neuron_info['reset_value']), "t_ref": float(neuron_info['absolute_refractory_period'])})
            #newlayer = ntop.CreateLayer({topo_extend, topo_center, pos, neuronmodel} )
            SubSubRegion_Excitatory.append(create_layers_ctx_M1(topo_extend, topo_center, Neuron_pos, neuronmodel))
            ctx_M1_layers[region_name + '_' + M1_layer_Name [l] + '_' + n_type] = SubSubRegion_Excitatory[-1]
            '''
            pos_tmp_exc=np.zeros((pos.shape[0],4))
            for i in range (pos.shape[0]):
                pos_tmp_exc[i, 0]=nest.GetNodes(SubSubRegion_Excitatory[-1])[0][i]
                pos_tmp_exc[i, 1:] =pos[i, :3]
            #SubRegion_SetStatus(SubSubRegion_Excitatory[-1], n_type_info)
            pos_Exc=np.append(pos_Exc, pos_tmp_exc, axis=0)
            np.savetxt("position_M1_" + M1_layer_Name[l]+"_"+n_type+".csv", pos_tmp_exc, delimiter=",")
            '''
            Neuron_GIDs=None
            Neuron_GIDs = nest.GetNodes(SubSubRegion_Excitatory[-1])[0]
            print ('Neuron_num:'+str(len(Neuron_GIDs)))
            nest.Connect(Neuron_GIDs, sd_list[-1])
            multimeter_exc.append(nest.Create("multimeter", params={"interval": 0.1, "record_from": ["V_m", "g_ex", "g_in"],
                                                      "withgid": True, "to_file": True,
                                                      "label": "mm_vLSPS_postsynaptic_cell_%s_%s" % (M1_layer_Name[l], n_type)}))
            print (ntop.FindCenterElement(SubSubRegion_Excitatory[-1]))
            nest.Connect(multimeter_exc[-1], ntop.FindCenterElement(SubSubRegion_Excitatory[-1]))
        elif n_type_info['EorI'] == "I":
            #pos = None
            #pos = Neuron_pos[np.where(Neuron_pos[:, :, :, 3] == n_type_index)]
            nest.SetDefaults(elements, {"I_e": float(neuron_info['I_ex']), "V_th": float(neuron_info['spike_threshold']), "V_reset": float(neuron_info['reset_value']), "t_ref": float(neuron_info['absolute_refractory_period'])})
            #newlayer = ntop.CreateLayer({topo_extend, topo_center, pos, neuronmodel})
            SubSubRegion_Inhibitory.append(create_layers_ctx_M1(topo_extend, topo_center, Neuron_pos, neuronmodel))
            ctx_M1_layers[region_name + '_' + M1_layer_Name [l] + '_' + n_type] = SubSubRegion_Inhibitory[-1]
            '''
            pos_tmp_inh=np.zeros((pos.shape[0], 4))
            for i in range (pos.shape[0]):
                pos_tmp_inh[i, 0]=nest.GetNodes(SubSubRegion_Inhibitory[-1])[0][i]
                pos_tmp_inh[i, 1:] =pos[i, :3]
            #SubRegion_SetStatus(SubSubRegion_Inhibitory[-1], n_type_info)
            pos_inh=np.append(pos_inh, pos_tmp_inh, axis=0)
            np.savetxt("position_M1_" + M1_layer_Name[l] + "_" + n_type + ".csv", pos_tmp_inh, delimiter=",")
            '''
            Neuron_GIDs=None
            Neuron_GIDs = nest.GetNodes(SubSubRegion_Inhibitory[-1])[0]
            print ('Neuron_num:'+str(len(Neuron_GIDs)))
            nest.Connect(Neuron_GIDs, sd_list[-1])
            multimeter_inh.append(nest.Create("multimeter", params={"interval": 0.1, "record_from": ["V_m", "g_ex", "g_in"],
                                                      "withgid": True, "to_file": True,
                                                      "label": "mm_vLSPS_postsynaptic_cell_%s_%s" % (M1_layer_Name[l], n_type)}))
            print (ntop.FindCenterElement(SubSubRegion_Inhibitory[-1]))
            nest.Connect(multimeter_inh[-1], ntop.FindCenterElement(SubSubRegion_Inhibitory[-1]))
        
        else:
            print('Unknow E or I')
'''
with open('log/ctx.log', 'a+') as f:
    for pre_layer_name in ctx_layers.keys():
        count_in  = 0
        count_out = 0
        for post_layer_name in ctx_layers.keys():
            print ('from layer '+ pre_layer_name+' to layer '+post_layer_name+' '+str(ctx_layers_conn[pre_layer_name][post_layer_name]['conn_num']) + ' synapses were created', file=f)
            print ('synapse/neuron_num: '+ str(ctx_layers_conn[pre_layer_name][post_layer_name]['conn_num']/ctx_layers_conn[pre_layer_name][post_layer_name]['neuron_num']), file=f)
            count_out  = count_out + ctx_layers_conn[pre_layer_name][post_layer_name]['conn_num']/ctx_layers_conn[pre_layer_name][post_layer_name]['neuron_num']
            count_in = count_in + ctx_layers_conn[post_layer_name][pre_layer_name]['conn_num']/ctx_layers_conn[post_layer_name][pre_layer_name]['neuron_num']
        print('Layer ' + pre_layer_name + ' indegree : ' +str(count_in), file=f)
        print('Layer ' + pre_layer_name + ' outdegree : ' + str(count_out), file=f)
f.close()
'''
np.savetxt("pos_Exc.csv", pos_Exc, delimiter=",")
np.savetxt("pos_inh.csv", pos_inh, delimiter=",")

M1_internal_connection = np.load(ctx_M1_params ['M1']['connection_info']['M1toM1'])
from collections import defaultdict
ctx_layers_conn = defaultdict(dict)
for pre_layer_name in ctx_M1_layers.keys():
    for post_layer_name in ctx_M1_layers.keys():
        print ('start to connect '+pre_layer_name+' with '+post_layer_name)
        connect_layers_ctx_M1(ctx_M1_layers[pre_layer_name], ctx_M1_layers[post_layer_name], M1_internal_connection[pre_layer_name][post_layer_name])
        conn_num= len(nest.GetConnections(nest.GetNodes(ctx_M1_layers[pre_layer_name])[0], nest.GetNodes(ctx_M1_layers[post_layer_name])[0]))
        ctx_layers_conn[pre_layer_name][post_layer_name]={'conn_num':conn_num, 'neuron_num': len(nest.GetNodes(ctx_M1_layers[pre_layer_name])[0])}

grid_laser_index=0

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
