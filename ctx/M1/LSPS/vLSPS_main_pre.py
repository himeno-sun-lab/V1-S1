#!/usr/bin/env python
# -*- coding: utf-8 -*-

# This is the program to prepare necessary file to check LSPS on M1 layers


import nest
import numpy as np
import nest.topology as ntop
print (nest.version())
nest.ResetKernel()
import configparser
import shelve
from simu_fun import *
from vLSPS_fun import *
import pickle

nest.ResetKernel()

#read configure file
config = configparser.ConfigParser()
config.read('simu.conf')
local_num_threads=int(config['parallel_computing']['local_num_threads'])
if local_num_threads!=1:
    print ('Setlocal_num_threads')
    nest.SetKernelStatus({"local_num_threads": local_num_threads})

save_filename='BrainSimu.pickle'
# flag c means if the file doesn't exist, then create a new file
with open(save_filename, 'rb') as handle:
    pickle_file = pickle.load(handle)
handle.close()
simulate_region=pickle_file['cerebralcortex']['vS1']

vS1_Layer_Name = simulate_region['structure_info']['Layer_Name']
vS1_layer_size=simulate_region['structure_info']['region_size']
vS1_Layer_Thickness=simulate_region['structure_info']['layer_thickness']
vS1_Layer_Thickness_ST=np.cumsum(vS1_Layer_Thickness)
vS1_Layer_Thickness_ST=np.insert(vS1_Layer_Thickness_ST, 0, 0.0)

#config LSPS laser
stimu_duration = 1  # ms
stimu_interval = 400  # ms
#vLSPS_volume = 100  # micron
stim_amplitude=10000.
stim_start=100

vLSPS_grid_shape = [16, 16]
vLSPS_grid_size = [vS1_layer_size[0] / 16., vS1_layer_size[2] / 16.]
vLSPS_radii = 45 # micro
grid_laser=[]

vLSPS_grid_center_x=np.linspace(-vS1_layer_size[0]/2.,vS1_layer_size[0]/2. ,num=vLSPS_grid_shape[0]+1, endpoint=True)+vLSPS_grid_size[0]/2
vLSPS_grid_center_y=np.linspace(-vS1_layer_size[2]/2., vS1_layer_size[2]/2.,num=vLSPS_grid_shape[1]+1, endpoint=True)+vLSPS_grid_size[1]/2
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
vS1_Layer_Name = simulate_region['structure_info']['Layer_Name']
pos_Exc=np.zeros((0, 4))
pos_inh=np.zeros((0, 4))
sd_list=[]

multimeter_exc=[]
multimeter_inh=[]

topo_extend=np.array([1440., 400., 1366.])+1
topo_extend_tmp=topo_extend-1
topo_center=np.array([0., 0., 0.])

ctx_layers={}
############################################################################
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
################################################################################

record_gid=[]
for l in range(len(vS1_Layer_Name)):
    print('###########################################')
    print('start to create layer: ' + vS1_Layer_Name[l])
    Neuron_pos_fileload = np.load(simulate_region['position_type_info'][vS1_Layer_Name[l]])
    Neuron_pos = Neuron_pos_fileload['Neuron_pos']
    print('Network Architecture:')
    print(Neuron_pos.shape)
    #Neuron_Types = np.unique(Neuron_pos[:, :, :, 3])
    # print (Neuron_Types.shape[0])
    #n_type_index = 0
    print (simulate_region['neuro_info'][vS1_Layer_Name[l]].keys())
    z_center=(vS1_Layer_Thickness_ST[l + 1]-vS1_Layer_Thickness_ST[l])/2.+vS1_Layer_Thickness_ST[l]
    topo_center[2]=z_center-topo_extend_tmp[2]/2.
    for n_type in simulate_region['neuro_info'][vS1_Layer_Name[l]].keys():
        n_type_index=simulate_region['neuro_info'][vS1_Layer_Name[l]][n_type]['n_type_index']
        print('n_type_index:', n_type_index)
        print(n_type)
        n_type_info = simulate_region['neuro_info'][vS1_Layer_Name[l]][n_type]
        sd_list.append(nest.Create("spike_detector", params={'to_file': True, "label": "sd_vS1_%s_%s"%(vS1_Layer_Name[l], n_type), "use_gid_in_filename": True}))
        neuronmodel = copy_neuron_model(ctx_M1_params [region_name] ['neuro_info'] [M1_layer_Name [l]] [n_type] ['neuron_model'], n_type_info, region_name + '_' + M1_layer_Name [l] + '_' + n_type )
        #neuronmodel=simulate_region['neuro_info'][vS1_Layer_Name[l]][n_type]['neuron_model']
        elements=neuronmodel
        neuron_info=simulate_region['neuro_info'][vS1_Layer_Name[l]][n_type]
        if n_type_info['EorI'] == "E":
            pos=None
            pos = Neuron_pos[np.where(Neuron_pos[:, :, :, 3] == n_type_index)]
            nest.SetDefaults(elements, {"I_e": float(neuron_info['I_ex']), "V_th": float(neuron_info['spike_threshold']), "V_reset": float(neuron_info['reset_value']), "t_ref": float(neuron_info['absolute_refractory_period'])})
            newlayer = CreateSubSubRegion(topo_extend, topo_center, pos, neuronmodel )
            SubSubRegion_Excitatory.append(newlayer)
            ctx_layers[vS1_Layer_Name[l]+n_type] = SubSubRegion_Excitatory[-1]
            pos_tmp_exc=np.zeros((pos.shape[0],4))
            for i in range (pos.shape[0]):
                pos_tmp_exc[i, 0]=nest.GetNodes(SubSubRegion_Excitatory[-1])[0][i]
                pos_tmp_exc[i, 1:] =pos[i, :3]
            SubRegion_SetStatus(SubSubRegion_Excitatory[-1], n_type_info)
            pos_Exc=np.append(pos_Exc, pos_tmp_exc, axis=0)
            np.savetxt("position_vS1_" + vS1_Layer_Name[l]+"_"+n_type+".csv", pos_tmp_exc, delimiter=",")
            Neuron_GIDs=None
            Neuron_GIDs = nest.GetNodes(SubSubRegion_Excitatory[-1])[0]
            print ('Neuron_num:'+str(len(Neuron_GIDs)))
            nest.Connect(Neuron_GIDs, sd_list[-1])
            multimeter_exc.append(nest.Create("multimeter",
                                              params={"interval": 0.1, "record_from": ["V_m", "g_ex", "g_in"],
                                                      "withgid": True, "to_file": True,
                                                      "label": "mm_vLSPS_postsynaptic_cell_%s_%s" % (vS1_Layer_Name[l], n_type)}))
            print (ntop.FindCenterElement(SubSubRegion_Excitatory[-1]))
            nest.Connect(multimeter_exc[-1], ntop.FindCenterElement(SubSubRegion_Excitatory[-1]))
        elif n_type_info['EorI'] == "I":
            pos = None
            pos = Neuron_pos[np.where(Neuron_pos[:, :, :, 3] == n_type_index)]
            nest.SetDefaults(elements, {"I_e": float(neuron_info['I_ex']), "V_th": float(neuron_info['spike_threshold']), "V_reset": float(neuron_info['reset_value']), "t_ref": float(neuron_info['absolute_refractory_period'])})
            newlayer = CreateSubSubRegion(topo_extend, topo_center, pos, neuronmodel )
            SubSubRegion_Inhibitory.append(newlayer)
            ctx_layers[vS1_Layer_Name[l]+n_type] = SubSubRegion_Inhibitory[-1]
            pos_tmp_inh=np.zeros((pos.shape[0], 4))
            for i in range (pos.shape[0]):
                pos_tmp_inh[i, 0]=nest.GetNodes(SubSubRegion_Inhibitory[-1])[0][i]
                pos_tmp_inh[i, 1:] =pos[i, :3]
            SubRegion_SetStatus(SubSubRegion_Inhibitory[-1], n_type_info)
            pos_inh=np.append(pos_inh, pos_tmp_inh, axis=0)
            np.savetxt("position_vS1_" + vS1_Layer_Name[l] + "_" + n_type + ".csv", pos_tmp_inh, delimiter=",")
            Neuron_GIDs=None
            Neuron_GIDs = nest.GetNodes(SubSubRegion_Inhibitory[-1])[0]
            print ('Neuron_num:'+str(len(Neuron_GIDs)))
            nest.Connect(Neuron_GIDs, sd_list[-1])
            multimeter_inh.append(nest.Create("multimeter",
                                              params={"interval": 0.1, "record_from": ["V_m", "g_ex", "g_in"],
                                                      "withgid": True, "to_file": True,
                                                      "label": "mm_vLSPS_postsynaptic_cell_%s_%s" % (vS1_Layer_Name[l], n_type)}))
            print (ntop.FindCenterElement(SubSubRegion_Inhibitory[-1]))
            nest.Connect(multimeter_inh[-1], ntop.FindCenterElement(SubSubRegion_Inhibitory[-1]))
        
        else:
            print('Unknow E or I')
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
np.savetxt("pos_Exc.csv", pos_Exc, delimiter=",")
np.savetxt("pos_inh.csv", pos_inh, delimiter=",")

S1_internal_connection = np.load(simulate_region['connection_info']['vS1toVS1'])
from collections import defaultdict
ctx_layers_conn = defaultdict(dict)
for pre_layer_name in ctx_layers.keys():
    for post_layer_name in ctx_layers.keys():
        print ('start to connect '+pre_layer_name+' with '+post_layer_name)
        SubSubRegion_Connect(ctx_layers[pre_layer_name], ctx_layers[post_layer_name], S1_internal_connection[pre_layer_name][post_layer_name])
        conn_num= len(nest.GetConnections(nest.GetNodes(ctx_layers[pre_layer_name])[0], nest.GetNodes(ctx_layers[post_layer_name])[0]))
        ctx_layers_conn[pre_layer_name][post_layer_name]={'conn_num':conn_num, 'neuron_num': len(nest.GetNodes(ctx_layers[pre_layer_name])[0])}

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
