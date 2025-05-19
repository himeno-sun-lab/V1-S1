#!/usr/bin/env python
# -*- coding: utf-8 -*-

# This is the all program for simulation of "secondary motor cortex" including 6 layers
# eventhough L4 is somehow neglectable for M2

import fetch_params
import ini_all
import nest_routine
import nest
import nest.topology as ntop
import numpy as np
import time
from nest.lib.hl_api_info import SetStatus


########################################################################################################################
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
def read_ctx_M2():
    try:
        from runpy import run_path
        file_params = run_path('baseCTXM2Params.py', init_globals=globals())
        ctx_M2_params = file_params['ctxM2Params']
        return ctx_M2_params
    except:
        raise ImportError(
            'The cortex-M2-region parameters could not be loaded. Please make sure that the file `baseCTXM2Params.py` exists and is a valid python defining the variable "ctxM2Params".')

################################
def initialize_nest(sim_params):
  nest.set_verbosity("M_WARNING")
  nest.SetKernelStatus({"overwrite_files": sim_params['overwrite_files']})            # should we erase previous traces when redoing a simulation?
  nest.SetKernelStatus({'local_num_threads': int(sim_params['nbcpu'])})
  nest.SetKernelStatus({"data_path": 'log/'})
  if sim_params['dt'] != 0.1:
    nest.SetKernelStatus({'resolution': float(sim_params['dt'])})

######################################
def instantiate_ctx_M2(ctx_M2_params):
  start_time = time.time()
  # set the parameters for M2 model
  M2_Layer_Name = ctx_M2_params['M2']['structure_info']['Layer_Name']
  M2_layer_size = ctx_M2_params['M2']['structure_info']['region_size']
  M2_layer_size = np.array(M2_layer_size)
  topo_extend = M2_layer_size + 1.
  topo_center = M2_layer_size/2.
  SubSubRegion_Excitatory = []
  SubSubRegion_Inhibitory = []
  SubSubRegion_Excitatory_ntype = []
  SubSubRegion_Inhibitory_ntype = []
  ctx_M2_layers = {}
  for l in range(len(M2_Layer_Name)):
    print('###########################################')
    print('start to create layer: ' + M2_Layer_Name[l])
    #Neuron_pos_fileload = np.load('ctx/M2/'+ ctx_M2_params['M2']['position_type_info'][M2_Layer_Name[l]])
    Neuron_pos_fileload = np.load(ctx_M2_params['M2']['position_type_info'][M2_Layer_Name[l]])      # This calls the related 'npz' file of called layer and save it
    Neuron_pos = Neuron_pos_fileload['Neuron_pos']
    print('Network Architecture:')
    print(Neuron_pos.shape)
    print(ctx_M2_params['M2']['neuro_info'][M2_Layer_Name[l]].keys())
    for n_type in ctx_M2_params['M2']['neuro_info'][M2_Layer_Name[l]].keys():
      n_type_index = ctx_M2_params['M2']['neuro_info'][M2_Layer_Name[l]][n_type]['n_type_index']
      neuronmodel=ctx_M2_params['M2']['neuro_info'][M2_Layer_Name[l]][n_type]['neuron_model']
      print (M2_Layer_Name[l])
      print('n_type_index:', n_type_index)
      print(n_type)
      n_type_info = ctx_M2_params['M2']['neuro_info'][M2_Layer_Name[l]][n_type]
      if n_type_info['EorI'] == "E":
        pos = Neuron_pos[np.where(Neuron_pos[:, :, :, 3] == n_type_index)]
        SubSubRegion_Excitatory.append(create_layers_ctx_M2(topo_extend, topo_center, pos, neuronmodel, n_type_info))
        SubSubRegion_Excitatory_ntype.append([M2_Layer_Name[l], n_type])
        print (len(nest.GetNodes(SubSubRegion_Excitatory[-1])[0]))
        ctx_M2_layers[M2_Layer_Name[l]+n_type] = SubSubRegion_Excitatory[-1]
      elif n_type_info['EorI'] == "I":
        pos = Neuron_pos[np.where(Neuron_pos[:, :, :, 3] == n_type_index)]
        SubSubRegion_Inhibitory.append(create_layers_ctx_M2(topo_extend, topo_center, pos, neuronmodel, n_type_info))
        SubSubRegion_Inhibitory_ntype.append([M2_Layer_Name[l], n_type])
        print(len(nest.GetNodes(SubSubRegion_Inhibitory[-1])[0]))
        ctx_M2_layers[M2_Layer_Name[l]+n_type] = SubSubRegion_Inhibitory[-1]
      else:
        print('Error: Unknow E or I')
  #with open('./log/'+'performance.txt', 'a') as file:
    #file.write('instantiate_ctx(create_layers_ctx_M2) '+str(time.time()-start_time)+'\n')

  start_time=time.time()
  print("Start to connect the layers")
  #M2_internal_connection = np.load('ctx/M2/'+ ctx_M2_params['M2']['connection_info']['M2toM2'])
  ctx_M2_internal_connection = np.load(ctx_M2_params['M2']['connection_info']['M2toM2'])

  from collections import defaultdict
  ctx_M2_layers_conn = defaultdict(dict)
  for pre_layer_name in ctx_M2_layers.keys():
      for post_layer_name in ctx_M2_layers.keys():
          print ('start to connect '+pre_layer_name+' with '+post_layer_name)
          connect_layers_ctx_M2(ctx_M2_layers[pre_layer_name], ctx_M2_layers[post_layer_name], ctx_M2_internal_connection[pre_layer_name][post_layer_name])
          conn_num= len(nest.GetConnections(nest.GetNodes(ctx_M2_layers[pre_layer_name])[0], nest.GetNodes(ctx_M2_layers[post_layer_name])[0]))
          ctx_M2_layers_conn[pre_layer_name][post_layer_name]={'conn_num':conn_num, 'neuron_num': len(nest.GetNodes(ctx_M2_layers[pre_layer_name])[0])}

  with open('log/ctx.log', 'a+') as f:
      for pre_layer_name in ctx_M2_layers.keys():
          count_in  = 0
          count_out = 0
          for post_layer_name in ctx_M2_layers.keys():
              print ('from layer '+ pre_layer_name+' to layer '+post_layer_name+' '+str(ctx_M2_layers_conn[pre_layer_name][post_layer_name]['conn_num']) + ' synapses were created', file=f)
              print ('synapse/neuron_num: '+ str(ctx_M2_layers_conn[pre_layer_name][post_layer_name]['conn_num']/ctx_M2_layers_conn[pre_layer_name][post_layer_name]['neuron_num']), file=f)
              count_out  = count_out + ctx_M2_layers_conn[pre_layer_name][post_layer_name]['conn_num']/ctx_M2_layers_conn[pre_layer_name][post_layer_name]['neuron_num']
              count_in = count_in + ctx_M2_layers_conn[post_layer_name][pre_layer_name]['conn_num']/ctx_M2_layers_conn[post_layer_name][pre_layer_name]['neuron_num']
              print('Layer ' + pre_layer_name + ' indegree : ' +str(count_in), file=f)
              print('Layer ' + pre_layer_name + ' outdegree : ' + str(count_out), file=f)
  f.close()
  #with open('./log/'+'performance.txt', 'a') as file:
      #file.write('instantiate_ctx(connect_layers_ctx_M2) '+str(time.time()-start_time)+'\n')
  return ctx_M2_layers


###########################################################################
def create_layers_ctx_M2(extent, center, positions, elements, neuron_info):
    Neuron_pos_list = positions[:, :3].tolist()
    nest.SetDefaults(elements, {"I_e": float(neuron_info['I_ex']), "V_th": float(neuron_info['spike_threshold']),
                                "V_reset": float(neuron_info['reset_value']),
                                "t_ref": float(neuron_info['absolute_refractory_period'])})
    newlayer = ntop.CreateLayer(
        {'extent': extent, 'center': center, 'positions': Neuron_pos_list, 'elements': elements})
    # Neurons = nest.GetNodes(newlayer)
    return newlayer


##########################################################################
def connect_layers_ctx_M2(pre_SubSubRegion, post_SubSubRegion, conn_dict):
    sigma_x = conn_dict['sigma'] / 1000.
    sigma_y = conn_dict['sigma'] / 1000.
    weight_distribution = conn_dict['weight_distribution']
    if weight_distribution == 'lognormal':
        conndict = {'connection_type': 'divergent',
                    'mask': {'spherical': {'radius': 2.0}},
                    'kernel': {
                        'gaussian2D': {'p_center': conn_dict['p_center'], 'sigma_x': sigma_x, 'sigma_y': sigma_y}},
                    'weights': {'lognormal': {'mu': conn_dict['weight'], 'sigma': 1.0}},
                    'delays': conn_dict['delay'],
                    'allow_autapses': False,
                    'allow_multapses': True}
    else:
        conndict = {'connection_type': 'divergent',
                    'mask': {'spherical': {'radius': 2.0}},
                    'kernel': {
                        'gaussian2D': {'p_center': conn_dict['p_center'], 'sigma_x': sigma_x, 'sigma_y': sigma_y}},
                    'weights': conn_dict['weight'],
                    'delays': conn_dict['delay'],
                    'allow_autapses': False,
                    'allow_multapses': True}
    if sigma_x != 0:
        ntop.ConnectLayers(pre_SubSubRegion, post_SubSubRegion, conndict)


################################################################################
#-------------------------------------------------------------------------------
# Instantiate a spike detector and connects it to the entire layer `layer_gid`
#-------------------------------------------------------------------------------
def layer_spike_detector(layer_gid, layer_name, params={"withgid": True, "withtime": True, "to_file": True}):
  print ('spike detector for '+layer_name)
  params.update({'label': layer_name})
  detector = nest.Create("spike_detector", params=params)
  nest.Connect(pre=nest.GetNodes(layer_gid)[0], post=detector)
  return detector


################################################################################
#-------------------------------------------------------------------------------
# Starts the Nest simulation, given the general parameters of `sim_params`
#-------------------------------------------------------------------------------
def run_simulation(sim_params):
  nest.ResetNetwork()
  nest.Simulate(sim_params['simDuration'])


#####################################################################################
#------------------------------------------------------------------------------------
# Returns the average firing rate of a population
# It is relative to the simulation duration `simDuration` and the population size `n`
#------------------------------------------------------------------------------------
def average_fr(detector, simDuration, n):
  return nest.GetStatus(detector, 'n_events')[0] / float(simDuration) / float(n) * 1000

################################################################################
#-------------------------------------------------------------------------------
# Returns the number of neurons inside a layer
#-------------------------------------------------------------------------------
def count_layer(layer_gid):
    return len(nest.GetNodes(layer_gid)[0])

################################################################################
#-------------------------------------------------------------------------------
# Returns the positions of neurons inside a layer -sun-20180911
#-------------------------------------------------------------------------------
def get_position(layer_gid):
    return ntop.GetPosition(layer_gid)

################################################################################
#-------------------------------------------------------------------------------
# Returns the connections of neurons inside a layer -sun-20180912
#-------------------------------------------------------------------------------
def get_connection(gids):
    return nest.GetConnections(gids)

#################################################################################
# -------------------------------------------------------------------------------
#Generator for efficient looping over local nodes
#Assumes nodes is a continous list of gids [1, 2, 3, ...], e.g., as
#returned by Create. Only works for nodes with proxies, i.e.,
#regular neurons.
# -------------------------------------------------------------------------------
def get_local_nodes(nodes):
    nvp = nest.GetKernelStatus('total_num_virtual_procs')  # step size
    i = 0
    print (nodes)
    while i < len(nodes):
        if nest.GetStatus([nodes[i]], 'local')[0]:
            yield nodes[i]
            i += nvp
        else:
            i += 1
########################################################################################################################
####################
#  making load files  #
####################
# making the npz files to restore the position information
import matplotlib.pyplot as plt
import matplotlib as mplb
import numpy as np
import pandas as pd
import math
import os

# read the information from Hook's paper data
M2_layer_size=np.array([1000, 1000, 1527])/1000. # micron
M2_Layer_Thickness=np.array([160.0, 34.0, 69.0, 244.0, 510.0, 510.0])/1000.
#M2_Layer_Thickness_ST=np.cumsum(M2_Layer_Thickness)
M2_Layer_Thickness_ST=np.insert(M2_Layer_Thickness, 0, 0.0)
print (M2_Layer_Thickness_ST)
M2_Density_csv=pd.read_csv('/home/morteza/Desktop/ConnectionMatrix/M2/Ueta_13/cell_density/M2.dat')
M2_Layer_Cellcont_mm2=np.squeeze(M2_Density_csv.values)[:-1]

M2_Layer_Density= [math.pow(M2_Layer_Cellcont_mm2[0]/(M2_Layer_Thickness[0]), 1.0/3), \
                   math.pow(M2_Layer_Cellcont_mm2[1]/(M2_Layer_Thickness[1]), 1.0/3), \
                   math.pow(M2_Layer_Cellcont_mm2[2]/(M2_Layer_Thickness[2]), 1.0/3),\
                   math.pow(M2_Layer_Cellcont_mm2[3]/(M2_Layer_Thickness[3]), 1.0/3),\
                   math.pow(M2_Layer_Cellcont_mm2[4]/(M2_Layer_Thickness[4]), 1.0/3),\
                   math.pow(M2_Layer_Cellcont_mm2[5]/(M2_Layer_Thickness[5]), 1.0/3),]

print (M2_Layer_Density)

Sub_Region_Architecture=np.zeros((len(M2_Layer_Density), 3))
for i in range (len(M2_Layer_Density)):
    Sub_Region_Architecture[i, :]=M2_Layer_Density[i]

Sub_Region_Architecture[:, 0]=np.round(Sub_Region_Architecture[:, 0]*M2_layer_size[0])
Sub_Region_Architecture[:, 1]=np.round(Sub_Region_Architecture[:, 1]*M2_layer_size[1])
Sub_Region_Architecture[:, 2]=np.round(Sub_Region_Architecture[:, 2]*M2_Layer_Thickness)
Sub_Region_Architecture=Sub_Region_Architecture.astype(np.int64)
print (Sub_Region_Architecture)

Layer_name=['L1', 'L23A', 'L23B', 'L5A', 'L5B', 'L6']

Layer_Neuron_Architecture=[[['IN_SBC', 'IN_ENGC'], [0.203,0.797]],
                           [['EN_CC', 'IN_FS', 'IN_LTS', 'IN_VIP'], [0.83, 0.0490926, 0.0196299, 0.1012775]],
                           [['EN_CC', 'IN_FS', 'IN_LTS','IN_VIP'], [0.91, 0.0259902, 0.0103923, 0.0536175]],
                           [['EN_CS', 'EN_CC' , 'EN_CT' , 'IN_FS', 'IN_LTS'], [0.307309, 0.307309, 0.307309, 0.01490311, 0.06316989]],
                           [['EN_CS', 'EN_CC' , 'EN_PT' , 'IN_FS', 'IN_LTS'], [0.287073, 0.287073, 0.287073, 0.07508246, 0.06369854]],
                           [['EN_CT', 'IN_FS', 'IN_LTS'], [0.859, 0.08, 0.061]],
                      ]

Sub_Region_L1={"Name": 'L1',
               "Architecture": Sub_Region_Architecture[0],
               "Size": [M2_layer_size[0], M2_layer_size[1], M2_Layer_Thickness[0]]
              }
Sub_Region_L23A={"Name": 'L23A',
               "Architecture": Sub_Region_Architecture[1],
               "Size": [M2_layer_size[0], M2_layer_size[1],  M2_Layer_Thickness[1]]
              }
Sub_Region_L23B={"Name": 'L23B',
               "Architecture": Sub_Region_Architecture[2],
               "Size": [M2_layer_size[0], M2_layer_size[1],  M2_Layer_Thickness[2]]
              }
Sub_Region_L5A={"Name": 'L5A',
               "Architecture":  Sub_Region_Architecture[3],
               "Size": [M2_layer_size[0], M2_layer_size[1],  M2_Layer_Thickness[3]]
              }
Sub_Region_L5B={"Name": 'L5B',
               "Architecture":  Sub_Region_Architecture[4],
               "Size": [M2_layer_size[0], M2_layer_size[1],  M2_Layer_Thickness[4]]
              }
Sub_Region_L6={"Name": 'L6',
               "Architecture":  Sub_Region_Architecture[5],
               "Size": [M2_layer_size[0], M2_layer_size[1],  M2_Layer_Thickness[5]]
              }

Sub_Region_list=[]
Sub_Region_list.append(Sub_Region_L1)
Sub_Region_list.append(Sub_Region_L23A)
Sub_Region_list.append(Sub_Region_L23B)
Sub_Region_list.append(Sub_Region_L5A)
Sub_Region_list.append(Sub_Region_L5B)
Sub_Region_list.append(Sub_Region_L6)

for l in range(len(Layer_name)):
    Sub_Region = Sub_Region_list[l]
    print('start to preprocess layer: ' + Sub_Region['Name'])
    Layer_architecture = Sub_Region['Architecture']
    Layer_size = Sub_Region['Size']
    print(Layer_architecture)
    Neuron_numbers = Layer_architecture[0] * Layer_architecture[1] * Layer_architecture[2]

    Neuron_pos_x = np.linspace(0.0, Layer_size[0], num=Layer_architecture[0], endpoint=True)
    Neuron_pos_y = np.linspace(0.0, Layer_size[1], num=Layer_architecture[1], endpoint=True)
    Neuron_pos_z = np.linspace(M2_Layer_Thickness_ST[l], M2_Layer_Thickness_ST[l + 1], num=Layer_architecture[2], endpoint=False)
    Neuron_pos = np.zeros((Layer_architecture[0], Layer_architecture[1], Layer_architecture[2], 4))
    print(Layer_Neuron_Architecture[l][1])
    Neuron_type = np.random.choice(range(len(Layer_Neuron_Architecture[l][0])), size=Sub_Region_Architecture[l],
                                   p=Layer_Neuron_Architecture[l][1])
    py_index = 0
    for i in range(Layer_architecture[0]):
        for j in range(Layer_architecture[1]):
            for k in range(Layer_architecture[2]):
                # print (i, j, k)
                # print (L1_architecture[1]*i+L1_architecture[2]*j+k)
                Neuron_pos[i, j, k, :3] = np.array([Neuron_pos_x[i], Neuron_pos_y[j], Neuron_pos_z[k]])
                Neuron_pos[i, j, k, 3] = Neuron_type[i, j, k]
    print(Neuron_pos)
    s_path = "/home/morteza/PycharmProjects/postk_wb/code/ctx/M2"
    np.savez(os.path.join(s_path, 'M2_Neuron_pos_' + Layer_name[l]), Neuron_pos=Neuron_pos)





########################################################################################################################
#################
#  main script  #
#################
print('Reading the simulation parameters using "baseSimParameters.py" file')
sim_params = read_sim()
print('Reading the M2 parameters using "baseCTXM2Parameters.py" file')
ctx_M2_params = read_ctx_M2()
print('Nest Initializations')
initialize_nest(sim_params)
print('create and connect the M2 layers')
ctx_M2_layers = instantiate_ctx_M2(ctx_M2_params)

# define detectors
print('start to define the detectors')
detectors = {}
for layer_name in ctx_M2_layers.keys():
    detectors[layer_name] = layer_spike_detector(ctx_M2_layers[layer_name], layer_name)

print('Starting the simulation...')
run_simulation(sim_params)

print('Simulation  in brief:')
for layer_name in ctx_M2_layers.keys():
  rate = average_fr(detectors[layer_name], sim_params['simDuration'], count_layer(ctx_M2_layers[layer_name]))
  print('Layer ' + layer_name + " fires at " + str(rate) + " Hz")


'''

if __name__ == '__main__':
    main()
'''