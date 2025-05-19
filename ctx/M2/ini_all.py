#!/usr/bin/env python
# -*- coding: utf-8 -*-

##
## ini_all.py
##
## This contains region-specific initialization functions, based on provided parameters.

# nest_routines contains all nest-specific calls
import numpy as np
import nest_routine
import nest
import time
# the functions below contain the logics of the instantiation, without any direct call to nest
def instantiate_ctx(ctx_params):
  start_time = time.time()
  # set the parameters for S1 model
  vS1_Layer_Name = ctx_params['vS1']['structure_info']['Layer_Name']
  vS1_layer_size = ctx_params['vS1']['structure_info']['region_size']
  vS1_layer_size=np.array(vS1_layer_size)
  topo_extend = vS1_layer_size + 1.
  topo_center = vS1_layer_size/2.
  SubSubRegion_Excitatory = []
  SubSubRegion_Inhibitory = []
  SubSubRegion_Excitatory_ntype = []
  SubSubRegion_Inhibitory_ntype = []
  ctx_layers = {}
  for l in range(len(vS1_Layer_Name)):
    print('###########################################')
    print('start to create layer: ' + vS1_Layer_Name[l])
    Neuron_pos_fileload = np.load('ctx/'+ ctx_params['vS1']['position_type_info'][vS1_Layer_Name[l]])
    Neuron_pos = Neuron_pos_fileload['Neuron_pos']
    print('Network Architecture:')
    print(Neuron_pos.shape)
    print(ctx_params['vS1']['neuro_info'][vS1_Layer_Name[l]].keys())
    for n_type in ctx_params['vS1']['neuro_info'][vS1_Layer_Name[l]].keys():
      n_type_index = ctx_params['vS1']['neuro_info'][vS1_Layer_Name[l]][n_type]['n_type_index']
      neuronmodel=ctx_params['vS1']['neuro_info'][vS1_Layer_Name[l]][n_type]['neuron_model']
      print (vS1_Layer_Name[l])
      print('n_type_index:', n_type_index)
      print(n_type)
      n_type_info = ctx_params['vS1']['neuro_info'][vS1_Layer_Name[l]][n_type]
      if n_type_info['EorI'] == "E":
        pos = Neuron_pos[np.where(Neuron_pos[:, :, :, 3] == n_type_index)]
        SubSubRegion_Excitatory.append(nest_routine.create_layers_ctx(topo_extend, topo_center, pos, neuronmodel, n_type_info))
        SubSubRegion_Excitatory_ntype.append([vS1_Layer_Name[l], n_type])
        print (len(nest.GetNodes(SubSubRegion_Excitatory[-1])[0]))
        ctx_layers[vS1_Layer_Name[l]+n_type] = SubSubRegion_Excitatory[-1]
      elif n_type_info['EorI'] == "I":
        pos = Neuron_pos[np.where(Neuron_pos[:, :, :, 3] == n_type_index)]
        SubSubRegion_Inhibitory.append(nest_routine.create_layers_ctx(topo_extend, topo_center, pos, neuronmodel, n_type_info))
        SubSubRegion_Inhibitory_ntype.append([vS1_Layer_Name[l], n_type])
        print(len(nest.GetNodes(SubSubRegion_Inhibitory[-1])[0]))
        ctx_layers[vS1_Layer_Name[l]+n_type] = SubSubRegion_Inhibitory[-1]
      else:
        print('Error: Unknow E or I')
  with open('./log/'+'performance.txt', 'a') as file:
    file.write('instantiate_ctx(create_layers_ctx) '+str(time.time()-start_time)+'\n')

  # make connections
  start_time=time.time()
  print("Start to connect the layers")
  # S1_internal_connection = np.load('ctx/'+ ctx_params['vS1']['connection_info']['vS1toVS1'])
  #
  # from collections import defaultdict
  # ctx_layers_conn = defaultdict(dict)
  # for pre_layer_name in ctx_layers.keys():
  #   for post_layer_name in ctx_layers.keys():
  #     print ('start to connect '+pre_layer_name+' with '+post_layer_name)
  #     nest_routine.connect_layers_ctx(ctx_layers[pre_layer_name], ctx_layers[post_layer_name], S1_internal_connection[pre_layer_name][post_layer_name])
  #     conn_num= len(nest.GetConnections(nest.GetNodes(ctx_layers[pre_layer_name])[0], nest.GetNodes(ctx_layers[post_layer_name])[0]))
  #     ctx_layers_conn[pre_layer_name][post_layer_name]={'conn_num':conn_num, 'neuron_num': len(nest.GetNodes(ctx_layers[pre_layer_name])[0])}
  #
  # with open('log/ctx.log', 'a+') as f:
  #     for pre_layer_name in ctx_layers.keys():
  #       count_in  = 0
  #       count_out = 0
  #       for post_layer_name in ctx_layers.keys():
  #         print ('from layer '+ pre_layer_name+' to layer '+post_layer_name+' '+str(ctx_layers_conn[pre_layer_name][post_layer_name]['conn_num']) + ' synapses were created', file=f)
  #         print ('synapse/neuron_num: '+ str(ctx_layers_conn[pre_layer_name][post_layer_name]['conn_num']/ctx_layers_conn[pre_layer_name][post_layer_name]['neuron_num']), file=f)
  #         count_out  = count_out + ctx_layers_conn[pre_layer_name][post_layer_name]['conn_num']/ctx_layers_conn[pre_layer_name][post_layer_name]['neuron_num']
  #         count_in = count_in + ctx_layers_conn[post_layer_name][pre_layer_name]['conn_num']/ctx_layers_conn[post_layer_name][pre_layer_name]['neuron_num']
  #       print('Layer ' + pre_layer_name + ' indegree : ' +str(count_in), file=f)
  #       print('Layer ' + pre_layer_name + ' outdegree : ' + str(count_out), file=f)
  # f.close()
  with open('./log/'+'performance.txt', 'a') as file:
    file.write('instantiate_ctx(connect_layers_ctx) '+str(time.time()-start_time)+'\n')
  return ctx_layers

'''
def instantiate_ctx_M1(ctx_M1_params):
  start_time = time.time()
  # set the parameters for M1 model
  M1_Layer_Name = ctx_M1_params['M1']['structure_info']['Layer_Name']
  M1_layer_size = ctx_M1_params['M1']['structure_info']['region_size']
  M1_layer_size = np.array(M1_layer_size)
  topo_extend = M1_layer_size + 1.
  topo_center = M1_layer_size/2.
  SubSubRegion_Excitatory = []
  SubSubRegion_Inhibitory = []
  SubSubRegion_Excitatory_ntype = []
  SubSubRegion_Inhibitory_ntype = []
  ctx_M1_layers = {}
  for l in range(len(M1_Layer_Name)):
    print('###########################################')
    print('start to create layer: ' + M1_Layer_Name[l])
    Neuron_pos_fileload = np.load('ctx/M1/'+ ctx_M1_params['M1']['position_type_info'][M1_Layer_Name[l]])
    Neuron_pos = Neuron_pos_fileload['Neuron_pos']
    print('Network Architecture:')
    print(Neuron_pos.shape)
    print(ctx_M1_params['M1']['neuro_info'][M1_Layer_Name[l]].keys())
    for n_type in ctx_M1_params['M1']['neuro_info'][M1_Layer_Name[l]].keys():
      n_type_index = ctx_M1_params['M1']['neuro_info'][M1_Layer_Name[l]][n_type]['n_type_index']
      neuronmodel=ctx_M1_params['M1']['neuro_info'][M1_Layer_Name[l]][n_type]['neuron_model']
      print (M1_Layer_Name[l])
      print('n_type_index:', n_type_index)
      print(n_type)
      n_type_info = ctx_M1_params['M1']['neuro_info'][M1_Layer_Name[l]][n_type]
      if n_type_info['EorI'] == "E":
        pos = Neuron_pos[np.where(Neuron_pos[:, :, :, 3] == n_type_index)]
        SubSubRegion_Excitatory.append(nest_routine.create_layers_ctx_M1(topo_extend, topo_center, pos, neuronmodel, n_type_info))
        SubSubRegion_Excitatory_ntype.append([M1_Layer_Name[l], n_type])
        print (len(nest.GetNodes(SubSubRegion_Excitatory[-1])[0]))
        ctx_M1_layers[M1_Layer_Name[l]+n_type] = SubSubRegion_Excitatory[-1]
      elif n_type_info['EorI'] == "I":
        pos = Neuron_pos[np.where(Neuron_pos[:, :, :, 3] == n_type_index)]
        SubSubRegion_Inhibitory.append(nest_routine.create_layers_ctx_M1(topo_extend, topo_center, pos, neuronmodel, n_type_info))
        SubSubRegion_Inhibitory_ntype.append([M1_Layer_Name[l], n_type])
        print(len(nest.GetNodes(SubSubRegion_Inhibitory[-1])[0]))
        ctx_M1_layers[M1_Layer_Name[l]+n_type] = SubSubRegion_Inhibitory[-1]
      else:
        print('Error: Unknow E or I')
  with open('./log/'+'performance.txt', 'a') as file:
    file.write('instantiate_ctx(create_layers_ctx_M1) '+str(time.time()-start_time)+'\n')

  start_time=time.time()
  print("Start to connect the layers")
  M1_internal_connection = np.load('ctx/M1/'+ ctx_M1_params['M1']['connection_info']['M1toM1'])

  from collections import defaultdict
  ctx_M1_layers_conn = defaultdict(dict)
  for pre_layer_name in ctx_M1_layers.keys():
      for post_layer_name in ctx_M1_layers.keys():
          print ('start to connect '+pre_layer_name+' with '+post_layer_name)
          connect_layers_ctx_M1(ctx_M1_layers[pre_layer_name], ctx_M1_layers[post_layer_name], M1_internal_connection[pre_layer_name][post_layer_name])
          conn_num= len(nest.GetConnections(nest.GetNodes(ctx_M1_layers[pre_layer_name])[0], nest.GetNodes(ctx_M1_layers[post_layer_name])[0]))
          ctx_M1_layers_conn[pre_layer_name][post_layer_name]={'conn_num':conn_num, 'neuron_num': len(nest.GetNodes(ctx_M1_layers[pre_layer_name])[0])}

  with open('log/ctx.log', 'a+') as f:
      for pre_layer_name in ctx_M1_layers.keys():
          count_in  = 0
          count_out = 0
          for post_layer_name in ctx_M1_layers.keys():
              print ('from layer '+ pre_layer_name+' to layer '+post_layer_name+' '+str(ctx_M1_layers_conn[pre_layer_name][post_layer_name]['conn_num']) + ' synapses were created', file=f)
              print ('synapse/neuron_num: '+ str(ctx_M1_layers_conn[pre_layer_name][post_layer_name]['conn_num']/ctx_M1_layers_conn[pre_layer_name][post_layer_name]['neuron_num']), file=f)
              count_out  = count_out + ctx_M1_layers_conn[pre_layer_name][post_layer_name]['conn_num']/ctx_M1_layers_conn[pre_layer_name][post_layer_name]['neuron_num']
              count_in = count_in + ctx_M1_layers_conn[post_layer_name][pre_layer_name]['conn_num']/ctx_M1_layers_conn[post_layer_name][pre_layer_name]['neuron_num']
              print('Layer ' + pre_layer_name + ' indegree : ' +str(count_in), file=f)
              print('Layer ' + pre_layer_name + ' outdegree : ' + str(count_out), file=f)
  f.close()
  with open('./log/'+'performance.txt', 'a') as file:
    file.write('instantiate_ctx(connect_layers_ctx_M1) '+str(time.time()-start_time)+'\n')
  return ctx_M1_layers



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
    Neuron_pos_fileload = np.load('ctx/M2/'+ ctx_M2_params['M2']['position_type_info'][M2_Layer_Name[l]])
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
        SubSubRegion_Excitatory.append(nest_routine.create_layers_ctx_M2(topo_extend, topo_center, pos, neuronmodel, n_type_info))
        SubSubRegion_Excitatory_ntype.append([M2_Layer_Name[l], n_type])
        print (len(nest.GetNodes(SubSubRegion_Excitatory[-1])[0]))
        ctx_M2_layers[M2_Layer_Name[l]+n_type] = SubSubRegion_Excitatory[-1]
      elif n_type_info['EorI'] == "I":
        pos = Neuron_pos[np.where(Neuron_pos[:, :, :, 3] == n_type_index)]
        SubSubRegion_Inhibitory.append(nest_routine.create_layers_ctx_M2(topo_extend, topo_center, pos, neuronmodel, n_type_info))
        SubSubRegion_Inhibitory_ntype.append([M2_Layer_Name[l], n_type])
        print(len(nest.GetNodes(SubSubRegion_Inhibitory[-1])[0]))
        ctx_M2_layers[M2_Layer_Name[l]+n_type] = SubSubRegion_Inhibitory[-1]
      else:
        print('Error: Unknow E or I')
  with open('./log/'+'performance.txt', 'a') as file:
    file.write('instantiate_ctx(create_layers_ctx_M2) '+str(time.time()-start_time)+'\n')

  start_time=time.time()
  print("Start to connect the layers")
  M2_internal_connection = np.load('ctx/M2/'+ ctx_M2_params['M2']['connection_info']['M2toM2'])

  from collections import defaultdict
  ctx_M2_layers_conn = defaultdict(dict)
  for pre_layer_name in ctx_M2_layers.keys():
      for post_layer_name in ctx_M2_layers.keys():
          print ('start to connect '+pre_layer_name+' with '+post_layer_name)
          connect_layers_ctx_M2(ctx_M2_layers[pre_layer_name], ctx_M2_layers[post_layer_name], M2_internal_connection[pre_layer_name][post_layer_name])
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
  with open('./log/'+'performance.txt', 'a') as file:
    file.write('instantiate_ctx(connect_layers_ctx_M2) '+str(time.time()-start_time)+'\n')
  return ctx_M2_layers

'''



def instantiate_th(th_params):
  start_time=time.time()
  # set the parameters for Th model
  th_layers={}
  for rg in th_params.keys():
    th_layers[rg]={}
    subregion_name = th_params[rg]['structure_info']['subregion_name']
    subregion_size = th_params[rg]['structure_info']['region_size']
    subregion_size=np.array(subregion_size)
    topo_extend = subregion_size + 1.
    topo_center = [0., 0., 0.]
    SubSubRegion_Excitatory = []
    SubSubRegion_Inhibitory = []
    SubSubRegion_Excitatory_ntype = []
    SubSubRegion_Inhibitory_ntype = []

    n_one_side_th_neurons = 40
    Neuron_pos_x = np.linspace(-subregion_size[0]/2., subregion_size[0]/2., n_one_side_th_neurons, endpoint=True)
    Neuron_pos_y = np.linspace(-subregion_size[1]/2., subregion_size[1]/2., n_one_side_th_neurons, endpoint=True)
    Neuron_pos_list = []

    for i in range(n_one_side_th_neurons):
      for j in range(n_one_side_th_neurons):
        Neuron_pos_list.append([Neuron_pos_x[i], Neuron_pos_y[j], 0.])

    for l in range(len(subregion_name)):
      print('###########################################')
      print('start to create subregions: ' + subregion_name[l])
      print(th_params[rg]['neuro_info'][subregion_name[l]].keys())
      for n_type in th_params[rg]['neuro_info'][subregion_name[l]].keys():
        n_type_index = th_params[rg]['neuro_info'][subregion_name[l]][n_type]['n_type_index']
        neuronmodel=th_params[rg]['neuro_info'][subregion_name[l]][n_type]['neuron_model']
        print('n_type_index:', n_type_index)
        print(n_type)
        n_type_info = th_params[rg]['neuro_info'][subregion_name[l]][n_type]
        if n_type_info['EorI'] == "E":
          SubSubRegion_Excitatory.append(nest_routine.create_layers_th(topo_extend, topo_center, Neuron_pos_list, neuronmodel, n_type_info))
          SubSubRegion_Excitatory_ntype.append([subregion_name[l], n_type])
          th_layers[rg][subregion_name[l]+n_type] = SubSubRegion_Excitatory[-1]
        elif n_type_info['EorI'] == "I":
          SubSubRegion_Inhibitory.append(nest_routine.create_layers_th(topo_extend, topo_center, Neuron_pos_list, neuronmodel, n_type_info))
          SubSubRegion_Inhibitory_ntype.append([subregion_name[l], n_type])
          th_layers[rg][subregion_name[l]+n_type] = SubSubRegion_Inhibitory[-1]
        else:
          print('Error: Unknow E or I')

    with open('./log/'+'performance.txt', 'a') as file:
      file.write('instantiate_th(create_layers_th) '+str(time.time()-start_time)+'\n')

    # make connections
    start_time=time.time()
    print("Start to connect the neurons")
    connection_info=th_params[rg]['connection_info']
    for pre_l in th_layers[rg].keys():
      for post_l in th_layers[rg].keys():
        print('start to connect ' + pre_l + ' with ' + post_l)
        nest_routine.connect_layers_th(th_layers[rg][pre_l], th_layers[rg][post_l], connection_info[pre_l][post_l])

  with open('./log/'+'performance.txt', 'a') as file:
    file.write('instantiate_th(connect_layers_th) '+str(time.time()-start_time)+'\n')
  return th_layers

def instantiate_bg(bg_params, fake_inputs=False, ctx_inputs=None):
  start_time = time.time()
  bg_layers = {}
  print('###########################################')
  print("BG instantiation")
  for nucleus in ['MSN','FSI','STN','GPe','GPi']:
    print("Creating nucleus "+nucleus+"...")
    bg_layers[nucleus] = nest_routine.create_layers_bg(bg_params, nucleus)
  if fake_inputs:
    for fake_nucleus in ['CSN','PTN','CMPf']:
      rate = bg_params['normalrate'][fake_nucleus][0]
      if ctx_inputs != None and fake_nucleus in ['CSN','PTN']:
        print('special handling of CSN/PTN input layer => 1,000 less Poisson Generators will be created')
        mirror_neurons = ctx_inputs[fake_nucleus]
      else:
        mirror_neurons = None
      print("Creating fake input "+fake_nucleus+" with fire rate "+str(rate)+" Hz...")
      bg_layers[fake_nucleus] = nest_routine.create_layers_bg(bg_params, fake_nucleus, fake=rate, mirror_neurons=mirror_neurons)
      if ctx_inputs != None and fake_nucleus in ['CSN','PTN']:
        print('special handling of CSN/PTN input layer => the remaining neurons will be connected to the original ctx neurons')
        nest_routine.connect_ctx_bg(mirror_neurons, bg_layers[fake_nucleus])
  with open('./log/'+'performance.txt', 'a') as file:
    file.write('instantiate_bg(create_layers_bg,connect_ctx_bg) '+str(time.time()-start_time)+'\n')

  start_time=time.time()
  print("BG connect layers")
  for connection in bg_params['alpha']:
    src = connection[0:3]
    tgt = connection[-3:]
    if src == 'CMP':
      src = 'CMPf' # string split on '->' would be better
    # if not fake_inputs, wire only in case of intra-BG connection
    if fake_inputs or src in ['MSN','FSI','STN','GPe','GPi']:
      if src in ['MSN','FSI','GPe','GPi']:
        nType = 'in'
      else:
        nType = 'ex'
      nest_routine.connect_layers_bg(bg_params, nType, bg_layers, src, tgt, projType=bg_params['cType'+src+tgt], redundancy=bg_params['redundancy'+src+tgt], RedundancyType=bg_params['RedundancyType'],verbose=True)
  with open('./log/'+'performance.txt', 'a') as file:
    file.write('instantiate_bg(connect_layers_bg) '+str(time.time()-start_time)+'\n')
  return bg_layers
    

def instantiate_cb():
  start_time=time.time()
  import CBnetwork
  import CBneurons
  # define all neurons
  CBneurons.create_neurons()
  # define all layers
  print("start to create CB layers")
  layer_gr = nest_routine.create_layers_cb(320, 320, elements=['GR'], extent=[1., 1. , 1.], center= [0., 0., 0.])
  layer_go = nest_routine.create_layers_cb(32, 32, elements=['GO'], extent=[1., 1. , 1.], center= [0., 0., 0.])
  layer_pkj = nest_routine.create_layers_cb(1, 16, elements=['PKJ'], extent=[0.5, 0.5, 0.5], center= [0., 0., 0.])
  layer_bs = nest_routine.create_layers_cb(1, 16, elements=['BS'], extent=[0.5, 0.5, 0.5], center= [0., 0., 0.])
  layer_vn = nest_routine.create_layers_cb(32, 32, elements=['VN'], extent=[1., 1. , 1.], center= [0., 0., 0.])
  layer_io = nest_routine.create_layers_cb(1, 1, elements=['IO'], extent=[1., 1. , 1.], center= [0., 0., 0.])
  #layer_io_input = nest_routine.create_layers_cb(1, 1, elements=['MF_IO'])

  cb_layers = {}
  cb_layers['layer_gr'] = layer_gr
  cb_layers['layer_go'] = layer_go
  cb_layers['layer_pkj'] = layer_pkj
  cb_layers['layer_bs'] = layer_bs
  cb_layers['layer_vn'] = layer_vn
  #cb_layers['layer_io'] = layer_io

  # connect layers
  # print("start to connect CB layers")
  # CBnetwork.go.go_to_gr(layer_go, layer_gr)
  # CBnetwork.gr.gr_to_go(layer_gr, layer_go)
  # CBnetwork.gr.gr_to_pkj(layer_gr, layer_pkj)
  # CBnetwork.gr.gr_to_bs(layer_gr, layer_bs)
  # CBnetwork.pkj.pkj_to_vn(layer_pkj, layer_vn)
  # CBnetwork.bs.bs_to_pkj(layer_bs, layer_pkj)


  with open('./log/'+'performance.txt', 'a') as file:
    file.write('instantiate_cb(create_layers_cb,connect_layers) '+str(time.time()-start_time)+'\n')
  return cb_layers


