#!/usr/bin/env python
# -*- coding: utf-8 -*-

##
## nest_routine.py
##
## This script defines creation, connection, and simulation routines using PyNest
##
## It is split in several parts, one for each brain region simulated:
## `CTX`, 'CTX_M2' `TH`, `BH`, `CERE`
##
## Functions should be sufffixed by their regions: `_ctx`, 'ctx_M2' `_th`, `_bg`, `_cb`

import nest.topology as ntop
from nest.lib.hl_api_info import SetStatus
import nest

import numpy as np
import time

###########
# General #
###########

#-------------------------------------------------------------------------------
# Nest initialization
#-------------------------------------------------------------------------------
def initialize_nest(sim_params):
  nest.set_verbosity("M_WARNING")
  nest.SetKernelStatus({"overwrite_files": sim_params['overwrite_files']}) # should we erase previous traces when redoing a simulation?
  nest.SetKernelStatus({'local_num_threads': int(sim_params['nbcpu'])})
  nest.SetKernelStatus({"data_path": 'log/'})
  if sim_params['dt'] != 0.1:
    nest.SetKernelStatus({'resolution': float(sim_params['dt'])})

#-------------------------------------------------------------------------------
# Starts the Nest simulation, given the general parameters of `sim_params`
#-------------------------------------------------------------------------------
def run_simulation(sim_params):
  nest.ResetNetwork()
  nest.Simulate(sim_params['simDuration'])

#-------------------------------------------------------------------------------
# Instantiate a spike detector and connects it to the entire layer `layer_gid`
#-------------------------------------------------------------------------------
def layer_spike_detector(layer_gid, layer_name, params={"withgid": True, "withtime": True, "to_file": True}):
  print ('spike detector for '+layer_name)
  params.update({'label': layer_name})
  detector = nest.Create("spike_detector", params=params)
  nest.Connect(pre=nest.GetNodes(layer_gid)[0], post=detector)
  return detector

#-------------------------------------------------------------------------------
# Returns the average firing rate of a population
# It is relative to the simulation duration `simDuration` and the population size `n`
#-------------------------------------------------------------------------------
def average_fr(detector, simDuration, n):
  return nest.GetStatus(detector, 'n_events')[0] / float(simDuration) / float(n) * 1000

#-------------------------------------------------------------------------------
# Returns the number of neurons inside a layer
#-------------------------------------------------------------------------------
def count_layer(layer_gid):
    return len(nest.GetNodes(layer_gid)[0])

#-------------------------------------------------------------------------------
# Returns the positions of neurons inside a layer -sun-20180911
#-------------------------------------------------------------------------------
def get_position(layer_gid):
    return ntop.GetPosition(layer_gid)

#-------------------------------------------------------------------------------
# Returns the connections of neurons inside a layer -sun-20180912
#-------------------------------------------------------------------------------
def get_connection(gids):
    return nest.GetConnections(gids)
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


#######
# CTX #
#######

######
# S1 #
######

def create_layers_ctx(extent, center, positions, elements, neuron_info):
    Neuron_pos_list=positions[:, :3].tolist()
    nest.SetDefaults(elements, {"I_e": float(neuron_info['I_ex']), "V_th": float(neuron_info['spike_threshold']), "V_reset": float(neuron_info['reset_value']), "t_ref": float(neuron_info['absolute_refractory_period'])})
    newlayer = ntop.CreateLayer({ 'extent': extent, 'center': center, 'positions' : Neuron_pos_list , 'elements': elements} )
    #Neurons = nest.GetNodes(newlayer)
    return newlayer
    #SetStatus(Neurons[0], {"I_e": float(neuron_info['I_ex']), "V_th": float(neuron_info['spike_threshold']),
    #                        "V_reset": float(neuron_info['reset_value']),
    #                        "t_ref": float(neuron_info['absolute_refractory_period'])})


# connect (intra regional connection?)
def connect_layers_ctx(pre_SubSubRegion, post_SubSubRegion, conn_dict):
    sigma_x = conn_dict['sigma']/1000.
    sigma_y = conn_dict['sigma']/1000.
    weight_distribution=conn_dict['weight_distribution']
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

'''
#######
# CTX #
#######

######
# M1 #
######

def create_layers_ctx_M1(extent, center, positions, elements, neuron_info):
    Neuron_pos_list=positions[:, :3].tolist()
    nest.SetDefaults(elements, {"I_e": float(neuron_info['I_ex']), "V_th": float(neuron_info['spike_threshold']), "V_reset": float(neuron_info['reset_value']),
                                "t_ref": float(neuron_info['absolute_refractory_period'])})
    newlayer = ntop.CreateLayer({ 'extent': extent, 'center': center, 'positions' : Neuron_pos_list , 'elements': elements} )
    #Neurons = nest.GetNodes(newlayer)
    return newlayer
    

# connect (intra regional connection?)
def connect_layers_ctx_M1(pre_SubSubRegion, post_SubSubRegion, conn_dict):
    sigma_x = conn_dict['sigma']/1000.
    sigma_y = conn_dict['sigma']/1000.
    weight_distribution=conn_dict['weight_distribution']
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
        

######
# M2 #
######

def create_layers_ctx_M2(extent, center, positions, elements, neuron_info):
    Neuron_pos_list=positions[:, :3].tolist()
    nest.SetDefaults(elements, {"I_e": float(neuron_info['I_ex']), "V_th": float(neuron_info['spike_threshold']), "V_reset": float(neuron_info['reset_value']),
                                "t_ref": float(neuron_info['absolute_refractory_period'])})
    newlayer = ntop.CreateLayer({ 'extent': extent, 'center': center, 'positions' : Neuron_pos_list , 'elements': elements} )
    #Neurons = nest.GetNodes(newlayer)
    return newlayer
    

# connect (intra regional connection?)
def connect_layers_ctx_M2(pre_SubSubRegion, post_SubSubRegion, conn_dict):
    sigma_x = conn_dict['sigma']/1000.
    sigma_y = conn_dict['sigma']/1000.
    weight_distribution=conn_dict['weight_distribution']
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
        
        
'''

######
# TH #
######

def create_layers_th(extent, center, positions, elements, neuron_info):
    newlayer = ntop.CreateLayer({ 'extent': extent, 'center': center, 'positions' : positions , 'elements': elements} )
    Neurons = nest.GetNodes(newlayer)
    SetStatus(Neurons[0], {"I_e": float(neuron_info['I_ex']), "V_th": float(neuron_info['spike_threshold']),
                            "V_reset": float(neuron_info['reset_value']),
                            "t_ref": float(neuron_info['absolute_refractory_period'])})
    return newlayer


# connect (intra regional connection?)
def connect_layers_th(pre_SubSubRegion, post_SubSubRegion, conn_dict):
    sigma_x = conn_dict['sigma']
    sigma_y = conn_dict['sigma']
    conndict = {'connection_type': 'divergent',
                'mask': {'spherical': {'radius': 2.0}},
                'kernel': {
                    'gaussian2D': {'p_center': conn_dict['p_center'], 'sigma_x': sigma_x, 'sigma_y': sigma_y}},
                'weights': conn_dict['weight'],
                'delays': conn_dict['delay'],
                'allow_autapses': False,
                'allow_multapses': True}
    if sigma_x != 0 and conn_dict['p_center']!=0. :
        ntop.ConnectLayers(pre_SubSubRegion, post_SubSubRegion, conndict)

########
# CERE #
########

#def connect_layers_bg():
#    print ("Fill connection later")
#    pass

#nest_routine for cb
def connect_layers_cb():
    pass
    
def create_layers_cb(rows, columns, elements, extent, center):
    print (elements)
    if elements[0]=='GR' or (elements[0]=='GO' or elements[0]=='VN'):

        pos_x = np.linspace(-extent[0] / 2., extent[0] / 2., num=rows, endpoint=True)
        pos_y = np.linspace(-extent[1] / 2., extent[1] / 2., num=columns, endpoint=True)
        pos_z=0.0
        positions = np.zeros((rows*columns, 3))
        for i in range(rows):
            for j in range (columns):
                positions[i*columns+j]=np.array([pos_x[i], pos_y[j], pos_z])
        layer = ntop.CreateLayer({'extent': [extent[0]+1.,extent[1]+1.,extent[2]+1. ], 'center': center, 'positions': positions.tolist(), 'elements': elements[0]})

    elif elements[0]=='PKJ' or elements[0]=='BS':
        pos_x = 0.0
        pos_y = np.linspace(-extent[1] / 2., extent[1] / 2., num=columns, endpoint=True)
        pos_z = 0.0
        for i in range (columns):
            positions = np.zeros((columns, 3))
            positions[i,:]=np.array([pos_x, pos_y[i], pos_z])
        layer = ntop.CreateLayer({'extent': [extent[0]+1.,extent[1]+1.,extent[2]+1. ], 'center': center, 'positions': positions.tolist(), 'elements': elements[0]})
    elif elements[0]=='IO':
        pos_x = 0.0
        pos_y = 0.0
        pos_z = 0.0
        positions=[[pos_x, pos_y, pos_z]]
        layer = ntop.CreateLayer({'extent': [extent[0]+1.,extent[1]+1.,extent[2]+1. ], 'center': center, 'positions': positions, 'elements': elements[0]})
    return layer

def create_neurons_cb():
    import CBneurons
    CBneurons.create_neurons()

######
# BG #
######

AMPASynapseCounter_bg = 0 # initialize global counter variable for AMPA/NMDA colocalization in BG (unfortunate design choice, but required by nest fast connect procedure)

#-------------------------------------------------------------------------------
# Establishes a topological layer and returns it
# bg_params: basal ganglia parameters
# nucleus: name of the nucleus to instantiate
# fake: numerical value - if 0, then a real population of iaf is instantiated
#                       - if fake > 0, then a Poisson generator population firing at `fake` Hz is instantiated
# force_pop_size: if defined, initialize only this number of neurons
#                 -> this is useful for the cortical connections, as some inputs will be derived from L5A and L5B layers
#-------------------------------------------------------------------------------
def create_layers_bg(bg_params, nucleus, fake=0, mirror_neurons=None):
  if mirror_neurons == None:
    # normal case: full input layer is created
    pop_size = int(bg_params['nb'+nucleus])
  else:
    # inputs come from existing ctx layer: only a fraction of poisson generators are created
    pop_size = int(bg_params['nb'+nucleus]) - len(mirror_neurons)

  print('population size: '+str(pop_size))
  positions = [np.random.uniform(0., 1., pop_size).tolist(),
               np.random.uniform(0., 1., pop_size).tolist()]

  if mirror_neurons != None:
    # Also retrieve the position of ctx neurons
    ctx_positions = np.array(ntop.GetPosition(list(get_local_nodes(mirror_neurons)))).transpose().tolist()
    positions[0] += ctx_positions[0]
    positions[1] += ctx_positions[1]
    # we ignore z-axis (for now?)

  if fake == 0:
    # fake == 0 is the normal case, where actual iaf neurons are instantiated
    nest.SetDefaults('iaf_psc_alpha_multisynapse', bg_params['common_iaf'])
    nest.SetDefaults('iaf_psc_alpha_multisynapse', bg_params[nucleus+'_iaf'])
    nest.SetDefaults('iaf_psc_alpha_multisynapse', {"I_e": bg_params['Ie'+nucleus]})
    element = 'iaf_psc_alpha_multisynapse'
  else:
    # when fake > 0, parrot neurons instantiated
    element = 'parrot_neuron'

  layer_gid = ntop.CreateLayer({'positions': [[positions[0][i], positions[1][i]] for i in range(len(positions[0]))], 'elements': element, 'extent': [2., 2.], 'center':[0.5, 0.5], 'edge_wrap': True})
  if fake > 0:
    # when fake > 0, parrot neurons are connected to poisson generators firing at `fake`Hz
    poisson = nest.Create('poisson_generator', pop_size)
    nest.SetStatus(poisson, {'rate': fake})
    nest.Connect(pre=poisson, post=nest.GetNodes(layer_gid)[0][0:pop_size], conn_spec={'rule':'one_to_one'})
  
  return layer_gid

#-------------------------------------------------------------------------------
# Establishes a topological connection between two populations
# bg_params: basal ganglia parameters
# nType : a string 'ex' or 'in', defining whether it is excitatory or inhibitory
# bg_layers : dictionary of basal ganglia layers
# nameTgt, nameSrc : strings naming the populations, as defined in NUCLEI list
# projType : type of projections. For the moment: 'focused' (only channel-to-channel connection) and
#            'diffuse' (all-to-one with uniform distribution)
# redundancy, RedundancyType : contrains the inDegree - see function `connect` for details
# LCGDelays : shall we use the delays obtained by (Liénard, Cos, Girard, in prep) or not (default = True)
# gain : allows to amplify the weight normally deduced from LG14
# stochastic_delays: to enable stochasticity in the axonal delays
# spreads: a 2-item list specifying the radius of focused and diffuse projections
#-------------------------------------------------------------------------------
def connect_layers_bg(bg_params, nType, bg_layers, nameSrc, nameTgt, projType, redundancy, RedundancyType, LCGDelays=True, gain=1., stochastic_delays=None, spreads=None, verbose=False):
  def printv(text):
    if verbose:
      print(text)

  printv("\n* connecting "+nameSrc+" -> "+nameTgt+" with "+nType+" "+projType+" connection")

  recType = {'AMPA':1,'NMDA':2,'GABA':3}

  if RedundancyType == 'inDegreeAbs':
    # inDegree is already provided in the right form
    inDegree = float(redundancy)
  elif RedundancyType == 'outDegreeAbs':
    #### fractional outDegree is expressed as a fraction of max axo-dendritic contacts
    inDegree = get_frac_bg(bg_params, 1./redundancy, nameSrc, nameTgt, bg_params['count'+nameSrc], bg_params['count'+nameTgt], verbose=verbose)
  elif RedundancyType == 'outDegreeCons':
    #### fractional outDegree is expressed as a ratio of min/max axo-dendritic contacts
    inDegree = get_frac_bg(bg_params, redundancy, nameSrc, nameTgt, bg_params['count'+nameSrc], bg_params['count'+nameTgt], useMin=True, verbose=verbose)
  else:
    raise KeyError('`RedundancyType` should be one of `inDegreeAbs`, `outDegreeAbs`, or `outDegreeCons`.')

  # check if in degree acceptable (not larger than number of neurons in the source nucleus)
  if projType == 'focused' and inDegree > bg_params['nb'+nameSrc]:
    printv("/!\ WARNING: required 'in degree' ("+str(inDegree)+") larger than number of neurons in individual source channels ("+str(bg_params['nb'+nameSrc])+"), thus reduced to the latter value")
    inDegree = bg_params['nb'+nameSrc]
  if projType == 'diffuse' and inDegree > bg_params['nb'+nameSrc]:
    printv("/!\ WARNING: required 'in degree' ("+str(inDegree)+") larger than number of neurons in the overall source population ("+str(bg_params['nb'+nameSrc])+"), thus reduced to the latter value")
    inDegree = bg_params['nb'+nameSrc]

  if inDegree == 0.:
    printv("/!\ WARNING: non-existent connection strength, will skip")
    return

  global AMPASynapseCounter_bg

  # prepare receptor type lists:
  if nType == 'ex':
    lRecType = ['AMPA','NMDA']
    AMPASynapseCounter_bg = AMPASynapseCounter_bg + 1
    lbl = AMPASynapseCounter_bg # needs to add NMDA later
  elif nType == 'AMPA':
    lRecType = ['AMPA']
    lbl = 0
  elif nType == 'NMDA':
    lRecType = ['NMDA']
    lbl = 0
  elif nType == 'in':
    lRecType = ['GABA']
    lbl = 0
  else:
    raise KeyError('Undefined connexion type: '+nType)

  # compute the global weight of the connection, for each receptor type:
  W = computeW_bg(bg_params, lRecType, nameSrc, nameTgt, inDegree, gain, verbose=verbose)

  printv("  W="+str(W)+" and inDegree="+str(inDegree))

  # determine which transmission delay to use:
  if LCGDelays:
    delay = bg_params['tau'][nameSrc+'->'+nameTgt]
  else:
    delay = 1.

  if projType == 'focused': # if projections focused, input come only from the same channel as tgtChannel
    mass_connect_bg(bg_params, bg_layers, nameSrc, nameTgt, lbl, inDegree, recType[lRecType[0]], W[lRecType[0]], delay, spread=bg_params['spread_focused'], stochastic_delays = stochastic_delays, verbose=verbose)
  elif projType == 'diffuse': # if projections diffused, input connections are shared among each possible input channel equally
    mass_connect_bg(bg_params, bg_layers, nameSrc, nameTgt, lbl, inDegree, recType[lRecType[0]], W[lRecType[0]], delay, spread=bg_params['spread_diffuse'], stochastic_delays = stochastic_delays, verbose=verbose)

  if nType == 'ex':
    # mirror the AMPA connection with similarly connected NMDA connections
    src_idx = 0
    mass_mirror_bg(nest.GetNodes(bg_layers[nameSrc])[src_idx], lbl, recType['NMDA'], W['NMDA'], delay, stochastic_delays = stochastic_delays)

  return W


#------------------------------------------------------------------------------
# Routine to perform the fast connection using nest built-in `connect` function
# - `bg_params` is basal ganglia parameters
# - `bg_layers` is the dictionary of basal ganglia layers
# - `sourceName` & `destName` are names of two different layers
# - `synapse_label` is used to tag connections and be able to find them quickly
#   with function `mass_mirror`, that adds NMDA on top of AMPA connections
# - `inDegree`, `receptor_type`, `weight`, `delay` are Nest connection params
# - `spread` is a parameter that affects the diffusion level of the connection
#------------------------------------------------------------------------------
def mass_connect_bg(bg_params, bg_layers, sourceName, destName, synapse_label, inDegree, receptor_type, weight, delay, spread, stochastic_delays=None, verbose=False):
  def printv(text):
    if verbose:
      print(text)

  # potential initialization of stochastic delays
  if stochastic_delays != None and delay > 0:
    printv('Using stochastic delays in mass-connect')
    low = delay * 0.5
    high = delay * 1.5
    sigma = delay * stochastic_delays
    delay =  {'distribution': 'normal_clipped', 'low': low, 'high': high, 'mu': delay, 'sigma': sigma}

  ## set default synapse model with the chosen label
  nest.SetDefaults('static_synapse_lbl', {'synapse_label': synapse_label, 'receptor_type': receptor_type})

  # creation of the topological connection dict
  conndict = {'connection_type': 'convergent',
              'mask': {'circular': {'radius': spread}},
              'synapse_model': 'static_synapse_lbl', 'weights': weight, 'delays':delay,
              'allow_oversized_mask': True, 'allow_multapses': True}

  # The first call ensures that all neurons in `destName`
  # have at least `int(inDegree)` incoming connections
  integer_inDegree = np.floor(inDegree)
  if integer_inDegree>0:
    printv('Adding '+str(int(integer_inDegree*bg_params['nb'+destName]))+' connections with rule `fixed_indegree`')
    integer_conndict = conndict.copy()
    integer_conndict.update({'number_of_connections': int(integer_inDegree)})
    ntop.ConnectLayers(bg_layers[sourceName], bg_layers[destName], integer_conndict)

  # The second call distributes the approximate number of remaining axonal
  # contacts at random (i.e. the remaining fractional part after the first step)
  # Why "approximate"? Because with pynest layers, there are only two ways to specify
  # the number of axons in a connection:
  #    1) with an integer, specified with respect to each source (alt. target) neurons
  #    2) as a probability
  # Here, we have a fractional part - not an integer number - so that leaves us option 2.
  # However, because the new axonal contacts are drawn at random, we will not have the
  # exact number of connections
  float_inDegree = inDegree - integer_inDegree
  remaining_connections = np.round(float_inDegree * bg_params['nb'+destName])
  if remaining_connections > 0:
    printv('Adding '+str(remaining_connections)+' remaining connections with rule `fixed_total_number`')
    float_conndict = conndict.copy()
    float_conndict.update({'kernel': 1. / (bg_params['nb'+sourceName] * float(remaining_connections))})
    ntop.ConnectLayers(bg_layers[sourceName], bg_layers[destName], float_conndict)

#------------------------------------------------------------------------------
# Routine to duplicate a connection made with a specific receptor, with another
# receptor (typically to add NMDA connections to existing AMPA connections)
# - `source` & `synapse_label` should uniquely define the connections of
#   interest - typically, they are the same as in the call to `mass_connect`
# - `receptor_type`, `weight`, `delay` are Nest connection params
#------------------------------------------------------------------------------
def mass_mirror_bg(source, synapse_label, receptor_type, weight, delay, stochastic_delays, verbose=False):
  def printv(text):
    if verbose:
      print(text)

  # find all AMPA connections for the given projection type
  printv('looking for AMPA connections to mirror with NMDA...\n')
  ampa_conns = nest.GetConnections(source=source, synapse_label=synapse_label)
  # in rare cases, there may be no connections, guard against that
  if ampa_conns:
    # extract just source and target GID lists, all other information is irrelevant here
    printv('found '+str(len(ampa_conns))+' AMPA connections\n')
    if stochastic_delays != None and delay > 0:
      printv('Using stochastic delays in mass-miror')
      delay = np.array(nest.GetStatus(ampa_conns, keys=['delay'])).flatten()
    src, tgt, _, _, _ = zip(*ampa_conns)
    nest.Connect(src, tgt, 'one_to_one',
                 {'model': 'static_synapse_lbl',
                  'synapse_label': synapse_label, # tag with the same number (doesn't matter)
                  'receptor_type': receptor_type, 'weight': weight, 'delay':delay})
  
#-------------------------------------------------------------------------------
# Helper function to set a basal ganglia internal projection
# computes the inDegree as a fraction of maximal possible inDegree
# `FractionalOutDegree` is the outDegree, expressed as a fraction
#-------------------------------------------------------------------------------
def get_frac_bg(bg_params, FractionalOutDegree, nameSrc, nameTgt, cntSrc, cntTgt, useMin=False, verbose=False):
  if useMin == False:
    # 'FractionalOutDegree' is taken to be relative to the maximal number of axo-dendritic contacts
    inDegree = get_input_range_bg(bg_params, nameSrc, nameTgt, cntSrc, cntTgt, verbose=verbose)[1] * FractionalOutDegree
  else:
    # 'FractionalOutDegree' is taken to be relative to the maximal number of axo-dendritic contacts and their minimal number
    r = get_input_range_bg(bg_params, nameSrc, nameTgt, cntSrc, cntTgt, verbose=verbose)
    inDegree = (r[1] - r[0]) * FractionalOutDegree + r[0]
  if verbose:
    print('\tConverting the fractional outDegree of '+nameSrc+' -> '+nameTgt+' from '+str(FractionalOutDegree)+' to inDegree neuron count: '+str(round(inDegree, 2))+' (relative to minimal value possible? '+str(useMin)+')')
  return inDegree

#-------------------------------------------------------------------------------
# Helper function to set a basal ganglia internal projection
# computes the weight of a connection, based on LG14 parameters
#-------------------------------------------------------------------------------
def computeW_bg(bg_params, listRecType, nameSrc, nameTgt, inDegree, gain=1.,verbose=False):
  recType = {'AMPA':1,'NMDA':2,'GABA':3}
  nu = get_input_range_bg(bg_params, nameSrc, nameTgt, bg_params['count'+nameSrc], bg_params['count'+nameTgt], verbose=verbose)[1]
  if verbose:
    print('\tCompare with the effective chosen inDegree   : '+str(inDegree))

  # attenuation due to the distance from the receptors to the soma of tgt:
  LX=bg_params['lx'][nameTgt]*np.sqrt((4.*bg_params['Ri'])/(bg_params['dx'][nameTgt]*bg_params['Rm']))
  attenuation = np.cosh(LX*(1-bg_params['distcontact'][nameSrc+'->'+nameTgt])) / np.cosh(LX)

  w={}
  for r in listRecType:
    w[r] = nu / float(inDegree) * attenuation * bg_params['wPSP'][recType[r]-1] * gain
  return w

#-------------------------------------------------------------------------------
# Helper function to set a basal ganglia internal projection
# returns the minimal & maximal numbers of distinct input neurons for one connection
#-------------------------------------------------------------------------------
def get_input_range_bg(bg_params, nameSrc, nameTgt, cntSrc, cntTgt, verbose=False):
  if nameSrc=='CSN' or nameSrc=='PTN':
    nu = bg_params['alpha'][nameSrc+'->'+nameTgt]
    nu0 = 0
    if verbose:
      print('\tMaximal number of distinct input neurons (nu): '+str(nu))
      print('\tMinimal number of distinct input neurons     : unknown (set to 0)')
  else:
    nu = cntSrc / float(cntTgt) * bg_params['ProjPercent'][nameSrc+'->'+nameTgt] * bg_params['alpha'][nameSrc+'->'+nameTgt]
    nu0 = cntSrc / float(cntTgt) * bg_params['ProjPercent'][nameSrc+'->'+nameTgt]
    if verbose:
      print('\tMaximal number of distinct input neurons (nu): '+str(nu))
      print('\tMinimal number of distinct input neurons     : '+str(nu0))
  return [nu0, nu]



######
# CB #
######

def connect_layers_cere():
  # CBnetwork was used instad of connect_layers_cere()
    pass

####################
# INTERCONNECTIONS #
####################

#-------------------------------------------------------------------------------
# Identify a pool of ctx neurons to project to bg
# 1000 neurons from l5a and 1000 neurons from layer 5b
#-------------------------------------------------------------------------------
def identify_proj_neurons_ctx_bg(ctx_layers):
    import random
    ctx_l5a_gids = list(nest.GetNodes(ctx_layers['L5APY'])[0])
    random.shuffle(ctx_l5a_gids)
    ctx_l5a = ctx_l5a_gids[:1000]
    print ("ctx_l5a neurons to connect: %d"%(len(ctx_l5a)))
    ctx_l5b_gids = list(nest.GetNodes(ctx_layers['L5BPY'])[0])
    random.shuffle(ctx_l5b_gids)
    ctx_l5b = ctx_l5b_gids[:1000]
    print ("ctx_l5b neurons to connect: %d"%(len(ctx_l5b)))
    return {'CSN': ctx_l5a, 'PTN': ctx_l5b}

#-------------------------------------------------------------------------------
# Connect the ctx neurons of the chosen subset to the basal ganglia
#-------------------------------------------------------------------------------
def connect_ctx_bg(ctx_neurons_gid, bg_layer_gid):
    #import ipdb; ipdb.set_trace()
    nest.Connect(pre=ctx_neurons_gid, post=nest.GetNodes(bg_layer_gid)[0][-len(ctx_neurons_gid):], conn_spec={'rule':'one_to_one'})


#-------------------------------------------------------------------------------
# Connect the ctx neurons of the chosen subset to the cerebellum
#-------------------------------------------------------------------------------
def connect_region_ctx_cb(layer_ctx, layer_cb):
    print ('Connect the ctx neurons of the chosen subset to the cerebellum')
    sigma_x = 0.25
    sigma_y = 0.25
    p_center = 0.1
    weight=0.01
    delay=1.0
    conndict = {'connection_type': 'divergent',
                'mask': {'spherical': {'radius': 2.0}},
                'kernel': {
                    'gaussian2D': {'p_center': p_center, 'sigma_x': sigma_x, 'sigma_y': sigma_y}},
                'weights': weight,
                'delays': delay}
    ntop.ConnectLayers(layer_ctx, layer_cb, conndict)



#-------------------------------------------------------------------------------
# Connect the ctx neurons of the chosen subset to the thalamus
#-------------------------------------------------------------------------------
def connect_region_ctx_th(layer_ctx, layer_th):
    print('Connect the ctx neurons of the chosen subset to the thalamus')
    #th to ctx
    #TC to L5B PY
    sigma_x = 0.1
    sigma_y = 0.1
    p_center = 0.1
    weight=0.01
    delay=10.0
    conndict = {'connection_type': 'divergent',
                'mask': {'spherical': {'radius': sigma_x*3.0}},
                'kernel': {
                    'gaussian2D': {'p_center': p_center, 'sigma_x': sigma_x, 'sigma_y': sigma_y}},
                'weights': weight,
                'delays': delay}
    ntop.ConnectLayers(layer_th['S1_EZ_th']['thalamic_nucleusTC'], layer_ctx['L5BPY'], conndict)
    ntop.ConnectLayers(layer_th['S1_IZ_th']['thalamic_nucleusTC'], layer_ctx['L5BPY'], conndict)

    # TC to L5B FS
    sigma_x = 0.1
    sigma_y = 0.1
    p_center = 0.1
    weight=0.01
    delay=10.0
    conndict = {'connection_type': 'divergent',
                'mask': {'spherical': {'radius': sigma_x*3.0}},
                'kernel': {
                    'gaussian2D': {'p_center': p_center, 'sigma_x': sigma_x, 'sigma_y': sigma_y}},
                'weights': weight,
                'delays': delay}
    ntop.ConnectLayers(layer_th['S1_EZ_th']['thalamic_nucleusTC'], layer_ctx['L5BPV'], conndict)
    ntop.ConnectLayers(layer_th['S1_IZ_th']['thalamic_nucleusTC'], layer_ctx['L5BPV'], conndict)
    # TC to L5B SST
    sigma_x = 0.1
    sigma_y = 0.1
    p_center = 0.1
    weight=0.01
    delay=10.0
    conndict = {'connection_type': 'divergent',
                'mask': {'spherical': {'radius': sigma_x*3.0}},
                'kernel': {
                    'gaussian2D': {'p_center': p_center, 'sigma_x': sigma_x, 'sigma_y': sigma_y}},
                'weights': weight,
                'delays': delay}
    ntop.ConnectLayers(layer_th['S1_EZ_th']['thalamic_nucleusTC'], layer_ctx['L5BSST'], conndict)
    ntop.ConnectLayers(layer_th['S1_IZ_th']['thalamic_nucleusTC'], layer_ctx['L5BSST'], conndict)

    #ctx to th
    #L6PY to TC
    sigma_x = 0.1
    sigma_y = 0.1
    p_center = 0.1
    weight=0.01
    delay=10.0
    conndict = {'connection_type': 'divergent',
                'mask': {'spherical': {'radius': sigma_x*3.0}},
                'kernel': {
                    'gaussian2D': {'p_center': p_center, 'sigma_x': sigma_x, 'sigma_y': sigma_y}},
                'weights': weight,
                'delays': delay}
    ntop.ConnectLayers( layer_ctx['L6PY'], layer_th['S1_EZ_th']['thalamic_nucleusTC'], conndict)
    ntop.ConnectLayers(layer_ctx['L6PY'], layer_th['S1_IZ_th']['thalamic_nucleusTC'], conndict)

    # L6PY to RE
    sigma_x = 0.1
    sigma_y = 0.1
    p_center = 0.1
    weight=0.01
    delay=10.0
    conndict = {'connection_type': 'divergent',
                'mask': {'spherical': {'radius': sigma_x*3.0}},
                'kernel': {
                    'gaussian2D': {'p_center': p_center, 'sigma_x': sigma_x, 'sigma_y': sigma_y}},
                'weights': weight,
                'delays': delay}
    ntop.ConnectLayers( layer_ctx['L6PY'], layer_th['S1_EZ_th']['thalamic_reticular_neucleusRE'], conndict)
    ntop.ConnectLayers(layer_ctx['L6PY'], layer_th['S1_IZ_th']['thalamic_reticular_neucleusRE'], conndict)




# Connest cb and th
def connect_region_cb_th(cb_layers, th_layers):
    print('Connect the cb neurons to the thalamus')
    #vn to TC
    sigma_x = 0.1
    sigma_y = 0.1
    p_center = 0.1
    weight=0.01
    delay=10.0
    conndict = {'connection_type': 'divergent',
                'mask': {'spherical': {'radius': sigma_x*3.0}},
                'kernel': {
                    'gaussian2D': {'p_center': p_center, 'sigma_x': sigma_x, 'sigma_y': sigma_y}},
                'weights': weight,
                'delays': delay}
    ntop.ConnectLayers(cb_layers['layer_vn'], th_layers['S1_EZ_th']['thalamic_nucleusTC'], conndict)

    #vn to IN
    sigma_x = 0.1
    sigma_y = 0.1
    p_center = 0.1
    weight=0.01
    delay=10.0
    conndict = {'connection_type': 'divergent',
                'mask': {'spherical': {'radius': sigma_x*3.0}},
                'kernel': {
                    'gaussian2D': {'p_center': p_center, 'sigma_x': sigma_x, 'sigma_y': sigma_y}},
                'weights': weight,
                'delays': delay}
    ntop.ConnectLayers(cb_layers['layer_vn'], th_layers['S1_EZ_th']['thalamic_nucleusIN'], conndict)


def connect_region_bg_th(bg_layers, th_layers):
    print('Connect the bg neurons to the thalamus')
    #GPI to TC
    #sigma_x = 0.1
    #sigma_y = 0.1
    #p_center = 0.1
    #weight= - 0.01
    #delay = 1.0
    # conndict = {'connection_type': 'divergent',
    #             'mask': {'spherical': {'radius': sigma_x*3.0}},
    #             'kernel': {
    #                 'gaussian2D': {'p_center': p_center, 'sigma_x': sigma_x, 'sigma_y': sigma_y}},
    #             'weights': weight,
    #             'delays': delay}
    nest.Connect(nest.GetNodes(bg_layers['GPi'])[0], nest.GetNodes(th_layers['S1_IZ_th']['thalamic_nucleusTC'])[0])