
####################################################################################
#                               Sim Pramas Example                                 #
####################################################################################

simParams = {
'whichSim':          'sim_S1_M1.py', # task to be run (default: initialize all regions, interconnect them, and establish baseline activity)
'simDuration':                1000., #1500., # simulation duration, in ms
'overwrite_files':          True, # should we overwrite nest configuration files?
'dt':                     '0.1', # nest temporal resolution
#'nestSeed':                     20, # nest seed (affects input poisson spike trains)
#'pythonSeed':                   10, # python seed (affects connection map)
'nbcpu':                         8, # number of CPUs to be used by nest
'nbnodes':                     1, # number of nodes, used by K computer
'scalefactor':                [1., 1.], # scale factor
'initial_ignore':              500., # the time in start of stimulation which is ignored (ms)
'msd':                         123456, #master seed (change this by a 2*Nvp+1 for independent experiments.)
'channels':                  True, # If all the neurons are organized as circular channels
'conn_params':               {
     "p_center": 0.1,
      "sigma": 411.40,
      "weight": 0.2,
      "delay": 1.5,
      "weight_distribution": "homogeneous"
    },

'channels_radius':            0.05,
'hex_radius':               0.10,
's1_circle_center':             [[0.2,0.2],[-0.2,-0.2],[0.2,-0.2],[-0.2,0.2]],
'm1_circle_center':             [[0,0.2],[0, -0.2]],
'channels_nb':                 4,
'macro_columns_nb':             7,
'micro_columns_nb':             7,
'Psg_params':               [{'rate': 100.0}, {'rate': 0.0}, {'rate': 0.0}, {'rate': 0.0}],
'weight_matrix':       [[1.0, 0.8], [0.8, 1.0], [0.6, 0.9], [0.9, 0.6]],
'stimulation_start_time':  500.0,

'sim_model': {'resting_state':{'on':False,'regions':{'S1':True, 'M1':True, 'BG':True, 'TH_M1':True, 'TH_S1':True, 'CB_S1':True, 'CB_M1':True}},
              'cb_learning':{'on':False,'regions':{'S1':True, 'M1':True, 'BG':True, 'TH_M1':True, 'TH_S1':True, 'CB_S1':True, 'CB_M1':True}, 'trials_nb': 1,'delta_t':500.},
              'multiple_arm_reaching':{'on':False,'regions':{'S1':True, 'M1':True, 'BG':True, 'TH_M1':True, 'TH_S1':True, 'CB_S1':True, 'CB_M1':True}, 'trials_nb': 3,'delta_t':500.},
              'single_arm_reaching':{'on':False,'regions':{'S1':True, 'M1':True, 'BG':True, 'TH_M1':True, 'TH_S1':True, 'CB_S1':True, 'CB_M1':True}},
              'reinf_learning':{'on':False,'regions':{'S1':True, 'M1':True, 'BG':True, 'TH_M1':True, 'TH_S1':True, 'CB_S1':False, 'CB_M1':False},'trials_nb':5,'delta_t':500.},
              'arm_movement':{'on':False,'regions':{'S1':True, 'M1':True, 'BG':True, 'TH_M1':True, 'TH_S1':True, 'CB_S1':False, 'CB_M1':False}},
              'BG_only':{'on':False,'regions':{'S1':False, 'M1':False, 'BG':True, 'TH_M1':False, 'TH_S1':False, 'CB_S1':False, 'CB_M1':False}},
              'S1_only':{'on':False,'regions':{'S1':True, 'M1':False, 'BG':False, 'TH_M1':False, 'TH_S1':False, 'CB_S1':False, 'CB_M1':False}},
              'CB_only':{'on':False,'regions':{'S1':False, 'M1':False, 'BG':False, 'TH_M1':False, 'TH_S1':False, 'CB_S1':True, 'CB_M1':True}},
              'M1_only':{'on':True,'regions':{'S1':False, 'M1':True, 'BG':False, 'TH_M1':False, 'TH_S1':False, 'CB_S1':False, 'CB_M1':False}},
              'TH_only':{'on':False,'regions':{'S1':False, 'M1':False, 'BG':False, 'TH_M1':True, 'TH_S1':True, 'CB_S1':False, 'CB_M1':False}}
              }
}

