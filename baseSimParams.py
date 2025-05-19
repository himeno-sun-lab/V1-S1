# Base parameters and their meaning
# these defaults can be overrided via a custom python parameter file or via the commandline (in this order: commandline arguments take precedence over customParams.py, which take precendence over the defaults defined here)

####################################################################################
# This file should not be modified! Use commandline arguments or custom param file #
####################################################################################

simParams = {
'whichSim':          'main_sim.py', # task to be run (default: initialize all regions, interconnect them, and establish baseline activity)
'simDuration':                1000., #1500., # simulation duration, in ms
'overwrite_files':          True, # should we overwrite nest configuration files?
'dt':                     '0.1', # nest temporal resolution
#'nestSeed':                     20, # nest seed (affects input poisson spike trains)
#'pythonSeed':                   10, # python seed (affects connection map)
'nbcpu':                         1, # number of CPUs to be used by nest
'nbnodes':                     1, # number of nodes, used by K computer
'scalefactor':                [1., 1.], # scale factor
'initial_ignore':              500., # the time in start of stimulation which is ignored (ms)
'msd':                         123456, #master seed (change this by a 2*Nvp+1 for independent experiments.)
'channels':                  False, # If all the neurons are organized as circular channels
'channels_radius':            0.16,
'hex_radius':               0.240,
'circle_center':             [],
'channels_nb':                 6,
'macro_columns_nb':             7,
'micro_columns_nb':             7,
'sim_model': {'resting_state':{'on':False,'regions':{'S1':True, 'M1':True, 'BG':True, 'TH_M1':True, 'TH_S1':True, 'CB_S1':True, 'CB_M1':True}},
              'cb_learning':{'on':False,'regions':{'S1':True, 'M1':True, 'BG':True, 'TH_M1':True, 'TH_S1':True, 'CB_S1':True, 'CB_M1':True}, 'trials_nb': 1,'delta_t':500.},
              'multiple_arm_reaching':{'on':False,'regions':{'S1':True, 'M1':True, 'BG':True, 'TH_M1':True, 'TH_S1':True, 'CB_S1':True, 'CB_M1':True}, 'trials_nb': 3,'delta_t':500.},
              'single_arm_reaching':{'on':False,'regions':{'S1':True, 'M1':True, 'BG':True, 'TH_M1':True, 'TH_S1':True, 'CB_S1':True, 'CB_M1':True}},
              'reinf_learning':{'on':False,'regions':{'S1':True, 'M1':True, 'BG':True, 'TH_M1':True, 'TH_S1':True, 'CB_S1':False, 'CB_M1':False},'trials_nb':5,'delta_t':500.},
              'arm_movement':{'on':False,'regions':{'S1':True, 'M1':True, 'BG':True, 'TH_M1':True, 'TH_S1':True, 'CB_S1':False, 'CB_M1':False}},
              'BG_only':{'on':False,'regions':{'S1':False, 'M1':False, 'BG':True, 'TH_M1':False, 'TH_S1':False, 'CB_S1':False, 'CB_M1':False}},
              'S1_only':{'on':False,'regions':{'S1':True, 'M1':False, 'BG':False, 'TH_M1':False, 'TH_S1':False, 'CB_S1':False, 'CB_M1':False}},
              'CB_only':{'on':True,'regions':{'S1':False, 'M1':False, 'BG':False, 'TH_M1':False, 'TH_S1':False, 'CB_S1':True, 'CB_M1':True}},
              'M1_only':{'on':False,'regions':{'S1':False, 'M1':True, 'BG':False, 'TH_M1':False, 'TH_S1':False, 'CB_S1':False, 'CB_M1':False}},
              'TH_only':{'on':False,'regions':{'S1':False, 'M1':False, 'BG':False, 'TH_M1':True, 'TH_S1':True, 'CB_S1':False, 'CB_M1':False}}
              }
}

