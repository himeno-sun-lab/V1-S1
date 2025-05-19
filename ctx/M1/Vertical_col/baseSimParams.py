# Base parameters and their meaning
# these defaults can be overrided via a custom python parameter file or via the commandline (in this order: commandline arguments take precedence over customParams.py, which take precendence over the defaults defined here)

####################################################################################
# This file should not be modified! Use commandline arguments or custom param file #
####################################################################################

simParams = {
'whichSim':          'main_sim.py', # task to be run (default: initialize all regions, interconnect them, and establish baseline activity)
'simDuration':                1500., # simulation duration, in ms
'overwrite_files':          True, # should we overwrite nest configuration files?
'dt':                     '0.1', # nest temporal resolution
#'nestSeed':                     20, # nest seed (affects input poisson spike trains)
#'pythonSeed':                   10, # python seed (affects connection map)
'nbcpu':                         20, # number of CPUs to be used by nest
'nbnodes':                     '1', # number of nodes, used by K computer
'scalefactor':                [1.0, 1.0], # scale factor
'initial_ignore':              500.0, # the time in start of stimulation which is ignored (ms)
'msd':                         123456, #master seed (change this by a 2*Nvp+1 for independent experiments.)
'channels':                  True, # If all the neurons are organized as circular channels
'channels_radius':            0.16,
'hex_radius':               0.240,
'circle_center':             [],
'channels_nb':                 6,
'macro_columns_nb':             7,
'micro_columns_nb':             7,
}
