# Base parameters and their meaning
# these defaults can be overrided via a custom python parameter file or via the commandline (in this order: commandline arguments take precedence over customParams.py, which take precendence over the defaults defined here)

####################################################################################
# This file should not be modified! Use commandline arguments or custom param file #
####################################################################################

simParams = {
'whichSim':          'main_sim.py', # task to be run (default: initialize all regions, interconnect them, and establish baseline activity)
'simDuration':                1000, # simulation duration, in ms
'overwrite_files':          True, # should we overwrite nest configuration files?
'dt':                     '0.1', # nest temporal resolution
#'nestSeed':                     20, # nest seed (affects input poisson spike trains)
#'pythonSeed':                   10, # python seed (affects connection map)
'nbcpu':                         1, # number of CPUs to be used by nest
'nbnodes':                     '1', # number of nodes, used by K computer
}

