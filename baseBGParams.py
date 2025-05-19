# Base parameters and their meaning
# these defaults can be overrided via a custom python parameter file or via the commandline (in this order: commandline arguments take precedence over customParams.py, which take precendence over the defaults defined here)

####################################################################################
# This file should not be modified! Use commandline arguments or custom param file #
####################################################################################

### The base parameters correspond to: 'LG14modelID':                   9, # ie. LG 2014 parameterization used of model #9
bg_scale = 4

bgParams = {
'nbMSN':        bg_scale*2644., # population size (default: 1/1000 of the BG)
'nbFSI':          bg_scale*53., # ^
'nbSTN':           bg_scale*8., # ^
'nbGPe':          bg_scale*25., # ^
'nbGPi':          bg_scale*14., # ^
'nbCSN':          bg_scale*3000., # used only if fake_inputs == True
'nbPTN':          bg_scale*3000., # ^
'nbCMPf':          bg_scale*3000., # ^
'countMSN':26448.0E3, # real count
'countFSI':532.0E3,
'countSTN':77.0E3,
'countGPe':251.0E3,
'countGPi':143.0E3,
'normalrate':{'CSN': [2.,2.+17.7], # rest/excited firing rate
              'PTN': [15.,15.+31.3],
              'CMPf': [4.,4.+30.],
              'MSN': [0.05,1], # minimal bound
              'FSI': [7.8,14.0], # the refined constraint of 10.9 +/- 3.1 Hz was extracted from the following papers: Adler et al., 2016; Yamada et al., 2016 (summarizing date from three different experiments); and Marche and Apicella, 2017
              'STN': [15.2,22.8],
              'GPe': [55.7,74.5],
              'GPi': [59.1,79.5],
             },
# electrotonic constant variables (diameter and length)
'dx':{'MSN':1.E-6,'FSI':1.5E-6,'STN':1.5E-6,'GPe':1.7E-6,'GPi':1.2E-6},
'lx':{'MSN':619E-6,'FSI':961E-6,'STN':750E-6,'GPe':865E-6,'GPi':1132E-6},
'Ri':200.E-2,   # Ohms.m
'Rm':20000.E-4, # Ohms.m^2
# p(X->Y) = distcontact(X->Y) is the relative distance on the dendrite from the soma, where neurons rom X projects to neurons of Y
# Warning: p is not P! It has been renamed here as 'distcontact' to avoid all confusion
'distcontact' : {'MSN->GPe':  0.48,
     'MSN->GPi':  0.59,
     'MSN->MSN':  0.77,
     'FSI->MSN':  0.19,
     'FSI->FSI':  0.16,
     'STN->GPe':  0.30,
     'STN->GPi':  0.59,
     'STN->MSN':  0.16,
     'STN->FSI':  0.41,
     'GPe->STN':  0.58,
     'GPe->GPe':  0.01,
     'GPe->GPi':  0.13,
     'GPe->MSN':  0.06,
     'GPe->FSI':  0.58,
     'CSN->MSN':  0.95,
     'CSN->FSI':  0.82,
     'PTN->MSN':  0.98,
     'PTN->FSI':  0.70,
     'PTN->STN':  0.97,
     'CMPf->STN': 0.46,
     'CMPf->MSN': 0.27,
     'CMPf->FSI': 0.06,
     'CMPf->GPe': 0.0,
     'CMPf->GPi': 0.48
    },
# tau: communication delays
'tau' : {'MSN->GPe':    7.,
       'MSN->GPi':   11.,
       'MSN->MSN':    1.,
       'FSI->MSN':    1.,
       'FSI->FSI':    1.,
       'STN->GPe':    3.,
       'STN->GPi':    3.,
       'STN->MSN':    3.,
       'STN->FSI':    3.,
       'GPe->STN':   10.,
       'GPe->GPe':    1.,
       'GPe->GPi':    3.,
       'GPe->MSN':    3.,
       'GPe->FSI':    3.,
       'CSN->MSN':    7.,
       'CSN->FSI':    7.,
       'PTN->MSN':    3.,
       'PTN->FSI':    3.,
       'PTN->STN':    3.,
       'CMPf->MSN':   7.,
       'CMPf->FSI':   7.,
       'CMPf->STN':   7.,#4
       'CMPf->GPe':   7.,#5
       'CMPf->GPi':   7.,#6
       },
'countCMPf':86.0E3,
'countCSN':None, # avoid keyerror
'countPTN':None,
'GMSN':                         1., # gain on all synapses (default: no gain)
'GFSI':                         1., # ^
'GSTN':                         1., # ^
'GGPe':                         1., # ^
'GGPi':                         1., # ^
'IeMSN':                        24.5, # tonic input currents
'IeFSI':                        8., # ^
'IeSTN':                        9.5, # ^
'IeGPe':                        12., # ^
'IeGPi':                        11., # ^
# There are 3 different format for setting the inDegreeXY (=number of different incoming neurons from X that target one neuron in Y)
# - 'inDegreeAbs': specify the absolute number of different input neurons from X that target one neuron of Y --- be careful when using this setting, as the population size varies widly between the striatum and the downstream nuclei
# - 'outDegreeAbs': specify the absolute number of contacts between an axon from X to each dendritic tree in Y
# - 'outDegreeCons': specify the constrained number of contacts between an axon from X to each dendritic tree in Y as a fraction between 0 (minimal number of contacts to achieve required axonal bouton counts) and 1 (maximal number of contacts with respect to population numbers)
'RedundancyType':   'outDegreeAbs', # by default all axons are hypothesized to target each dendritic tree at 3 different locations
'redundancyCSNMSN':              3, # ^
'redundancyPTNMSN':              3, # ^
'redundancyCMPfMSN':             3, # ^
'redundancyMSNMSN':              3, # ^
'redundancyFSIMSN':              3, # ^
'redundancySTNMSN':              3, # ^
'redundancyGPeMSN':              3, # ^
'redundancyCSNFSI':              3, # ^
'redundancyPTNFSI':              3, # ^
'redundancySTNFSI':              3, # ^
'redundancyGPeFSI':              3, # ^
'redundancyCMPfFSI':             3, # ^
'redundancyFSIFSI':              3, # ^
'redundancyPTNSTN':              3, # ^
'redundancyCMPfSTN':             3, # ^
'redundancyGPeSTN':              3, # ^
'redundancyCMPfGPe':             3, # ^
'redundancySTNGPe':              3, # ^
'redundancyMSNGPe':              3, # ^
'redundancyGPeGPe':              3, # ^
'redundancyMSNGPi':              3, # ^
'redundancySTNGPi':              3, # ^
'redundancyGPeGPi':              3, # ^
'redundancyCMPfGPi':             3, # ^
'cTypeCSNMSN':           'focused', # defining connection types for channel-based models (focused or diffuse) based on LG14 - refer to this paper for justification
'cTypePTNMSN':           'focused', # ^
'cTypeCMPfMSN':          'diffuse', # ^
'cTypeMSNMSN':           'diffuse', # ^
'cTypeFSIMSN':           'diffuse', # ^
'cTypeSTNMSN':           'diffuse', # ^
'cTypeGPeMSN':           'diffuse', # ^
'cTypeCSNFSI':           'focused', # ^
'cTypePTNFSI':           'focused', # ^
'cTypeCMPfFSI':          'diffuse', # ^
'cTypeFSIFSI':           'diffuse', # ^
'cTypeSTNFSI':           'diffuse', # ^
'cTypeGPeFSI':           'diffuse', # ^
'cTypePTNSTN':           'focused', # ^
'cTypeCMPfSTN':          'diffuse', # ^
'cTypeGPeSTN':           'focused', # ^
'cTypeCMPfGPe':          'diffuse', # ^
'cTypeMSNGPe':           'focused', # ^
'cTypeSTNGPe':           'diffuse', # ^
'cTypeGPeGPe':           'diffuse', # ^
'cTypeCMPfGPi':          'diffuse', # ^
'cTypeMSNGPi':           'focused', # ^
'cTypeSTNGPi':           'diffuse', # ^
'cTypeGPeGPi':           'diffuse', # LG14: no data available to decide; setting to diffuse improve selection properties
'spread_focused':              0.15, # connection spread of focused projections
'spread_diffuse':               2., # connection spread of diffuse projections
'parrotCMPf' :                True, # Should the CMPf be simulated using parrot neurons?
'stochastic_delays':          None, # If specified, gives the relative sd of a clipped Gaussian distribution for the delays
# alpha X->Y: average number of synaptic contacts made by one neuron of X to one neuron of Y, when there is a connexion
# for the moment set from one specific parameterization, should be read from Jean's solution file
'alpha':{'MSN->GPe':   171,
         'MSN->GPi':   210,
         'MSN->MSN':   210,
         'FSI->MSN':  4362,
         'FSI->FSI':   116,
         'STN->GPe':   428,
         'STN->GPi':   233,
         'STN->MSN':     0,
         'STN->FSI':    91,
         'GPe->STN':    19,
         'GPe->GPe':    38,
         'GPe->GPi':    16,
         'GPe->MSN':     0,
         'GPe->FSI':   353,
         'CSN->MSN':   342, # here, represents directly \nu
         'CSN->FSI':   250, # here, represents directly \nu
         'PTN->MSN':     5, # here, represents directly \nu
         'PTN->FSI':     5, # here, represents directly \nu
         'PTN->STN':   259, # here, represents directly \nu
         'CMPf->MSN': 4965,
         'CMPf->FSI': 1053,
         'CMPf->STN':   76,
         'CMPf->GPe':   79,
         'CMPf->GPi':  131
        },
# P(X->Y) = ProjPercent: probability that a given neuron from X projects to at least neuron of Y
'ProjPercent' : {'MSN->GPe': 1.,
     'MSN->GPi': 0.82,
     'MSN->MSN': 1.,
     'FSI->MSN': 1.,
     'FSI->FSI': 1.,
     'STN->GPe': 0.83,
     'STN->GPi': 0.72,
     'STN->MSN': 0.17,
     'STN->FSI': 0.17,
     'GPe->STN': 1.,
     'GPe->GPe': 0.84,
     'GPe->GPi': 0.84,
     'GPe->MSN': 0.16,
     'GPe->FSI': 0.16,
     'CSN->MSN': 1.,
     'CSN->FSI': 1.,
     'PTN->MSN': 1.,
     'PTN->FSI': 1.,
     'PTN->STN': 1.,
     'CMPf->STN': 1.,
     'CMPf->MSN': 1.,
     'CMPf->FSI': 1.,
     'CMPf->GPe': 1.,
     'CMPf->GPi': 1.
    },
'wPSP': [1., 0.025, -0.25],
'common_iaf':  {'t_ref':         2.0,
                'V_m':           0.0,
                'V_th':         10.0, # dummy value to avoid NEST complaining about identical V_th and V_reset values
                'E_L':           0.0,
                'V_reset':       0.0,
                'I_e':           0.0,
                'V_min':       -20.0, # as in HSG06
                'tau_syn':   [5./2.718281828459045, 100./2.718281828459045, 5./2.718281828459045] # half-time of AMPA, NMDA and GABAA
               },
'MSN_iaf' : {'tau_m':        13.0, # according to SBE12
             'V_th':         30.0, # value of the LG14 example model, table 9
             'C_m':          13.0  # so that R_m=1, C_m=tau_m
            },
'FSI_iaf' : {'tau_m':         3.1, # from http://www.neuroelectro.org/article/75165/
             'V_th':         16.0, # value of the LG14 example model, table 9
             'C_m':           3.1  # so that R_m=1, C_m=tau_m
            },
'STN_iaf' : {'tau_m':         6.0, # as in HSG06 (but they model rats...)
             'V_th':         26.0, # value of the LG14 example model, table 9
             'C_m':           6.0  # so that R_m=1, C_m=tau_m
            },
'GPe_iaf' : {'tau_m':        14.0, # 20 -> 14 based on Johnson & McIntyre 2008, JNphy)
             'V_th':         11.0, # value of the LG14 example model, table 9
             'C_m':          14.0  # so that R_m=1, C_m=tau_m
            },
'GPi_iaf' : {'tau_m':        14.0, # 20 -> 14 based on Johnson & McIntyre 2008, JNphy)
             'V_th':          6.0, # value of the LG14 example model, table 9
             'C_m':          14.0  # so that R_m=1, C_m=tau_m
            },
'num_neurons':               1000, # M parameter: number of neurons to take from each layer in models M1,S1,M2
'channels':                  False, # If all the neurons are organized as circular channels
'channels_radius':            0.16,
'circle_center':             [],  # circle centers, If channels true, it needs to be completed here, or within the code before BG instantiation
'plastic_syn':               False  #define whether plastic synapses (dopamine syn) or static syn will be used to connect CTX to BG.

}

