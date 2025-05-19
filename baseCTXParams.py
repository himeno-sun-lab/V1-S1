# Base parameters and their meaning
# these defaults can be overrided via a custom python parameter file or via the commandline (in this order: commandline arguments take precedence over customParams.py, which take precendence over the defaults defined here)

####################################################################################
# This file should not be modified! Use commandline arguments or custom param file #
####################################################################################
ctxParams={
        'S1':{
            'structure_info':{
            'region_name': "S1",
            'region_size':[1., 1., 1.366],
            'layer_thickness':[0.128, 0.056, 0.235, 0.207, 0.107, 0.273, 0.360],
            'layer_depth':[0, 0.128, 0.184, 0.419, 0.626, 0.733, 1.006, 1.366],
            'Layer_Name':['L1', 'L2', 'L3', 'L4', 'L5A', 'L5B', 'L6'],
            'Layer_Cellcount_mm2': [4575.334,3270.034,16693.623,21268.353,5631.072,14903.491,27734.67]
            },

            'neuro_info':{
                'L1':{
                    'SBC':{
                        "neuron_model": "iaf_cond_alpha",
                        "EorI": "I",
                        "membrane_time_constant": 10.0,
                        "spike_threshold": -50.0,
                        "reset_value": -70.0,
                        "E_rest": -70.0,
                        "I_ex": 0.0,
                        "absolute_refractory_period": 1.0,
                        "E_ex" : 0.0,
                        "E_in": -80.,
                        "tau_syn_ex": 0.5,
                        "tau_syn_in": 3.33,
                        "Cellcount_mm2": 3019.72044
                    },
                    'ENGC':{
                        "neuron_model": "iaf_cond_alpha",
                        "EorI": "I",
                        "membrane_time_constant": 10.0,
                        "spike_threshold": -50.0,
                        "reset_value": -70.0,
                        "E_rest": -70.0,
                        "I_ex": 0.0,
                        "E_ex": 0.0,
                        "E_in": -80.,
                        "tau_syn_ex": 0.5,
                        "tau_syn_in": 3.33,
                        "absolute_refractory_period": 1.0,
                        "Cellcount_mm2": 1555.61356
                    }
                },
                'L2': {
                    'Pyr': {
                        "neuron_model": "iaf_cond_alpha",
                        "EorI": "E",
                        "membrane_time_constant": 20.0,
                        "spike_threshold": -50.0,
                        "reset_value": -70.0,
                        "E_rest": -70.0,
                        "I_ex": 0.0,
                        "E_ex": 0.0,
                        "E_in": -80.,
                        "tau_syn_ex": 0.5,
                        "tau_syn_in": 3.33,
                        "absolute_refractory_period": 1.0,
                        "Cellcount_mm2": 2714.12822
                    },
                    'PV': {
                        "neuron_model": "iaf_cond_alpha",
                        "EorI": "I",
                        "membrane_time_constant": 10.0,
                        "spike_threshold": -50.0,
                        "reset_value": -70.0,
                        "E_rest": -70.0,
                        "I_ex": 0.0,
                        "E_ex": 0.0,
                        "E_in": -80.,
                        "tau_syn_ex": 0.5,
                        "tau_syn_in": 3.33,
                        "absolute_refractory_period": 1.0,
                        "Cellcount_mm2": 160.5344711
                    },
                    'SST': {
                        "neuron_model": "iaf_cond_alpha",
                        "EorI": "I",
                        "membrane_time_constant": 20.0,
                        "spike_threshold": -50.0,
                        "reset_value": -70.0,
                        "E_rest": -70.0,
                        "I_ex": 0.0,
                        "E_ex": 0.0,
                        "E_in": -80.,
                        "tau_syn_ex": 0.5,
                        "tau_syn_in": 3.33,
                        "absolute_refractory_period": 1.0,
                        "Cellcount_mm2": 64.19044042
                    },
                    'VIP': {
                        "neuron_model": "iaf_cond_alpha",
                        "EorI": "I",
                        "membrane_time_constant": 10.0,
                        "spike_threshold": -50.0,
                        "reset_value": -70.0,
                        "E_rest": -70.0,
                        "I_ex": 0.0,
                        "E_ex": 0.0,
                        "E_in": -80.,
                        "tau_syn_ex": 0.5,
                        "tau_syn_in": 3.33,
                        "absolute_refractory_period": 1.0,
                        "Cellcount_mm2": 331.1808684
                    },
                },

                'L3': {
                    'Pyr': {
                        "neuron_model": "iaf_cond_alpha",
                        "EorI": "E",
                        "membrane_time_constant": 20.0,
                        "spike_threshold": -50.0,
                        "reset_value": -70.0,
                        "E_rest": -70.0,
                        "I_ex": 0.0,
                        "E_ex": 0.0,
                        "E_in": -80.,
                        "tau_syn_ex": 0.5,
                        "tau_syn_in": 3.33,
                        "absolute_refractory_period": 1.0,
                        "Cellcount_mm2": 15191.19693
                    },
                    'PV': {
                        "neuron_model": "iaf_cond_alpha",
                        "EorI": "I",
                        "membrane_time_constant": 10.0,
                        "spike_threshold": -50.0,
                        "reset_value": -70.0,
                        "E_rest": -70.0,
                        "I_ex": 0.0,
                        "E_ex": 0.0,
                        "E_in": -80.,
                        "tau_syn_ex": 0.5,
                        "tau_syn_in": 3.33,
                        "absolute_refractory_period": 1.0,
                        "Cellcount_mm2": 433.8706005
                    },
                    'SST': {
                        "neuron_model": "iaf_cond_alpha",
                        "EorI": "I",
                        "membrane_time_constant": 20.0,
                        "spike_threshold": -50.0,
                        "reset_value": -70.0,
                        "E_rest": -70.0,
                        "I_ex": 0.0,
                        "E_ex": 0.0,
                        "E_in": -80.,
                        "tau_syn_ex": 0.5,
                        "tau_syn_in": 3.33,
                        "absolute_refractory_period": 1.0,
                        "Cellcount_mm2": 173.4851383
                    },
                    'VIP': {
                        "neuron_model": "iaf_cond_alpha",
                        "EorI": "I",
                        "membrane_time_constant": 10.0,
                        "spike_threshold": -50.0,
                        "reset_value": -70.0,
                        "E_rest": -70.0,
                        "I_ex": 0.0,
                        "E_ex": 0.0,
                        "E_in": -80.,
                        "tau_syn_ex": 0.5,
                        "tau_syn_in": 3.33,
                        "absolute_refractory_period": 1.0,
                        "Cellcount_mm2": 895.0703312,
                    },

                },

            'L4': {
                'Pyr': {
                    "neuron_model": "iaf_cond_alpha",
                    "EorI": "E",
                    "membrane_time_constant": 20.0,
                    "spike_threshold": -50.0,
                    "reset_value": -70.0,
                    "E_rest": -70.0,
                    "I_ex": 0.0,
                    "E_ex": 0.0,
                    "E_in": -80.,
                    "tau_syn_ex": 0.5,
                    "tau_syn_in": 3.33,
                    "absolute_refractory_period": 1.0,
                    "Cellcount_mm2": 19545.61641
                },
                'PV': {
                    "neuron_model": "iaf_cond_alpha",
                    "EorI": "I",
                    "membrane_time_constant": 10.0,
                    "spike_threshold": -50.0,
                    "reset_value": -70.0,
                    "E_rest": -70.0,
                    "I_ex": 0.0,
                    "E_ex": 0.0,
                    "E_in": -80.,
                    "tau_syn_ex": 0.5,
                    "tau_syn_in": 3.33,
                    "absolute_refractory_period": 1.0,
                    "Cellcount_mm2": 1364.873436
                },
                'SST': {
                    "neuron_model": "iaf_cond_alpha",
                    "EorI": "I",
                    "membrane_time_constant": 20.0,
                    "spike_threshold": -50.0,
                    "reset_value": -70.0,
                    "E_rest": -70.0,
                    "I_ex": 0.0,
                    "E_ex": 0.0,
                    "E_in": -80.,
                    "tau_syn_ex": 0.5,
                    "tau_syn_in": 3.33,
                    "absolute_refractory_period": 1.0,
                    "Cellcount_mm2": 357.863157,
                },
            },

            'L5A': {
                'Pyr': {
                    "neuron_model": "iaf_cond_alpha",
                    "EorI": "E",
                    "membrane_time_constant": 20.0,
                    "spike_threshold": -50.0,
                    "reset_value": -70.0,
                    "E_rest": -70.0,
                    "I_ex": 0.0,
                    "E_ex": 0.0,
                    "E_in": -80.,
                    "tau_syn_ex": 0.5,
                    "tau_syn_in": 3.33,
                    "absolute_refractory_period": 1.0,
                    "Cellcount_mm2": 4510.488672
                },
                'PV': {
                    "neuron_model": "iaf_cond_alpha",
                    "EorI": "I",
                    "membrane_time_constant": 10.0,
                    "spike_threshold": -50.0,
                    "reset_value": -70.0,
                    "E_rest": -70.0,
                    "I_ex": 0.0,
                    "E_ex": 0.0,
                    "E_in": -80.,
                    "tau_syn_ex": 0.5,
                    "tau_syn_in": 3.33,
                    "absolute_refractory_period": 1.0,
                    "Cellcount_mm2": 710.7989056
                },
                'SST': {
                    "neuron_model": "iaf_cond_alpha",
                    "EorI": "I",
                    "membrane_time_constant": 20.0,
                    "spike_threshold": -50.0,
                    "reset_value": -70.0,
                    "E_rest": -70.0,
                    "I_ex": 0.0,
                    "E_ex": 0.0,
                    "E_in": -80.,
                    "tau_syn_ex": 0.5,
                    "tau_syn_in": 3.33,
                    "absolute_refractory_period": 1.0,
                    "Cellcount_mm2": 409.7844224
                },
            },

            'L5B': {
                'Pyr': {
                    "neuron_model": "iaf_cond_alpha",
                    "EorI": "E",
                    "membrane_time_constant": 20.0,
                    "spike_threshold": -50.0,
                    "reset_value": -70.0,
                    "E_rest": -70.0,
                    "I_ex": 0.0,
                    "E_ex": 0.0,
                    "E_in": -80.,
                    "tau_syn_ex": 0.5,
                    "tau_syn_in": 3.33,
                    "absolute_refractory_period": 1.0,
                    "Cellcount_mm2": 12489.12546
                },
                'PV': {
                    "neuron_model": "iaf_cond_alpha",
                    "EorI": "I",
                    "membrane_time_constant": 10.0,
                    "spike_threshold": -50.0,
                    "reset_value": -70.0,
                    "E_rest": -70.0,
                    "I_ex": 0.0,
                    "E_ex": 0.0,
                    "E_in": -80.,
                    "tau_syn_ex": 0.5,
                    "tau_syn_in": 3.33,
                    "absolute_refractory_period": 1.0,
                    "Cellcount_mm2": 1301.725431
                },
                'SST': {
                    "neuron_model": "iaf_cond_alpha",
                    "EorI": "I",
                    "membrane_time_constant": 20.0,
                    "spike_threshold": -50.0,
                    "reset_value": -70.0,
                    "E_rest": -70.0,
                    "I_ex": 0.0,
                    "E_ex": 0.0,
                    "E_in": -80.,
                    "tau_syn_ex": 0.5,
                    "tau_syn_in": 3.33,
                    "absolute_refractory_period": 1.0,
                    "Cellcount_mm2": 1112.640111
                },
            },

            'L6': {
                'Pyr': {
                    "neuron_model": "iaf_cond_alpha",
                    "EorI": "E",
                    "membrane_time_constant": 20.0,
                    "spike_threshold": -50.0,
                    "reset_value": -70.0,
                    "E_rest": -70.0,
                    "I_ex": 0.0,
                    "E_ex": 0.0,
                    "E_in": -80.,
                    "tau_syn_ex": 0.5,
                    "tau_syn_in": 3.33,
                    "absolute_refractory_period": 1.0,
                    "Cellcount_mm2": 25072.14168
                },
                'PV': {
                    "neuron_model": "iaf_cond_alpha",
                    "EorI": "I",
                    "membrane_time_constant": 10.0,
                    "spike_threshold": -50.0,
                    "reset_value": -70.0,
                    "E_rest": -70.0,
                    "I_ex": 0.0,
                    "E_ex": 0.0,
                    "E_in": -80.,
                    "tau_syn_ex": 0.5,
                    "tau_syn_in": 3.33,
                    "absolute_refractory_period": 1.0,
                    "Cellcount_mm2": 1533.281691
                },
                'SST': {
                    "neuron_model": "iaf_cond_alpha",
                    "EorI": "I",
                    "membrane_time_constant": 20.0,
                    "spike_threshold": -50.0,
                    "reset_value": -70.0,
                    "E_rest": -70.0,
                    "I_ex": 0.0,
                    "E_ex": 0.0,
                    "E_in": -80.,
                    "tau_syn_ex": 0.5,
                    "tau_syn_in": 3.33,
                    "absolute_refractory_period": 1.0,
                    "Cellcount_mm2": 1129.246629
                },
            }
        },
            'connection_info':{
            'internal':'S1_internal_connection.json'
            }
        }
    }
