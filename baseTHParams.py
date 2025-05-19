# Base parameters and their meaning
# these defaults can be overrided via a custom python parameter file or via the commandline (in this order: commandline arguments take precedence over customParams.py, which take precendence over the defaults defined here)

####################################################################################
# This file should not be modified! Use commandline arguments or custom param file #
####################################################################################
thParams={
    'TH_S1_EZ':
    {
        'structure_info':
        {
            'region_name': "S1_EZ_th",
            'region_size':[1., 1., 0.15],
            'layer_thickness':[0.1, 0.05],
            'subregion_name':['thalamic_nucleus', 'thalamic_reticular_neucleus']
        },
        
    'neuro_info':
        {
            'thalamic_nucleus':
            {
                'TC':
                {
                    "neuron_model": "iaf_cond_alpha",
                    "EorI": "E",
                    "membrane_time_constant": 20.0,
                    "spike_threshold": -50.0,
                    "reset_value": -70.0,
                    "E_rest": -70.0,
                    "I_ex": 15.0,
                    "E_ex": 0.0,
                    "E_in": -80.,
                    "tau_syn_ex": 0.5,
                    "tau_syn_in": 3.33,
                    "absolute_refractory_period": 1.0,
                    "n_type_index" :0
                },
                'IN':
                {
                    "neuron_model": "iaf_cond_alpha",
                    "EorI": "I",
                    "membrane_time_constant": 10.0,
                    "spike_threshold": -50.0,
                    "reset_value": -70.0,
                    "E_rest": -70.0,
                    "I_ex": 15.0,
                    "E_ex": 0.0,
                    "E_in": -80.,
                    "tau_syn_ex": 0.5,
                    "tau_syn_in": 3.33,
                    "absolute_refractory_period": 1.0,
                    "n_type_index" :1
                }
            },
            'thalamic_reticular_neucleus':
            {
                'RE':
                {
                    "neuron_model": "iaf_cond_alpha",
                    "EorI": "I",
                    "membrane_time_constant": 20.0,
                    "spike_threshold": -50.0,
                    "reset_value": -70.0,
                    "E_rest": -70.0,
                    "I_ex": 15.0,
                    "E_ex": 0.0,
                    "E_in": -80.,
                    "tau_syn_ex": 0.5,
                    "tau_syn_in": 3.33,
                    "absolute_refractory_period": 1.0,
                    "n_type_index": 0
                }
            }
        },
        'connection_info':
        {

            'thalamic_nucleus_TC':{
                'thalamic_nucleus_TC':{'p_center': 0.0, 'sigma': 0.01, 'delay':1., 'weight': 1.},
                'thalamic_nucleus_IN':{'p_center': 1.0, 'sigma': 0.01, 'delay':1., 'weight': 1.},
                'thalamic_reticular_neucleus_RE':{'p_center': 1.0, 'sigma': 0.01, 'delay':1., 'weight': 1.}
            },
            'thalamic_nucleus_IN':{
                'thalamic_nucleus_TC': {'p_center': 1.0, 'sigma': 0.01, 'delay': 1., 'weight': 1.},
                'thalamic_nucleus_IN': {'p_center': 0.0, 'sigma': 0.01, 'delay': 1., 'weight': 1.},
                'thalamic_reticular_neucleus_RE': {'p_center': 0.0, 'sigma': 0.01, 'delay': 1., 'weight': 1.}
            },
            'thalamic_reticular_neucleus_RE':{
                'thalamic_nucleus_TC': {'p_center': 1.0, 'sigma': 0.01, 'delay': 1., 'weight': 1.},
                'thalamic_nucleus_IN': {'p_center': 0.0, 'sigma': 0.01, 'delay': 1., 'weight': 1.},
                'thalamic_reticular_neucleus_RE': {'p_center': 0.0, 'sigma': 0.01, 'delay': 1., 'weight': 1.}
            },
        },



    },


    'TH_S1_IZ':
        {
            'structure_info':
                {
                    'region_name': "S1_IZ_th",
                    'region_size': [1., 1., 0.2],
                    'layer_thickness': [0.1, 0.05],
                    'subregion_name': ['thalamic_nucleus', 'thalamic_reticular_neucleus']
                },

            'neuro_info':
                {
                    'thalamic_nucleus':
                        {
                            'TC':
                                {
                                    "neuron_model": "iaf_cond_alpha",
                                    "EorI": "E",
                                    "membrane_time_constant": 20.0,
                                    "spike_threshold": -50.0,
                                    "reset_value": -70.0,
                                    "E_rest": -70.0,
                                    "I_ex": 15.0,
                                    "E_ex": 0.0,
                                    "E_in": -80.,
                                    "tau_syn_ex": 0.5,
                                    "tau_syn_in": 3.33,
                                    "absolute_refractory_period": 1.0,
                                    "n_type_index": 0
                                },
                            'IN':
                                {
                                    "neuron_model": "iaf_cond_alpha",
                                    "EorI": "I",
                                    "membrane_time_constant": 10.0,
                                    "spike_threshold": -50.0,
                                    "reset_value": -70.0,
                                    "E_rest": -70.0,
                                    "I_ex": 15.0,
                                    "E_ex": 0.0,
                                    "E_in": -80.,
                                    "tau_syn_ex": 0.5,
                                    "tau_syn_in": 3.33,
                                    "absolute_refractory_period": 1.0,
                                    "n_type_index": 1
                                }
                        },
                    'thalamic_reticular_neucleus':
                        {
                            'RE':
                                {
                                    "neuron_model": "iaf_cond_alpha",
                                    "EorI": "I",
                                    "membrane_time_constant": 20.0,
                                    "spike_threshold": -50.0,
                                    "reset_value": -70.0,
                                    "E_rest": -70.0,
                                    "I_ex": 15.0,
                                    "E_ex": 0.0,
                                    "E_in": -80.,
                                    "tau_syn_ex": 0.5,
                                    "tau_syn_in": 3.33,
                                    "absolute_refractory_period": 1.0,
                                    "n_type_index": 0
                                }
                        }
                },
            'connection_info':
                {

                    'thalamic_nucleus_TC': {
                        'thalamic_nucleus_TC': {'p_center': 0.0, 'sigma': 0.01, 'delay': 1., 'weight': 1.},
                        'thalamic_nucleus_IN': {'p_center': 1.0, 'sigma': 0.01, 'delay': 1., 'weight': 1.},
                        'thalamic_reticular_neucleus_RE': {'p_center': 1.0, 'sigma': 0.01, 'delay': 1., 'weight': 1.}
                    },
                    'thalamic_nucleus_IN': {
                        'thalamic_nucleus_TC': {'p_center': 1.0, 'sigma': 0.01, 'delay': 1., 'weight': 1.},
                        'thalamic_nucleus_IN': {'p_center': 0.0, 'sigma': 0.01, 'delay': 1., 'weight': 1.},
                        'thalamic_reticular_neucleus_RE': {'p_center': 0.0, 'sigma': 0.01, 'delay': 1., 'weight': 1.}
                    },
                    'thalamic_reticular_neucleus_RE': {
                        'thalamic_nucleus_TC': {'p_center': 1.0, 'sigma': 0.01, 'delay': 1., 'weight': 1.},
                        'thalamic_nucleus_IN': {'p_center': 0.0, 'sigma': 0.01, 'delay': 1., 'weight': 1.},
                        'thalamic_reticular_neucleus_RE': {'p_center': 0.0, 'sigma': 0.01, 'delay': 1., 'weight': 1.}
                    },
                },

        },
    'TH_M1_EZ':
        {
            'structure_info':
                {
                    'region_name': "M1_EZ_th",
                    'region_size': [1., 1., 0.15],
                    'layer_thickness': [0.1, 0.05],
                    'subregion_name': ['thalamic_nucleus', 'thalamic_reticular_neucleus']
                },

            'neuro_info':
                {
                    'thalamic_nucleus':
                        {
                            'TC':
                                {
                                    "neuron_model": "iaf_cond_alpha",
                                    "EorI": "E",
                                    "membrane_time_constant": 20.0,
                                    "spike_threshold": -50.0,
                                    "reset_value": -70.0,
                                    "E_rest": -70.0,
                                    "I_ex": 15.0,
                                    "E_ex": 0.0,
                                    "E_in": -80.,
                                    "tau_syn_ex": 0.5,
                                    "tau_syn_in": 3.33,
                                    "absolute_refractory_period": 1.0,
                                    "n_type_index": 0
                                },
                            'IN':
                                {
                                    "neuron_model": "iaf_cond_alpha",
                                    "EorI": "I",
                                    "membrane_time_constant": 10.0,
                                    "spike_threshold": -50.0,
                                    "reset_value": -70.0,
                                    "E_rest": -70.0,
                                    "I_ex": 15.0,
                                    "E_ex": 0.0,
                                    "E_in": -80.,
                                    "tau_syn_ex": 0.5,
                                    "tau_syn_in": 3.33,
                                    "absolute_refractory_period": 1.0,
                                    "n_type_index": 1
                                }
                        },
                    'thalamic_reticular_neucleus':
                        {
                            'RE':
                                {
                                    "neuron_model": "iaf_cond_alpha",
                                    "EorI": "I",
                                    "membrane_time_constant": 20.0,
                                    "spike_threshold": -50.0,
                                    "reset_value": -70.0,
                                    "E_rest": -70.0,
                                    "I_ex": 15.0,
                                    "E_ex": 0.0,
                                    "E_in": -80.,
                                    "tau_syn_ex": 0.5,
                                    "tau_syn_in": 3.33,
                                    "absolute_refractory_period": 1.0,
                                    "n_type_index": 0
                                }
                        }
                },
            'connection_info':
                {

                    'thalamic_nucleus_TC': {
                        'thalamic_nucleus_TC': {'p_center': 0.0, 'sigma': 0.01, 'delay': 1., 'weight': 1.},
                        'thalamic_nucleus_IN': {'p_center': 1.0, 'sigma': 0.01, 'delay': 1., 'weight': 1.},
                        'thalamic_reticular_neucleus_RE': {'p_center': 1.0, 'sigma': 0.01, 'delay': 1., 'weight': 1.}
                    },
                    'thalamic_nucleus_IN': {
                        'thalamic_nucleus_TC': {'p_center': 1.0, 'sigma': 0.01, 'delay': 1., 'weight': 1.},
                        'thalamic_nucleus_IN': {'p_center': 0.0, 'sigma': 0.01, 'delay': 1., 'weight': 1.},
                        'thalamic_reticular_neucleus_RE': {'p_center': 0.0, 'sigma': 0.01, 'delay': 1., 'weight': 1.}
                    },
                    'thalamic_reticular_neucleus_RE': {
                        'thalamic_nucleus_TC': {'p_center': 1.0, 'sigma': 0.01, 'delay': 1., 'weight': 1.},
                        'thalamic_nucleus_IN': {'p_center': 0.0, 'sigma': 0.01, 'delay': 1., 'weight': 1.},
                        'thalamic_reticular_neucleus_RE': {'p_center': 0.0, 'sigma': 0.01, 'delay': 1., 'weight': 1.}
                    },
                },

        },

    'TH_M1_IZ':
        {
            'structure_info':
                {
                    'region_name': "M1_IZ_th",
                    'region_size': [1., 1., 0.2],
                    'layer_thickness': [0.1, 0.05],
                    'subregion_name': ['thalamic_nucleus', 'thalamic_reticular_neucleus']
                },

            'neuro_info':
                {
                    'thalamic_nucleus':
                        {
                            'TC':
                                {
                                    "neuron_model": "iaf_cond_alpha",
                                    "EorI": "E",
                                    "membrane_time_constant": 20.0,
                                    "spike_threshold": -50.0,
                                    "reset_value": -70.0,
                                    "E_rest": -70.0,
                                    "I_ex": 15.0,
                                    "E_ex": 0.0,
                                    "E_in": -80.,
                                    "tau_syn_ex": 0.5,
                                    "tau_syn_in": 3.33,
                                    "absolute_refractory_period": 1.0,
                                    "n_type_index": 0
                                },
                            'IN':
                                {
                                    "neuron_model": "iaf_cond_alpha",
                                    "EorI": "I",
                                    "membrane_time_constant": 10.0,
                                    "spike_threshold": -50.0,
                                    "reset_value": -70.0,
                                    "E_rest": -70.0,
                                    "I_ex": 15.0,
                                    "E_ex": 0.0,
                                    "E_in": -80.,
                                    "tau_syn_ex": 0.5,
                                    "tau_syn_in": 3.33,
                                    "absolute_refractory_period": 1.0,
                                    "n_type_index": 1
                                }
                        },
                    'thalamic_reticular_neucleus':
                        {
                            'RE':
                                {
                                    "neuron_model": "iaf_cond_alpha",
                                    "EorI": "I",
                                    "membrane_time_constant": 20.0,
                                    "spike_threshold": -50.0,
                                    "reset_value": -70.0,
                                    "E_rest": -70.0,
                                    "I_ex": 15.0,
                                    "E_ex": 0.0,
                                    "E_in": -80.,
                                    "tau_syn_ex": 0.5,
                                    "tau_syn_in": 3.33,
                                    "absolute_refractory_period": 1.0,
                                    "n_type_index": 0
                                }
                        }
                },
            'connection_info':
                {

                    'thalamic_nucleus_TC': {
                        'thalamic_nucleus_TC': {'p_center': 0.0, 'sigma': 0.01, 'delay': 1., 'weight': 1.},
                        'thalamic_nucleus_IN': {'p_center': 1.0, 'sigma': 0.01, 'delay': 1., 'weight': 1.},
                        'thalamic_reticular_neucleus_RE': {'p_center': 1.0, 'sigma': 0.01, 'delay': 1., 'weight': 1.}
                    },
                    'thalamic_nucleus_IN': {
                        'thalamic_nucleus_TC': {'p_center': 1.0, 'sigma': 0.01, 'delay': 1., 'weight': 1.},
                        'thalamic_nucleus_IN': {'p_center': 0.0, 'sigma': 0.01, 'delay': 1., 'weight': 1.},
                        'thalamic_reticular_neucleus_RE': {'p_center': 0.0, 'sigma': 0.01, 'delay': 1., 'weight': 1.}
                    },
                    'thalamic_reticular_neucleus_RE': {
                        'thalamic_nucleus_TC': {'p_center': 1.0, 'sigma': 0.01, 'delay': 1., 'weight': 1.},
                        'thalamic_nucleus_IN': {'p_center': 0.0, 'sigma': 0.01, 'delay': 1., 'weight': 1.},
                        'thalamic_reticular_neucleus_RE': {'p_center': 0.0, 'sigma': 0.01, 'delay': 1., 'weight': 1.}
                    },
                },

        }
}
