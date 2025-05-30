#!/usr/bin/env python 

## This file was auto-generated by run.py called with the following arguments:
# run.py --platform Local --whichSim sim_M1 --nbcpu 8

## ID string of experiment:
# 2025_04_15_11_07_31_xp000000

## Reproducibility info:
#  platform = Local
#  git commit ID not available
#  Git status not available

ctxM1Params =\
{
    "M1": {
        "connection_info": {
            "internal": "M1_internal_connection.json"
        },
        "neuro_info": {
            "L1": {
                "ENGC": {
                    "Cellcount_mm2": 611.9966,
                    "E_ex": 0.0,
                    "E_in": -80.0,
                    "E_rest": -70.0,
                    "EorI": "I",
                    "I_ex": 0.0,
                    "absolute_refractory_period": 1.0,
                    "membrane_time_constant": 10.0,
                    "n_type_index": 1,
                    "neuron_model": "iaf_cond_alpha",
                    "reset_value": -70.0,
                    "spike_threshold": -50.0,
                    "tau_syn_ex": 0.5,
                    "tau_syn_in": 3.33
                },
                "SBC": {
                    "Cellcount_mm2": 1187.9934,
                    "E_ex": 0.0,
                    "E_in": -80.0,
                    "E_rest": -70.0,
                    "EorI": "I",
                    "I_ex": 0.0,
                    "absolute_refractory_period": 1.0,
                    "membrane_time_constant": 10.0,
                    "n_type_index": 0,
                    "neuron_model": "iaf_cond_alpha",
                    "reset_value": -70.0,
                    "spike_threshold": -50.0,
                    "tau_syn_ex": 0.5,
                    "tau_syn_in": 3.33
                }
            },
            "L23": {
                "CC": {
                    "Cellcount_mm2": 14659.2,
                    "E_ex": 0.0,
                    "E_in": -80.0,
                    "E_rest": -70.0,
                    "EorI": "E",
                    "I_ex": 0.0,
                    "absolute_refractory_period": 1.0,
                    "membrane_time_constant": 20.0,
                    "n_type_index": 0,
                    "neuron_model": "iaf_cond_alpha",
                    "reset_value": -70.0,
                    "spike_threshold": -50.0,
                    "tau_syn_ex": 0.5,
                    "tau_syn_in": 3.33
                },
                "PV": {
                    "Cellcount_mm2": 1256.83,
                    "E_ex": 0.0,
                    "E_in": -80.0,
                    "E_rest": -70.0,
                    "EorI": "I",
                    "I_ex": 0.0,
                    "absolute_refractory_period": 1.0,
                    "membrane_time_constant": 10.0,
                    "n_type_index": 1,
                    "neuron_model": "iaf_cond_alpha",
                    "reset_value": -70.0,
                    "spike_threshold": -50.0,
                    "tau_syn_ex": 0.5,
                    "tau_syn_in": 3.33
                },
                "SST": {
                    "Cellcount_mm2": 1113.13,
                    "E_ex": 0.0,
                    "E_in": -80.0,
                    "E_rest": -70.0,
                    "EorI": "I",
                    "I_ex": 0.0,
                    "absolute_refractory_period": 1.0,
                    "membrane_time_constant": 20.0,
                    "n_type_index": 2,
                    "neuron_model": "iaf_cond_alpha",
                    "reset_value": -70.0,
                    "spike_threshold": -50.0,
                    "tau_syn_ex": 0.5,
                    "tau_syn_in": 3.33
                },
                "VIP": {
                    "Cellcount_mm2": 1291.18,
                    "E_ex": 0.0,
                    "E_in": -80.0,
                    "E_rest": -70.0,
                    "EorI": "I",
                    "I_ex": 0.0,
                    "absolute_refractory_period": 1.0,
                    "membrane_time_constant": 10.0,
                    "n_type_index": 3,
                    "neuron_model": "iaf_cond_alpha",
                    "reset_value": -70.0,
                    "spike_threshold": -50.0,
                    "tau_syn_ex": 0.5,
                    "tau_syn_in": 3.33
                }
            },
            "L5A": {
                "CC": {
                    "Cellcount_mm2": 1702.0327,
                    "E_ex": 0.0,
                    "E_in": -80.0,
                    "E_rest": -70.0,
                    "EorI": "E",
                    "I_ex": 0.0,
                    "absolute_refractory_period": 1.0,
                    "membrane_time_constant": 20.0,
                    "n_type_index": 1,
                    "neuron_model": "iaf_cond_alpha",
                    "reset_value": -70.0,
                    "spike_threshold": -50.0,
                    "tau_syn_ex": 0.5,
                    "tau_syn_in": 3.33
                },
                "CS": {
                    "Cellcount_mm2": 1702.0327,
                    "E_ex": 0.0,
                    "E_in": -80.0,
                    "E_rest": -70.0,
                    "EorI": "E",
                    "I_ex": 0.0,
                    "absolute_refractory_period": 1.0,
                    "membrane_time_constant": 20.0,
                    "n_type_index": 0,
                    "neuron_model": "iaf_cond_alpha",
                    "reset_value": -70.0,
                    "spike_threshold": -50.0,
                    "tau_syn_ex": 0.5,
                    "tau_syn_in": 3.33
                },
                "CT": {
                    "Cellcount_mm2": 1702.0327,
                    "E_ex": 0.0,
                    "E_in": -80.0,
                    "E_rest": -70.0,
                    "EorI": "E",
                    "I_ex": 0.0,
                    "absolute_refractory_period": 1.0,
                    "membrane_time_constant": 20.0,
                    "n_type_index": 2,
                    "neuron_model": "iaf_cond_alpha",
                    "reset_value": -70.0,
                    "spike_threshold": -50.0,
                    "tau_syn_ex": 0.5,
                    "tau_syn_in": 3.33
                },
                "PV": {
                    "Cellcount_mm2": 638.26,
                    "E_ex": 0.0,
                    "E_in": -80.0,
                    "E_rest": -70.0,
                    "EorI": "I",
                    "I_ex": 0.0,
                    "absolute_refractory_period": 1.0,
                    "membrane_time_constant": 10.0,
                    "n_type_index": 3,
                    "neuron_model": "iaf_cond_alpha",
                    "reset_value": -70.0,
                    "spike_threshold": -50.0,
                    "tau_syn_ex": 0.5,
                    "tau_syn_in": 3.33
                },
                "SST": {
                    "Cellcount_mm2": 651.16,
                    "E_ex": 0.0,
                    "E_in": -80.0,
                    "E_rest": -70.0,
                    "EorI": "I",
                    "I_ex": 0.0,
                    "absolute_refractory_period": 1.0,
                    "membrane_time_constant": 20.0,
                    "n_type_index": 4,
                    "neuron_model": "iaf_cond_alpha",
                    "reset_value": -70.0,
                    "spike_threshold": -50.0,
                    "tau_syn_ex": 0.5,
                    "tau_syn_in": 3.33
                }
            },
            "L5B": {
                "CC": {
                    "Cellcount_mm2": 3036.5812,
                    "E_ex": 0.0,
                    "E_in": -80.0,
                    "E_rest": -70.0,
                    "EorI": "E",
                    "I_ex": 0.0,
                    "absolute_refractory_period": 1.0,
                    "membrane_time_constant": 20.0,
                    "n_type_index": 2,
                    "neuron_model": "iaf_cond_alpha",
                    "reset_value": -70.0,
                    "spike_threshold": -50.0,
                    "tau_syn_ex": 0.5,
                    "tau_syn_in": 3.33
                },
                "CS": {
                    "Cellcount_mm2": 3036.5812,
                    "E_ex": 0.0,
                    "E_in": -80.0,
                    "E_rest": -70.0,
                    "EorI": "E",
                    "I_ex": 0.0,
                    "absolute_refractory_period": 1.0,
                    "membrane_time_constant": 20.0,
                    "n_type_index": 1,
                    "neuron_model": "iaf_cond_alpha",
                    "reset_value": -70.0,
                    "spike_threshold": -50.0,
                    "tau_syn_ex": 0.5,
                    "tau_syn_in": 3.33
                },
                "PT": {
                    "Cellcount_mm2": 6073.1624,
                    "E_ex": 0.0,
                    "E_in": -80.0,
                    "E_rest": -70.0,
                    "EorI": "E",
                    "I_ex": 0.0,
                    "absolute_refractory_period": 1.0,
                    "membrane_time_constant": 20.0,
                    "n_type_index": 0,
                    "neuron_model": "iaf_cond_alpha",
                    "reset_value": -70.0,
                    "spike_threshold": -50.0,
                    "tau_syn_ex": 0.5,
                    "tau_syn_in": 3.33
                },
                "PV": {
                    "Cellcount_mm2": 1503.11,
                    "E_ex": 0.0,
                    "E_in": -80.0,
                    "E_rest": -70.0,
                    "EorI": "I",
                    "I_ex": 0.0,
                    "absolute_refractory_period": 1.0,
                    "membrane_time_constant": 10.0,
                    "n_type_index": 3,
                    "neuron_model": "iaf_cond_alpha",
                    "reset_value": -70.0,
                    "spike_threshold": -50.0,
                    "tau_syn_ex": 0.5,
                    "tau_syn_in": 3.33
                },
                "SST": {
                    "Cellcount_mm2": 1533.47,
                    "E_ex": 0.0,
                    "E_in": -80.0,
                    "E_rest": -70.0,
                    "EorI": "I",
                    "I_ex": 0.0,
                    "absolute_refractory_period": 1.0,
                    "membrane_time_constant": 20.0,
                    "n_type_index": 4,
                    "neuron_model": "iaf_cond_alpha",
                    "reset_value": -70.0,
                    "spike_threshold": -50.0,
                    "tau_syn_ex": 0.5,
                    "tau_syn_in": 3.33
                }
            },
            "L6": {
                "CT": {
                    "Cellcount_mm2": 14102.4,
                    "E_ex": 0.0,
                    "E_in": -80.0,
                    "E_rest": -70.0,
                    "EorI": "E",
                    "I_ex": 0.0,
                    "absolute_refractory_period": 1.0,
                    "membrane_time_constant": 20.0,
                    "n_type_index": 0,
                    "neuron_model": "iaf_cond_alpha",
                    "reset_value": -70.0,
                    "spike_threshold": -50.0,
                    "tau_syn_ex": 0.5,
                    "tau_syn_in": 3.33
                },
                "PV": {
                    "Cellcount_mm2": 1393.55,
                    "E_ex": 0.0,
                    "E_in": -80.0,
                    "E_rest": -70.0,
                    "EorI": "I",
                    "I_ex": 0.0,
                    "absolute_refractory_period": 1.0,
                    "membrane_time_constant": 10.0,
                    "n_type_index": 1,
                    "neuron_model": "iaf_cond_alpha",
                    "reset_value": -70.0,
                    "spike_threshold": -50.0,
                    "tau_syn_ex": 0.5,
                    "tau_syn_in": 3.33
                },
                "SST": {
                    "Cellcount_mm2": 2132.05,
                    "E_ex": 0.0,
                    "E_in": -80.0,
                    "E_rest": -70.0,
                    "EorI": "I",
                    "I_ex": 0.0,
                    "absolute_refractory_period": 1.0,
                    "membrane_time_constant": 20.0,
                    "n_type_index": 2,
                    "neuron_model": "iaf_cond_alpha",
                    "reset_value": -70.0,
                    "spike_threshold": -50.0,
                    "tau_syn_ex": 0.5,
                    "tau_syn_in": 3.33
                }
            }
        },
        "position_type_info": {
            "L1": "M1_Neuron_pos_L1.npz",
            "L23": "M1_Neuron_pos_L23.npz",
            "L5A": "M1_Neuron_pos_L5A.npz",
            "L5B": "M1_Neuron_pos_L5B.npz",
            "L6": "M1_Neuron_pos_L6.npz"
        },
        "structure_info": {
            "Layer_Cellcount_mm2": [
                1799.99,
                18320.334,
                6395.5168,
                15182.9,
                17628.0
            ],
            "Layer_Name": [
                "L1",
                "L23",
                "L5A",
                "L5B",
                "L6"
            ],
            "layer_depth": [
                0.0,
                0.14,
                0.444,
                0.644,
                1.115,
                1.4
            ],
            "layer_thickness": [
                0.14,
                0.304,
                0.2,
                0.471,
                0.285
            ],
            "region_name": "M1",
            "region_size": [
                1.0,
                1.0,
                1.4
            ]
        }
    }
}