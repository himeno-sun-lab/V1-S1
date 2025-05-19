#########################
#2019-8-9
#main function for the cb module test
###########################
import fetch_params
import ini_all
import nest_routine
import nest
#import nest.topology as ntop
import numpy as np
import time

def main():
    # 1) reads parameters
    sim_params = fetch_params.read_sim()
    # 1.5) initialize nest
    nest_routine.initialize_nest(sim_params)
    ctx_M1_params = fetch_params.read_ctx_M1()
    #cb_params = fetch_params.read_cb()
    #bg_params = fetch_params.read_bg()

    start_time = time.time()
    # 2) instantiates regions
    ctx_M1_layers = ini_all.instantiate_ctx_M1(ctx_M1_params, sim_params['scalefactor'], sim_params['initial_ignore'])
#    cb_layers_S1 = ini_all.instantiate_cb('S1', sim_params['scalefactor'])
    cb_layers_M1 = ini_all.instantiate_cb('M1', sim_params['scalefactor'],sim_params)
    ctx_M1_params['circle_center'] = nest_routine.get_channel_centers(sim_params, hex_center=[0,0],ci=6,hex_radius=0.200)
    M1_L5B_PT_gids = nest_routine.get_columns_data('M1_L5B_PT',ctx_M1_params['circle_center'], bg_params['channels_radius'])
    #direction=[3,2,1,0,1,2]
    nest_routine.apply_direction_stimulus_generic(M1_L5B_PT_gids, direction, time_start=100., time_stop=400.,strength=2)

    #4)detectors for columns
    # M1_col_detectors = {}
    # for i in range(len(M1_L5B_PT_gids)):
    #     layer_name = 'M1_L5B_PT_col' + str(i)
    #     M1_col_detectors[i] = nest_routine.layer_spike_detector(M1_L5B_PT_gids[i][0], layer_name, sim_params['initial_ignore'])

    circle_center_params = nest_routine.get_channel_centers(sim_params, hex_center=[0, 0], ci=6, hex_radius=0.240)
    _ = nest_routine.connect_region_ctx_cb(ctx_M1_layers['M1_L5B_PT'], cb_layers_M1['CB_M1_layer_pons'], 'M1')

    #columns_data_pons = nest_routine.get_columns_data('CB_M1_layer_pons', circle_center_params, radius_small=0.16)

    #for i in range(len(M1_L5B_PT_gids)):
    #    source = [gid[0] for gid in M1_L5B_PT_gids[i]]
    #    target = [gid[0] for gid in columns_data_pons[i]]
    #    nest.Connect(pre=source, post=target ,
    #                       conn_spec={'rule':'all_to_all'},syn_spec={'weight': 10.0, 'delay':10.})

    with open('./log/' + 'performance.txt', 'a') as file:
        file.write('Instantiates_Regions_Time ' + str(time.time() - start_time) + '\n')

    #input_circles_gids=nest_routine.get_input_column_layers_ctx_M1(ctx_M1_layers)
    #print (input_circles_gids)

    # 2.5) detectors
    detectors = {}
    start_time = time.time()

    for layer_name in ctx_M1_layers.keys():
        detectors[layer_name] = nest_routine.layer_spike_detector(ctx_M1_layers[layer_name], layer_name,
                                                                  sim_params['initial_ignore'])
    for layer_name in cb_layers_M1.keys():
        detectors[layer_name] = nest_routine.layer_spike_detector(cb_layers_M1[layer_name], layer_name, sim_params['initial_ignore'])
    with open('./log/' + 'performance.txt', 'a') as file:
        file.write('Detectors_Elapse_Time ' + str(time.time() - start_time) + '\n')

    if sim_params['cb_learning']:
        learning_rate = [[1., 1., 0.5, 1., 1., 1.],
                         [1., 1., 0.5, 1., 1., 1.],
                         [1., 1., 0.5, 1., 1., 1.],
                         [1., 1., 0.5, 1., 1., 1.],
                         [1., 1., 0.5, 1., 1., 1.]]
        print ('weight udpate for CB')
        columns_data_gr = nest_routine.get_columns_data('CB_M1_layer_gr', circle_center_params, radius_small=sim_params['channels_radius'])
        print ('columns_data_gr selected')
        # print (columns_data_gr)
        columns_data_pkj = nest_routine.get_columns_data('CB_M1_layer_pkj', circle_center_params, radius_small=sim_params['channels_radius'])
        print ('columns_data_pkj selected')

        columns_data_vn = nest_routine.get_columns_data('CB_M1_layer_vn', circle_center_params, radius_small=sim_params['channels_radius'])
        print ('columns_data_vn selected')

        #learning_rate = [1., 1., 0.5, 1., 1., 1.]


        #_ = nest_routine.update_conn_info(columns_data_gr, columns_data_pkj, 'conn_gr_pkj', 2, learning_rate)

        circle_gid_detector_gr = {}
        circle_j_gids_nb_gr={}
        print (len(columns_data_gr))
        print(len(columns_data_gr[0]))
        print(len(columns_data_gr[0][0]))
        for j in np.arange(len(columns_data_gr)):  # for each circle of gids
            circle_j_gids = [k[0] for k in columns_data_gr[j]]
            spike_detector_name='CB_GR_circle_' + str(j)
            params = {'label': spike_detector_name}
            circle_gid_detector_gr[spike_detector_name] = nest.Create("spike_detector", params=params)
            circle_j_gids_nb_gr[spike_detector_name] = len(columns_data_gr[j])
            nest.Connect(pre=circle_j_gids, post=circle_gid_detector_gr[spike_detector_name])

        circle_gid_detector_pkj = {}
        circle_j_gids_nb_pkj = {}
        for j in np.arange(len(columns_data_pkj)):  # for each circle of gids
            circle_j_gids = [k[0] for k in columns_data_pkj[j]]
            spike_detector_name ='CB_PKJ_circle_' + str(j)
            params = {'label': spike_detector_name}
            circle_gid_detector_pkj[spike_detector_name] = nest.Create("spike_detector", params=params)
            circle_j_gids_nb_pkj[spike_detector_name] = len(columns_data_pkj[j])
            nest.Connect(pre=circle_j_gids, post=circle_gid_detector_pkj[spike_detector_name])

        circle_gid_detector_vn = {}
        circle_j_gids_nb_vn = {}
        for j in np.arange(len(columns_data_vn)):  # for each circle of gids
            circle_j_gids = [k[0] for k in columns_data_vn[j]]
            spike_detector_name = 'CB_VN_circle_' + str(j)
            params = {'label': spike_detector_name}
            circle_gid_detector_vn[spike_detector_name] = nest.Create("spike_detector", params=params)
            circle_j_gids_nb_vn[spike_detector_name] = len(columns_data_vn[j])
            nest.Connect(pre=circle_j_gids, post=circle_gid_detector_vn[spike_detector_name])

    #simulation_time = sim_params['simDuration']+sim_params['initial_ignore']
    #print('Simulation Started:')
    #start_time = time.time()
    #nest.Simulate(int(simulation_time))
    #with open('./log/' + 'performance.txt', 'a') as file:
    #    file.write('Simulation_Elapse_Time ' + str(time.time() - start_time) + '\n')
    #print ('Simulation Finish')
    # 5) output average firing rate
    #start_time = time.time()
    #print('Simulation debrief:')

#    for layer_name in cb_layers_S1.keys():
#        rate = nest_routine.average_fr(detectors[layer_name], sim_params['simDuration'],
#                                       nest_routine.count_layer(cb_layers_S1[layer_name]))
#        print('Layer ' + layer_name + " fires at " + str(rate) + " Hz")
#     for layer_name in ctx_M1_layers.keys():
#         rate = nest_routine.average_fr(detectors[layer_name], sim_params['simDuration'],nest_routine.count_layer(ctx_M1_layers[layer_name]))
#         print('Layer ' + layer_name + " fires at " + str(rate) + " Hz")
#
#     for layer_name in cb_layers_M1.keys():
#         rate = nest_routine.average_fr(detectors[layer_name], sim_params['simDuration'],
#                                        nest_routine.count_layer(cb_layers_M1[layer_name]))
#         print('Layer ' + layer_name + " fires at " + str(rate) + " Hz")
#
#     with open('./log/' + 'performance.txt', 'a') as file:
#         file.write('Output_average_firing_rate ' + str(time.time() - start_time) + '\n')
    nest.Simulate(int(sim_params['initial_ignore']))
    maximum_iterations=5
    iteration_counter=0
    if sim_params['cb_learning']:
        while iteration_counter < maximum_iterations:
            nest.Simulate(int(sim_params['simDuration']))
            for layer_name in ctx_M1_layers.keys():
                rate = nest_routine.average_fr_pre(detectors[layer_name],nest_routine.count_layer(ctx_M1_layers[layer_name]),start_time=iteration_counter * sim_params['simDuration']+500,
                                                            end_time=(iteration_counter + 1)* sim_params['simDuration']+500)
                print('Layer ' + layer_name + " fires at " + str(rate) + " Hz")

            for layer_name in cb_layers_M1.keys():
                rate = nest_routine.average_fr_pre(detectors[layer_name],
                                               nest_routine.count_layer(cb_layers_M1[layer_name]), start_time=iteration_counter * sim_params['simDuration']+500,
                                                            end_time=(iteration_counter + 1)* sim_params['simDuration']+500)
                print('Layer ' + layer_name + " fires at " + str(rate) + " Hz")

            for layer_name in circle_gid_detector_gr.keys():
                rate = nest_routine.average_fr_pre(circle_gid_detector_gr[layer_name], circle_j_gids_nb_gr[layer_name],
                                                            start_time=iteration_counter * sim_params['simDuration']+500,
                                                            end_time=(iteration_counter + 1)* sim_params['simDuration']+500)
                print('Layer ' + layer_name + " fires at " + str(rate) + " Hz")

            for layer_name in circle_gid_detector_pkj.keys():
                rate = nest_routine.average_fr_pre(circle_gid_detector_pkj[layer_name], circle_j_gids_nb_pkj[layer_name],
                                                            start_time=iteration_counter * sim_params['simDuration']+500,
                                                            end_time=(iteration_counter + 1)* sim_params['simDuration']+500)
                print('Layer ' + layer_name + " fires at " + str(rate) + " Hz")
            for layer_name in circle_gid_detector_vn.keys():
                rate = nest_routine.average_fr_pre(circle_gid_detector_vn[layer_name],
                                                   circle_j_gids_nb_vn[layer_name],
                                                   start_time=iteration_counter * sim_params['simDuration']+500,
                                                   end_time=(iteration_counter + 1) * sim_params['simDuration']+500)
                print('Layer ' + layer_name + " fires at " + str(rate) + " Hz")
            _ = nest_routine.update_conn_info(columns_data_gr, columns_data_pkj, 'conn_gr_pkj', iteration_counter + 1,
                                              learning_rate[iteration_counter])

            iteration_counter += 1
if __name__ == '__main__':
    main()
