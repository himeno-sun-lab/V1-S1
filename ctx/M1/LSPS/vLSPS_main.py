#! /usr/bin/env/ python
# -*- coding: utf-8 -*-

# import modules
import matplotlib.pylab as pl
import numpy as np
import commands, re, sys, os
import copy

# set switches
switch_raw_current_trace = False
switch_show_figs = False
switch_show_debug_figs = False
switch_bulk_mode = False

# set parameters
syn_ex_holding_potential = 70.
syn_ex_reversal_potential = 0.
syn_in_holding_potential = 0.
syn_in_reversal_potential = -70.
start_time_current_mean = 1
end_time_current_mean = 50
n_instances = 20
n_steps = 10

# set colormap
cmap = pl.cm.jet

# common information of cells and regions
execfile("../../../programs/model/cells_and_regions.py")

# check input arguments
if len(sys.argv) != 2 and len(sys.argv) !=3:
    print "Usage: vLSPS.py data_directory_name (optional: job_index)"
    print "The number of arguments were", len(sys.argv) - 1
    exit()
elif len(sys.argv) == 2:
    print "# Bulk mode # "
    switch_bulk_mode = True

# cell types
n_regions = len(regions)
n_cell_types = [len(ct) for ct in cell_types]

# paths of simulation result directoriebs
if len(sys.argv) == 3:
    simulation_result_dir = '../../../outputs/simulations/' + sys.argv[1] + '/' + sys.argv[2] 
# bulk mode
elif len(sys.argv) == 2:
    simulation_result_dirs_for_bulk = []
    # search bulk job directory
    output_ld = os.listdir('../../../outputs/simulations/' + sys.argv[1] + '/')
    for old in output_ld:
        if os.path.isdir('../../../outputs/simulations/' + sys.argv[1] + '/'+ old):
            if old.isdigit():
                simulation_result_dirs_for_bulk.append('../../../outputs/simulations/' + sys.argv[1] + '/' + old )
    print len(simulation_result_dirs_for_bulk), "bulk job results are found"

def get_values_from_setting_file(filepath, target_word_array, target_position_array):
    target_value_array = [ [] for i in range(len(target_word_array))]

    # check two array sizes
    if len(target_word_array) != len (target_position_array):
        print "Error: get_values_from_setting_file(): second and third array should be the same length", target_word_array, target_position_array
        exit()

    # read lines
    f_input = open( filepath, 'r')
    lines = f_input.readlines()
    f_input.close()

    # search target words at each line
    for i_line, line in enumerate(lines):
        for i_twa, twa in enumerate(target_word_array):
            if twa in line:                
                for i_word, word in enumerate(line.rstrip("\n").split(' ')):
                    # get value word after detected target word by target_position_array
                    if word == twa:
                        target_value_array[i_twa] = float( line.rstrip("\n").split(' ')[i_word + target_position_array[i_twa] ])

    # check whether word is detected
    for i_tva, tva in enumerate(target_value_array):
        if tva == []:
            print "Error: get_values_from_setting_file():", i_tva, target_word_array[i_tva], "was not detected." 

    return target_value_array

def get_values_from_directory_in_setting_file(filepath, target_word_array, target_directrory_words, target_position_array):
    target_value_array = [ [] for i in range(len(target_word_array))]
    target_directory_lines = []

    # check two array sizes
    if len(target_word_array) != len (target_position_array):
        print "Error: get_values_from_setting_file(): second and third array should be the same length", target_word_array, target_position_array
        exit()

    # read lines
    f_input = open( filepath, 'r')
    lines = f_input.readlines()
    f_input.close()

    # search target directory
    switch_register_directory_lines = False
    for i_line, line in enumerate(lines):
        if target_directory_words[0] in line:
            target_directory_lines.append(i_line)
            switch_register_directory_lines = True
        if switch_register_directory_lines == True:
            for tdw in target_directory_words[1:]:
                if tdw in line:
                    target_directory_lines.append(i_line)
                if target_directory_words[-1] in line:
                    switch_register_directory_lines = False

    # search target words from target_directory_lines
    for i_line, line in enumerate(lines):
        if target_directory_lines[0] <= i_line <= target_directory_lines[-1]:
            for i_twa, twa in enumerate(target_word_array):
                if twa in line:                
                    for i_word, word in enumerate(line.rstrip("\n").split(' ')):
                        # get value word after detected target word by target_position_array
                        if word == twa:
                            target_value_array[i_twa] = float( line.rstrip("\n").split(' ')[i_word + target_position_array[i_twa] ])

    # check whether word is detected
    for i_tva, tva in enumerate(target_value_array):
        if tva == []:
            print "Error: get_values_from_setting_file():", i_tva, target_word_array[i_tva], "was not detected." 

    return target_value_array

# get information from search_parameterX.sli
if switch_bulk_mode == False:
    filepath = '../../../outputs/simulations/' + sys.argv[1] + '/search_parameters' + sys.argv[2] + '.sli'
# In bulk mode, use #0 setting
elif switch_bulk_mode == True:
    filepath = '../../../outputs/simulations/' + sys.argv[1] + '/search_parameters0.sli'
print filepath

target_word_array = [ "/simulation_time", "/length_on_a_side" ]
target_position_array = [ 1, 1 ]
[ simulation_time, length_on_a_side ] = get_values_from_setting_file(filepath, target_word_array, target_position_array)


# get information from vLSPS.sli
filepath = '../../../outputs/simulations/' + sys.argv[1] + '/vLSPS.sli'
target_word_array = [ "/stimu_interval", "/start_time", "/grid_span" ]
target_directory_words = [ "vLSPS_dict_E", "<<", ">>", "def" ]
target_position_array = [ 1, 1, 1 ]
[ stimu_interval, start_stimulus, grid_span ] = get_values_from_directory_in_setting_file(filepath, target_word_array, target_directory_words, target_position_array)

stimu_interval=int(stimu_interval)
start_stimulus=int(start_stimulus)
grid_span=int(grid_span)

# get information from M1_parameters.sli
filepath = '../../../outputs/simulations/' + sys.argv[1] + '/M1_parameters.sli'
target_word_array = [ "/subregion_thicknesses", "/subregion_z_upper_limits" ]
target_position_array = [ 6, 6 ]
[ subregion_thicknesses, subregion_z_upper_limits ]  = get_values_from_setting_file(filepath, target_word_array, target_position_array)

n_horizontal_grid = (int)(length_on_a_side/grid_span)
n_vertical_grid = (int)((subregion_thicknesses+subregion_z_upper_limits)/grid_span)

print "length_on_a_side:", length_on_a_side
print "simulation_time:", simulation_time
print "n_horizontal_grid, n_vertical_grid:",n_horizontal_grid, n_vertical_grid
print "stimu_interval, start_stimulus, grid_span:", stimu_interval, start_stimulus, grid_span

# prepare analysis directory
if switch_bulk_mode == False:
    analysis_dir = '../../../outputs/analysis/' + sys.argv[1] + '_a/' + sys.argv[2] + '/vLSPS/'
# bulk mode
elif switch_bulk_mode == True:
    analysis_dir = '../../../outputs/analysis/' + sys.argv[1] + '_a/all_bulk_jobs/vLSPS/'

try:
    os.makedirs(analysis_dir)
except:
    pass # in case directory exists from before

mm_vLSPS_files = []
# seacrh target mm_vLSPS files at job base directory
if switch_bulk_mode == False:
    mm_vLSPS_files = commands.getstatusoutput('find ' + str(simulation_result_dir) + ' -name mm_vLSPS_*.dat | xargs ls')[1].split('\n')
# bulk mode
elif switch_bulk_mode == True:
    for srdfb in simulation_result_dirs_for_bulk:
        mm_vLSPS_files += commands.getstatusoutput('find ' + str(srdfb) + ' -name mm_vLSPS_*.dat | xargs ls')[1].split('\n')

# prepare arrays for LSPS maps
vLSPS_map_I_ex_mean = [ [ [  np.zeros(shape=(n_horizontal_grid, n_vertical_grid), dtype='float64')  for k in range(n_instances) ] for l in range(n_cell_types[m]) ] for m in range(n_regions) ]
vLSPS_map_I_in_mean = [ [ [  np.zeros(shape=(n_horizontal_grid, n_vertical_grid), dtype='float64')  for k in range(n_instances) ] for l in range(n_cell_types[m]) ] for m in range(n_regions) ]

# prepare arrays for population mean LSPS maps
vLSPS_map_I_ex_population_mean = np.zeros(shape=(n_horizontal_grid, n_vertical_grid), dtype='float64') 
vLSPS_map_I_in_population_mean = np.zeros(shape=(n_horizontal_grid, n_vertical_grid), dtype='float64') 

# count for calculated instance data
instance_counts = [ [ 0 for l in range(n_cell_types[m]) ] for m in range(n_regions) ]

# prepare log file
f_vLSPS_analysis_log = open(analysis_dir + '/vLSPS_analysis_log.txt', 'w')
f_vLSPS_analysis_log.write("### The numbers of used neurons ###\n")
f_vLSPS_analysis_log.close()

# loop region
for i_r, r in enumerate(regions):
    # loop cell types
    for i_ct, ct in enumerate(cell_types[i_r]):
        print r, ct
        #for i initial_value
        count_instances = 0
        # loop files in current directroy
        for i_mvf, mvf in enumerate(mm_vLSPS_files):
            # matching cell type for files
            if ct in mvf:
                dest = mvf 
                f_input = open(dest, 'r')
                lines = f_input.readlines()
                f_input.close()

                # detect files including data
                if lines != []:
                    print dest

                    # sort data 
                    GID, time_mm, V_m_mm, g_ex_mm, g_in_mm, = [], [], [], [], []
                    for i_ml, ml in enumerate(lines):
                        if len(ml.rstrip("\n").split('\t')) == 6:
                            GID, time, V_m, g_ex, g_in, none =  ml.rstrip("\n").split('\t')
                            time_mm.append(float(time))
                            V_m_mm.append(float(V_m))
                            g_ex_mm.append(float(g_ex))
                            g_in_mm.append(float(g_in))

                    # average excitatory current in each grid 
                    last_i_tm = 0
                    for i_vg in range(n_vertical_grid): # z axsis
                        for i_hg in range(n_horizontal_grid): # x axis
                            start = n_horizontal_grid * i_vg * stimu_interval + i_hg * stimu_interval + start_stimulus + start_time_current_mean
                            end = n_horizontal_grid * i_vg * stimu_interval + i_hg * stimu_interval + start_stimulus + end_time_current_mean
                            sum_g_ex_mm = sum(g_ex_mm[start * n_steps : end * n_steps])
                            count_g_ex_mm = (end - start) * n_steps

                            if count_g_ex_mm != 0:
                                vLSPS_map_I_ex_mean[i_r][i_ct][count_instances][i_hg][i_vg] = sum_g_ex_mm/count_g_ex_mm * (syn_ex_holding_potential - syn_ex_reversal_potential)
                                if switch_show_debug_figs == True:
                                    print "plot g_ex"
                                    pl.plot(time_mm, g_ex_mm, "g")
                                    pl.plot(time_mm[start * n_steps : end * n_steps], g_ex_mm[start * n_steps : end * n_steps], "r")
                                    pl.show()

                    last_i_tm = 0
                    # average inhibitory current in each grid 
                    for i_vg in range(n_vertical_grid): # z axsis
                        for i_hg in range(n_horizontal_grid): # x axis
                            start = n_horizontal_grid * i_vg * stimu_interval + i_hg * stimu_interval + start_stimulus + start_time_current_mean
                            end = n_horizontal_grid * i_vg * stimu_interval + i_hg * stimu_interval + start_stimulus + end_time_current_mean
                            sum_g_in_mm = sum(g_in_mm[start * n_steps : end * n_steps])
                            count_g_in_mm = (end - start) * n_steps

                            if count_g_in_mm != 0:
                                vLSPS_map_I_in_mean[i_r][i_ct][count_instances][i_hg][i_vg] = sum_g_in_mm/count_g_in_mm * (syn_in_holding_potential - syn_in_reversal_potential)
                                if switch_show_debug_figs == True:
                                    print "plot g_in_"
                                    pl.plot(time_mm, g_in_mm, "g")
                                    pl.plot(time_mm[start * n_steps : end * n_steps], g_ex_mm[start * n_steps : end * n_steps], "r")
                                    pl.show()

                    # plot raw currents data
                    if switch_raw_current_trace == True:
                        # V_m
                        pl.plot(time_mm, V_m_mm, "k")
                        pl.savefig(analysis_dir + 'V_m_' + r + '_' + ct + str(count_instances) + '.png', dpi=300)
                        pl.clf()
                        
                        # g_ex
                        pl.plot(time_mm, g_ex_mm, "k")
                        pl.savefig(analysis_dir + 'g_ex_' + r + '_' + ct + str(count_instances) + '.png', dpi=300)
                        pl.clf()
                        # g_in
                        pl.plot(time_mm, g_in_mm, "k")
                        pl.savefig(analysis_dir + 'g_in_' + r + '_' + ct + str(count_instances) + '.png', dpi=300)
                        pl.clf()


                    # plot vLSPS map_I_ex_mean
                    ex_max = np.amax(vLSPS_map_I_ex_mean[i_r][i_ct][count_instances])
                    ex_min = np.amin(vLSPS_map_I_ex_mean[i_r][i_ct][count_instances])
                    vmax = (ex_max - ex_min) * 1.15 + ex_min
                    vmin = -(ex_max - ex_min) * 0.15 + ex_min
                    pl.imshow(vLSPS_map_I_ex_mean[i_r][i_ct][count_instances].transpose(), interpolation="nearest", cmap=cmap, vmax=vmax, vmin=vmin)
                    pl.colorbar()
                    pl.savefig(analysis_dir + 'vLSPS_map_I_ex_mean_' + r + '_' + ct + str(count_instances) + '.png', dpi=300)
                    if switch_show_figs == True:
                        pl.show()
                    pl.clf()

                    # plot vLSPS map_I_in_mean
                    in_max = np.amax(vLSPS_map_I_in_mean[i_r][i_ct][count_instances])
                    in_min = np.amin(vLSPS_map_I_in_mean[i_r][i_ct][count_instances])
                    vmax = (in_max - in_min) * 1.15 + in_min
                    vmin = -(in_max - in_min) * 0.15 + in_min
                    pl.imshow(vLSPS_map_I_in_mean[i_r][i_ct][count_instances].transpose(), interpolation="nearest", cmap=cmap, vmax=vmax, vmin=vmin)
                    pl.colorbar()
                    pl.savefig(analysis_dir + 'vLSPS_map_I_in_mean_' + r + '_' + ct + str(count_instances) + '.png', dpi=300)
                    if switch_show_figs == True:
                        pl.show()
                    pl.clf()

                    # exit with maximum count instances
                    count_instances +=1
                    if count_instances == n_instances:
                        break
                        

        # record number of cells for mean
        print count_instances, "cells recorded"
        f_vLSPS_analysis_log = open(analysis_dir + '/vLSPS_analysis_log.txt', 'a')
        f_vLSPS_analysis_log.write("%s %s : %d neurons \n" % (r, ct, count_instances))
        f_vLSPS_analysis_log.close()

        instance_counts[i_r][i_ct] = count_instances

# all mean 
# loop region
for i_r, r in enumerate(regions):
    # loop cell types
    for i_ct, ct in enumerate(cell_types[i_r]):
        for i_ci in range(instance_counts[i_r][i_ct]):
            vLSPS_map_I_ex_population_mean += vLSPS_map_I_ex_mean[i_r][i_ct][i_ci]
            vLSPS_map_I_in_population_mean += vLSPS_map_I_in_mean[i_r][i_ct][i_ci]

        if instance_counts[i_r][i_ct] != 0:
            vLSPS_map_I_ex_population_mean /= instance_counts[i_r][i_ct]
            vLSPS_map_I_in_population_mean /= instance_counts[i_r][i_ct]

            # plot vLSPS map_I_ex_population_mean
            ex_max = np.amax(vLSPS_map_I_ex_population_mean)
            ex_min = np.amin(vLSPS_map_I_ex_population_mean)
            vmax = (ex_max - ex_min) * 1.15 + ex_min
            vmin = -(ex_max - ex_min) * 0.15 + ex_min
            pl.imshow(vLSPS_map_I_ex_population_mean.transpose(), interpolation="nearest", cmap=cmap, vmax=vmax, vmin=vmin)
            pl.colorbar()
            pl.savefig(analysis_dir + 'vLSPS_map_I_ex_population_mean_' + r + '_' + ct + '.png', dpi=300)
            if switch_show_figs == True:
                pl.show()
            pl.clf()

            # plot vLSPS map_I_in_population_mean
            in_max = np.amax(vLSPS_map_I_in_population_mean)
            in_min = np.amin(vLSPS_map_I_in_population_mean)
            vmax = (in_max - in_min) * 1.15 + in_min
            vmin = -(in_max - in_min) * 0.15 + in_min
            pl.imshow(vLSPS_map_I_in_population_mean.transpose(), interpolation="nearest", cmap=cmap, vmax=vmax, vmin=vmin)
            pl.colorbar()
            pl.savefig(analysis_dir + 'vLSPS_map_I_in_population_mean_' + r + '_' + ct + '.png', dpi=300)
            if switch_show_figs == True:
                pl.show()
            pl.clf()

