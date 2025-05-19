#! /usr/bin/env/ python
# -*- coding: utf-8 -*-

# this is the modifies form of file called " transform _ID_to running_number" which is working with command line arguments
# for this file as a arqument we have

import matplotlib.pylab as pl
import matplotlib.pylab as pl
import numpy as np
#import commands
from subprocess import call
import subprocess
import re, sys, os

# common information of cells and regions
regions = [ "M1" ]
cell_types = [[ "L1_SBC","L1_ENGC", "L23_CC", "L23_PV", "L23_SST","L23_VIP", "L5A_CC", "L5A_CS", "L5A_CT", "L5A_PV", "L5A_SST", "L5B_PT", "L5B_CC", "L5B_CS", "L5B_PV", "L5B_SST", "L6_CT", "L6_PV", "L6_SST"]]

# check input arguments
if len(sys.argv) !=3:
    print("Usage: python transform_ID_to_running_number.py data_directory_name (optional: job_index)")
    print("The number of arguments were", len(sys.argv) - 1)
    #exit()

n_regions = len(regions)
n_cell_types = [len(ct) for ct in cell_types]

# simulation directories
simulation_result_dir = '../../main/inputs/' + sys.argv[1] + '/' + sys.argv[2]
    
# get information from search_parameterX.sli
f_input = open('../../main/inputs/' + sys.argv[1] + '/search_parameters' + sys.argv[2] + '.sli', 'r')
lines = f_input.readlines()
f_input.close()

# detect line with simulation time
for i_line, line in enumerate(lines):
    if "/simulation_time" in line:
        for i_word, word in enumerate(line.rstrip("\n").split(' ')):
            if word == "/simulation_time":
                simulation_time = float(line.rstrip("\n").split(' ')[i_word+1])
                print (simulation_time)

# paths for analysis directroies
running_numbers_with_positions_dir = '../../main/outputs/' + sys.argv[1] + '_a/' + sys.argv[2] + '/running_numbers_with_positions/'
running_numbers_with_spike_times_dir = '../../main/outputs/' + sys.argv[1] + '_a/' + sys.argv[2] + '/running_numbers_with_spike_times/'
running_numbers_with_connection_patterns_dir = '../../main/outputs/' + sys.argv[1] + '_a/' + sys.argv[2] + '/running_numbers_with_connection_patterns/'
cell_numbers_dir = '../../main/outputs/' + sys.argv[1] + '_a/' + sys.argv[2] + '/cell_numbers/'

# prepare analysis directories 
if os.path.exists(simulation_result_dir):
    for ad in [ running_numbers_with_positions_dir, running_numbers_with_spike_times_dir, running_numbers_with_connection_patterns_dir, cell_numbers_dir ]:
        try:
            os.makedirs(ad)
        except:
            pass # in case directory exists from before
else:
    print("There is no simulation output directory")
    exit()

# get file lists
positions_files = subprocess.getstatusoutput('find ' + str(simulation_result_dir) + ' -name position*.csv | xargs ls')[1].split('\n')
spike_times_files = subprocess.getstatusoutput('find ' + str(simulation_result_dir) + ' -name sd*.gdf | xargs ls')[1].split('\n')
connection_patterns_files = subprocess.getstatusoutput('find ' + str(simulation_result_dir) + ' -name connection_patterns*.dat | xargs ls')[1].split('\n')


GIDs_running_numbers_tables = [[ [] for ct in cts ] for cts in cell_types]
n_cells_per_cell_type = [[ 0 for ct in cts ] for cts in cell_types]

# get GID 
for i_r, r in enumerate(regions):
    for i_ct, ct in enumerate(cell_types[i_r]):
        dest = running_numbers_with_positions_dir + "running_numbers_with_positions_" + r + "_" + ct + ".dat"
        f_output = open(dest, 'w')
        merged_lines = []
        for i_gpf, gpf in enumerate(positions_files):
            if ct in gpf and r in gpf:
                f_input = open(gpf)
                lines = f_input.readlines()
                f_input.close()
                merged_lines = merged_lines + lines[:-1]

        # prepare GID->running number transformation table using max GID
        if len(merged_lines) > 0:
            GIDs_running_numbers_tables[i_r][i_ct] = np.zeros(int(float((merged_lines[-1].split(','))[0]))+2, dtype='int64')

            # create GID->running number transformation table 
            for i_ml, ml in enumerate(merged_lines):
                if len(ml.rstrip("\n").split(',')) == 4:
                    x, y, z, GID = ml.rstrip("\n").split(',')
                    GIDs_running_numbers_tables[i_r][i_ct][int(float(GID))] = i_ml
                    f_output.writelines( str(GIDs_running_numbers_tables[i_r][i_ct][int(float(GID))]) + str(', ') 
                                         + str(float(x)) + str(', ') + str(float(y)) + str(', ') + str(float(z)) + '\n')
        f_output.close()

        # get number of cells per cell types
        n_cells_per_cell_type[i_r][i_ct] = len(merged_lines)

# save the number of different types of cells"
print("save the number of different types of cells")
for i_r, r in enumerate(regions):
    for i_ct, ct in enumerate(cell_types[i_r]):
        # GIDs 
        dest = cell_numbers_dir + "cell_numbers_" + r + "_" + ct + ".dat"
        f_output = open(dest, 'w')
        f_output.writelines(str(n_cells_per_cell_type[i_r][i_ct]) + '\n' )
        f_output.close()

# get spike times
n_total_cell = 0
for i_r, r in enumerate(regions):
    for i_ct, ct in enumerate(cell_types[i_r]):    
        dest = running_numbers_with_spike_times_dir + "running_numbers_with_spike_times_" + r + "_" + ct + ".dat"
        f_output = open(dest, 'w')
        merged_lines = []
        # merge same cell type's spike times
        for i_stf, stf in enumerate(spike_times_files):
            if ct in stf and r in stf:
                f_input = open(stf)
                lines = f_input.readlines()
                f_input.close()
                merged_lines = merged_lines + lines[:-1]

        temp_spike_times=[]
        # transform from GID to running number in each cell type
        print(len(merged_lines))
        if len(merged_lines) > 0:
            for i_ml, ml in enumerate(merged_lines):
                if len(ml.strip().split('\t')) == 2:
                    GID, spike_time, null = ml.rstrip("\n").split('\t')
                    temp_spike_times.append([int(GIDs_running_numbers_tables[i_r][i_ct][int(GID)]), float(spike_time)])

        temp_spike_times.sort(key=lambda x:(x[1]))
        for tst in temp_spike_times: 
            f_output.writelines( str(tst[0]) + str(', ') + str(tst[1])+ '\n')
        f_output.close()

        print("#" + r + " " + ct)
        print("#n_cells", n_cells_per_cell_type[i_r][i_ct])
        print("#n_spikes", len(merged_lines))
        if len(merged_lines) > 0:
            print("#firing rate", float(len(merged_lines)) / float(n_cells_per_cell_type[i_r][i_ct]) / (float(simulation_time) / 1000.))
        elif len(merged_lines) == 0:
            print("#spikes_per_cell 0")
        print("")
        n_total_cell+=n_cells_per_cell_type[i_r][i_ct]

print("#n_total_cell", n_total_cell)

# get connection patterns 
for i_pre_r, pre_r in enumerate(regions):
    for i_pre_ct, pre_ct in enumerate(cell_types[i_pre_r]):    
        for i_post_r, post_r in enumerate(regions):
            for i_post_ct, post_ct in enumerate(cell_types[i_post_r]):   
                merged_lines = []
                for i_cf, cf in enumerate(connection_patterns_files):
                    if str('../../main/inputs/' + sys.argv[1] + '/' + sys.argv[2] + '/connection_patterns_' + pre_r + '_' + pre_ct) in cf:
                        if str(post_r + '_' + post_ct + '.dat') in cf:
                            # open raw connection patterns data file
                            f_input = open(cf)
                            lines = f_input.readlines()
                            f_input.close()
                            merged_lines = merged_lines + lines[:-1]

                # transform from GID to running number in each cell types
                if merged_lines != []:
                    dest = running_numbers_with_connection_patterns_dir + "running_numbers_with_connection_patterns_" + pre_r + "_" + pre_ct + "_" + post_r + "_" + post_ct + ".dat"
                    f_output = open(dest, 'w')
                    for i_ml, ml in enumerate(merged_lines):
                        if len(ml.rstrip(" \n").split(',')) == 6:
                            pre, post, null, null, null, null = ml.rstrip("\n").split(',')
                            f_output.writelines( str(GIDs_running_numbers_tables[i_pre_r][i_pre_ct][int(pre)]) + str(', ') + str(GIDs_running_numbers_tables[i_post_r][i_post_ct][int(post)]) + '\n')
                    f_output.close()

