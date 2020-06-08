'''This file is part of Jarkus Analysis Toolbox.
   
JAT is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.
   
JAT is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.
   
You should have received a copy of the GNU General Public License
along with JAT.  If not, see <http://www.gnu.org/licenses/>.
   
JAT  Copyright (C) 2020 Christa van IJzendoorn
c.o.vanijzendoorn@tudelft.nl
Delft University of Technology
Faculty of Civil Engineering and Geosciences
Stevinweg 1
2628CN Delft
The Netherlands
'''
"""

# -*- coding: utf-8 -*-
"""
Created on Mon Oct 14 16:58:52 2019

@author: cijzendoornvan
"""
##################################
####          PACKAGES        ####
##################################
import json
import pandas as pd
import os.path
import math
from jarkus.transects import Transects
import pickle
import numpy as np
import matplotlib.pyplot as plt 
from matplotlib import gridspec
import scipy.stats

import matplotlib.colors as clr
from IPython import get_ipython
get_ipython().run_line_magic('matplotlib', 'auto') ## %matplotlib auto TO GET WINDOW FIGURE
#################################
####        FUNCTIONS        ####
#################################


##################################
####    USER-DEFINED REQUEST  ####
##################################
# load basic settings from .txt file
with open("C:/Users/cijzendoornvan/Documents/GitHub/jarkus/jarkus/Settings.txt") as file:
    settings = json.load(file)

Dir_per_transect = settings['Dir_C3']
Dir_per_variable = settings['Dir_D1']
Dir_neat_plots = settings['Dir_D5']

years_requested = list(range(1965, 2020))

##################################
####      INITIALISATION      ####
##################################
# Collect the JARKUS data from the server
Jk = Transects(url= settings['url'])
ids = Jk.get_data('id') # ids of all available transects

with open("C:/Users/cijzendoornvan/Documents/GitHub/jarkus/jarkus/Filter.txt") as file:
    filter_transects = json.load(file)

filter_all = []
for key in filter_transects:
    filter_all += list(range(int(filter_transects[key]["begin"]), int(filter_transects[key]["eind"])))

ids_filtered = [x for x in ids if x not in filter_all]

# Create conversion table
ids = Jk.get_data('id') # ids of all available transects
area_bounds = [2000000, 3000000, 4000000, 5000000, 6000000, 7000000, 8000000, 9000000, 10000000, 11000000, 12000000, 13000000, 14000000, 15000000, 16000000, 17000000, 18000000]

for i, val in enumerate(area_bounds):
    if i == 0:
        ids_filt = ids[np.where(np.logical_and(ids>=area_bounds[i], ids<area_bounds[i+1]))] #- area_bounds[i]
        ids_filt = [max(ids_filt) - ids for ids in ids_filt]
        
        ids_alongshore = ids_filt
        ids_filt_old = ids_filt
    elif i == 16:
        print("Converted all areacodes to alongshore values")
    else:
        # De volgende moet plus de max van de vorige
        ids_filt = ids[np.where(np.logical_and(ids>=area_bounds[i], ids<area_bounds[i+1]))] - area_bounds[i]
        ids_filt = [max(ids_filt) - ids for ids in ids_filt]
        ids_filt = max(ids_filt_old) + ids_filt
        
        ids_alongshore = np.concatenate((ids_alongshore, ids_filt))
        ids_filt_old = ids_filt
        
conversion_ids2alongshore = dict(zip(map(str,ids), ids_alongshore))

#%%
##################################
####       PREPARATIONS       ####
##################################

#variables = ['max_TWL_topo_dis']
#figure_title = 'Alongshore and temporal variation of the maximum Total Water Level (m)'
#colorbar_label = 'Total water level (m)'
#colormap_var = "Blues"
#file_name = 'TWL'

# variables = ['MHW_y_var']
# figure_title = 'Alongshore and temporal variation of MHW (as included in Jarkus dataset) (m)'
# colorbar_label = 'MHW elevation (m)'
# colormap_var = "Blues"
# file_name = 'mean_high_water_var'

#variables = ['MLW_y_var']
#figure_title = 'Alongshore and temporal variation of MLW (as included in Jarkus dataset) (m)'
#colorbar_label = 'MLW elevation (m)'
#colormap_var = "Blues"
#file_name = 'mean_low_water_var'

#variables = ['MSL_y_var']
#figure_title = 'Alongshore and temporal variation of MSL (based on MHW and MLW in Jarkus dataset) (m)'
#colorbar_label = 'MSL elevation (m)'
#colormap_var = "Blues"
#file_name = 'mean_sea_level_var'

# variables = ['DF_der_y']
# figure_title = 'Alongshore and temporal variation of dune foot elevation (m)'
# colorbar_label = 'elevation (m)'
# colormap_var = "Greens"
# file_name = 'dune_foot_elevation'

variables = ['DF_der_x']
figure_title = 'Alongshore and temporal variation of the cross-shore dune foot location resp. to MSL (m)'
colorbar_label = 'cross-shore location (m)'
colormap_var = "Greens"
file_name = 'dune_foot_location'

# variables = ['DF_pybeach_mix_y']
# figure_title = 'Alongshore and temporal variation of dune foot elevation (m) based on pybeach ML (mixed classifier)'
# colorbar_label = 'Dune foot elevation (m)'
# colormap_var = "Greens"
# file_name = 'dune_foot_elevation_ML'

# variables = ['DF_pybeach_mix_x']
# figure_title = 'Alongshore and temporal variation of the cross-shore dune foot location resp. to MSL (m) based on pybeach M (mixed classifier)L'
# colorbar_label = 'Cross-shore dune foot location (m)'
# colormap_var = "Greens"
# file_name = 'dune_foot_location_ML'

# variables = ['DFront_der_prim_grad']
# figure_title = 'Alongshore and temporal variation of the dune front gradient (-)'
# colorbar_label = 'gradient (m/m)'
# colormap_var = "Greens"
# file_name = 'dune_front_grad'

# variables = ['BW_der_var']
# figure_title = 'Alongshore and temporal variation of the beach width (m)'
# colorbar_label = 'Beach width (m)'
# colormap_var = "Greens"
# file_name = 'beach_width'

# variables = ['DT_prim_y']
# figure_title = 'Alongshore and temporal variation of the primary dune crest elevation (m)'
# colorbar_label = 'elevation (m)'
# colormap_var = "Greens"
# file_name = 'dune_top'

# variables = ['DT_prim_x_normalized']
# figure_title = 'Alongshore and temporal variation of the primary dune crest cross shore location (m)'
# colorbar_label = 'Primary dune crest location (m)'
# colormap_var = "Greens"
# file_name = 'dune_top_location'

# variables = ['DVol_fix']
# figure_title = 'Alongshore and temporal variation of the dune volume (m^3/m) based on 3 m DF level'
# colorbar_label = 'Dune volume (m\N{SUPERSCRIPT THREE}/m)'
# colormap_var = "Greens"
# file_name = 'dune_volume_fix'

# variables = ['DVol_der']
# figure_title = 'Alongshore and temporal variation of the dune volume (m^3/m) based on the derivative'
# colorbar_label = 'Dune volume (m^3/m)'
# colormap_var = "Greens"
# file_name = 'dune_volume_der'

# variables = ['FSVol_fix']
# figure_title = 'Alongshore and temporal variation of foreshore volume (m)'
# colorbar_label = 'Foreshore volume (m)'
# colormap_var = "Greens"
# file_name = 'foreshore_volume'

# variables = ['FS_grad']
# figure_title = 'Alongshore and temporal variation of foreshore gradient (-)'
# colorbar_label = 'Foreshore gradient (-)'
# colormap_var = "Reds"
# file_name = 'foreshore_gradient'

# variables = ['APVol_fix']
# figure_title = 'Alongshore and temporal variation of active profile volume (m)'
# colorbar_label = 'Active profile volume (m)'
# colormap_var = "Greens"
# file_name = 'active_profile_volume'

# variables = ['AP_grad']
# figure_title = 'Alongshore and temporal variation of active profile gradient (-)'
# colorbar_label = 'Active profile gradient (-)'
# colormap_var = "Reds"
# file_name = 'active_profile_gradient'

# variables = ['B_grad_fix']
# figure_title = 'Alongshore and temporal variation of beach gradient (-)'
# colorbar_label = 'Beach gradient (-)'
# colormap_var = "Reds"
# file_name = 'beach_gradient'

# load dataframe with distribution of values of variable
pickle_file0 = Dir_per_variable + variables[0] + '_dataframe.pickle'    
Variable_values0 = pickle.load(open(pickle_file0, 'rb')) #load pickle of dimension    

# Get transect that have to be visualized
transects_to_visualize = [x for x in ids_filtered if x in ids]
transects_to_visualize_str = [str(x) for x in ids_filtered if x in ids]

#%%
##################################
####   FILTERING SUBREGIONS   ####
##################################
Variable_values0_subregion = Variable_values0 # NO FILTER, keep only this line if you want to apply no filtering

code_beginraai = 7002023  # Removal of Hondsbossche
code_eindraai = 7002629
region_remove = [str(i) for i in ids_filtered if i >= code_beginraai and i <= code_eindraai]
for c in Variable_values0_subregion.columns: # for each column check whether it is in the remove list, if it is set all values to nan
    if c in region_remove:
        Variable_values0_subregion[c] = np.nan
        
# code_beginraai = 2000000  # Remove Wadden Coast
# code_eindraai = 3000000
# region_remove = [str(i) for i in ids_filtered if i >= code_beginraai and i <= code_eindraai]
# for c in Variable_values0_subregion.columns: # for each column check whether it is in the remove list, if it is set all values to nan
#     if c in region_remove:
#         Variable_values0_subregion[c] = np.nan
        
# code_beginraai = 10000000  # Remove Delta Coast
# code_eindraai = 18000000
# region_remove = [str(i) for i in ids_filtered if i >= code_beginraai and i <= code_eindraai]
# for c in Variable_values0_subregion.columns: # for each column check whether it is in the remove list, if it is set all values to nan
#     if c in region_remove:
#         Variable_values0_subregion[c] = np.nan

# #### Filtering of Nourishments ####
# Nourishments = pd.read_excel("C:/Users/cijzendoornvan/Documents/Data/JARKUS/Suppletiedatabase.xlsx")
# region_remove = []
# for index, row in Nourishments.iterrows():
#     if math.isnan(row['BeginRaai']) or math.isnan(row['EindRaai']):# or row['Volume/m'] > 50: # or row['JaarBeginUitvoering'] < 2010: 
#         continue
#     else:
#         code_beginraai = int(row['KustVakNummer'] * 1000000 + row['BeginRaai'] * 100)
#         code_eindraai = int(row['KustVakNummer'] * 1000000 + row['EindRaai'] * 100)
#         filtered = [i for i in ids_filtered if i >= code_beginraai and i <= code_eindraai]
#         region_remove = region_remove + [str(i) for i in filtered if i not in region_remove]        
# for c in Variable_values0_subregion.columns: # for each column check whether it is in the remove list, if it is set all values to nan
#     if c in region_remove:
#         Variable_values0_subregion[c] = np.nan
        
# #### Filtering of Dynamic coasts ####
# DynamischeKust = pd.read_excel("C:/Users/cijzendoornvan/Documents/Data/JARKUS/DynamischeKust.xlsx")
# region_remove = []
# for index, row in DynamischeKust.iterrows():
#     if math.isnan(row['BeginRaai']) or math.isnan(row['EindRaai']):# or row['Volume/m'] > 50: # or row['JaarBeginUitvoering'] < 2010: 
#         continue
#     else:
#         code_beginraai = int(row['KustVakNummer'] * 1000000 + row['BeginRaai'] * 100)
#         code_eindraai = int(row['KustVakNummer'] * 1000000 + row['EindRaai'] * 100)
#         filtered = [i for i in ids_filtered if i >= code_beginraai and i <= code_eindraai]
#         region_remove = region_remove + [str(i) for i in filtered if i not in region_remove]
# for c in Variable_values0_subregion.columns: # for each column check whether it is in the remove list, if it is set all values to nan
#     if c not in region_remove:
#         Variable_values0_subregion[c] = np.nan
        
# Convert transect codes (i.e. column names) into alongshore decameter values
#Variable_values0_subregion = Variable_values0_subregion.rename(columns=conversion_ids2alongshore)

# Rearrange dataframe based on alongshore decameters
#Variable_values0_subregion.sort_index(axis=1, inplace=True)

# Calculate spatial and temporal average
average_through_space = Variable_values0_subregion.mean(axis=0)
average_through_time = Variable_values0_subregion.mean(axis=1)

# Calculate overall average and stddev, used for range of colorbar
average         = np.nanmean(Variable_values0_subregion.values)
stddev          = np.nanstd(Variable_values0_subregion.values, ddof=1)
range_value     = 2*stddev
range_value_avg = stddev
vmin            = average - range_value
vmax            = average + range_value
vmin_avg        = average - range_value_avg
vmax_avg        = average + range_value_avg

# Fill these in if you want to fix the limits of the plots
#vmin            = 2.0
#vmax            = 3.5
vmin_avg         = -50
vmax_avg         = 100

# Create an array with locations and an array with labels of the ticks
ticks_x = [350, 1000, 1600]
labels_x = ['Wadden Coast', 'Holland Coast', 'Delta Coast']

ticks_y = range(0, len(years_requested))[0::5]
labels_y = [str(yr) for yr in years_requested][0::5]

#%%
##################################
####       PLOTTING       ####
##################################

# Set-up of figure
fig = plt.figure() 

fig.suptitle(figure_title, fontsize=26)
gs = gridspec.GridSpec(2, 2, width_ratios=[3, 1], height_ratios=[3,2]) 

# PLOT TEMPORAL AND SPATIAL DISTRIBUTION OF VARIABLE
ax1 = fig.add_subplot(gs[0])
cmap = plt.cm.get_cmap(colormap_var) # Define color use for colorbar
colorplot = ax1.pcolor(Variable_values0_subregion, vmin=vmin, vmax=vmax, cmap=cmap)
# Set labels and ticks of x and y axis
ax1.set_yticks(ticks_y)
ax1.set_yticklabels(labels_y)
ax1.tick_params(axis='y', which='both',length=5, labelsize = 20)
ax1.set_xticks(ticks_x)
ax1.set_xticklabels(labels_x)
ax1.tick_params(axis='x', which='both',length=0, labelsize = 24)
# plot boundaries between coastal regions
ax1.axvline(x=686, color='r') # Boundary kustvak 6 en 7, Den Helder, trsct 7000000, decameter 23417
ax1.axvline(x=1288, color='r')# Boundary kustvak 9 en 10, Hoek van Holland, trsct 10000000, decameter 35908

# PLOT YEARLY AVERAGE OF VARIABLE
ax2 = fig.add_subplot(gs[1])
ax2.plot(average_through_time, average_through_time.index, color=colormap_var[:-1], linewidth=2.5)
# Calculate and plot regression line
mask = ~np.isnan(average_through_time.index) & ~np.isnan(average_through_time)
slope, intercept, r_value, p_value, std_err = scipy.stats.linregress(average_through_time.index[mask], average_through_time[mask])
regression_line = slope*average_through_time.index + intercept
ax2.plot(regression_line, average_through_time.index, 'r--', linewidth=2)
# plot slope of the regression line
plt.text(regression_line[0]+0.1, average_through_time.index[0], str(round(slope*1000, 1)) + ' mm per year', fontsize=20)
#plt.text(regression_line[0]+20, average_through_time.index[0], str(round(slope, 1)) + ' m\N{SUPERSCRIPT THREE} per year', fontsize=20)
#plt.text(regression_line[0]+50, average_through_time.index[0], str(round(slope, 1)) + ' m\N{SUPERSCRIPT THREE}/m per year', fontsize=20)
#plt.text(regression_line[0]+0.3, average_through_time.index[0], str(round(slope*100, 1)) + ' cm per year', fontsize=20)
# plt.text(regression_line[0]+30, average_through_time.index[0], str(round(slope, 2)) + ' m per year', fontsize=20)
#plt.text(regression_line[0]+0.03, average_through_time.index[0], str(round(slope, 4)) + ' m/m\nper year', fontsize=20)

# Set labels and ticks of x and y axis
ticks_y2 = average_through_time.index[0::5]
ax2.set_xlabel(colorbar_label, fontsize=20)
ax2.tick_params(axis='y', which='both',length=5, labelsize = 20)
ax2.set_yticks(ticks_y2, labels_y)
ax2.tick_params(axis='x', which='both',length=5, labelsize = 20)
ax2.set_xlim([vmin_avg, vmax_avg])

# PLOT SPATIAL AVERAGES OF VARIABLE
ax3 = fig.add_subplot(gs[2])
ax3.scatter(range(0, len(average_through_space)), average_through_space, c=average_through_space, cmap=cmap, vmin=vmin, vmax=vmax)
# Set labels and ticks of x and y axis
ax3.set_xlim([0, len(average_through_space)])
ax3.set_xticks(ticks_x)
ax3.set_xticklabels(labels_x)
ax3.set_ylabel(colorbar_label, fontsize=20)
ax3.tick_params(axis='x', which='both',length=0, labelsize = 24)
ax3.set_ylim([vmin, vmax])
ax3.tick_params(axis='y', which='both',length=5, labelsize = 20)
ax3.axvline(x=686, color='r') # Boundary kustvak 6 en 7, Den Helder, trsct 7000000, decameter 23417
ax3.axvline(x=1288, color='r')# Boundary kustvak 9 en 10, Hoek van Holland, trsct 10000000, decameter 35908

# Plot colorbar
cbar = fig.colorbar(colorplot, ax=[ax1, ax2, ax3])
cbar.set_label(colorbar_label,size=20, labelpad = 20)
cbar.ax.tick_params(labelsize=20) 

plt.tight_layout

manager = plt.get_current_fig_manager()
manager.window.showMaximized()
plt.show()

#plt.savefig(Dir_neat_plots + 'Nourished_' + file_name + "_variation" + "_plot" + ".png")
#plt.savefig(Dir_neat_plots + 'Dynamic_' + file_name + "_variation" + "_plot" + ".png")
plt.savefig(Dir_neat_plots + file_name + '_variation' + '_plot.png')
pickle.dump(fig, open(Dir_neat_plots + file_name + '_variation' + '_plot.fig.pickle', 'wb'))

plt.close()

##################################
####       PLOTTING TREND     ####
##################################

#yticks = [11, 12, 13, 14]
#ylabels = ['11', '12', '13', '14']

# Set-up of figure
fig = plt.figure() 

#fig.suptitle(figure_title, fontsize=26)
gs = gridspec.GridSpec(2, 2, width_ratios=[3, 1], height_ratios=[3,2]) 

# PLOT TEMPORAL AND SPATIAL DISTRIBUTION OF VARIABLE
ax1 = fig.add_subplot(gs[0])
# PLOT YEARLY AVERAGE OF VARIABLE
ax1.plot(average_through_time.index, average_through_time, color=colormap_var[:-1], linewidth=2.5)
# Calculate and plot regression line
mask = ~np.isnan(average_through_time.index) & ~np.isnan(average_through_time)
slope, intercept, r_value, p_value, std_err = scipy.stats.linregress(average_through_time.index[mask], average_through_time[mask])
regression_line = slope*average_through_time.index + intercept
ax1.plot(average_through_time.index, regression_line, 'r--', linewidth=2)
# plot slope of the regression line
#plt.text(regression_line[0], average_through_time.index[0], str(round(slope*1000, 1)) + ' mm per year', fontsize=20)
#plt.text(regression_line[0]+0.3, average_through_time.index[0], str(round(slope*100, 1)) + ' cm per year', fontsize=20)
# plt.text(regression_line[0], average_through_time.index[0], str(round(slope, 2)) + ' m per year', fontsize=20)

# Set labels and ticks of x and y axis
ax1.ticks_x2 = average_through_time.index[0::5]
ax1.set_ylabel(colorbar_label, fontsize=36)
ax1.set_xticks(ticks_y2, labels_y)
ax1.set_ylim([vmin_avg, vmax_avg])
ax1.tick_params(axis='x', which='both',length=5, labelsize = 30)
ax1.tick_params(axis='y', which='both',length=5, labelsize = 30)
#ax1.set_yticks(yticks, ylabels)


##################################
####       PLOTTING TREND     ####
##################################
# WITH TREND BREAK
#yticks = [11, 12, 13, 14]
#ylabels = ['11', '12', '13', '14']

# Set-up of figure
fig = plt.figure() 

#fig.suptitle(figure_title, fontsize=26)
gs = gridspec.GridSpec(2, 2, width_ratios=[3, 1], height_ratios=[3,2]) 

# PLOT TEMPORAL AND SPATIAL DISTRIBUTION OF VARIABLE
ax1 = fig.add_subplot(gs[0])
# PLOT YEARLY AVERAGE OF VARIABLE
ax1.plot(average_through_time.index, average_through_time, color=colormap_var[:-1], linewidth=2.5)
# Calculate and plot regression line
mask = ~np.isnan(average_through_time.index) & ~np.isnan(average_through_time)
yrs_masked = average_through_time.index[mask]
loc_masked = average_through_time[mask]
slope1, intercept1, r_value1, p_value1, std_err1 = scipy.stats.linregress(yrs_masked[0:30], loc_masked[0:30])
regression_line1 = slope1*yrs_masked[0:30] + intercept1
ax1.plot(yrs_masked[0:30], regression_line1, 'r--', linewidth=2)

slope2, intercept2, r_value2, p_value2, std_err2 = scipy.stats.linregress(yrs_masked[29:], loc_masked[29:])
regression_line2= slope2*yrs_masked[29:] + intercept2
ax1.plot(yrs_masked[29:], regression_line2, 'r--', linewidth=2)
# plot slope of the regression line
#plt.text(regression_line[0], average_through_time.index[0], str(round(slope*1000, 1)) + ' mm per year', fontsize=20)
#plt.text(regression_line[0]+0.3, average_through_time.index[0], str(round(slope*100, 1)) + ' cm per year', fontsize=20)
# plt.text(regression_line[0], average_through_time.index[0], str(round(slope, 2)) + ' m per year', fontsize=20)

# Set labels and ticks of x and y axis
ax1.ticks_x2 = average_through_time.index[0::5]
ax1.set_ylabel(colorbar_label, fontsize=36)
ax1.set_xticks(ticks_y2, labels_y)
ax1.set_ylim([vmin_avg, vmax_avg])
ax1.tick_params(axis='x', which='both',length=5, labelsize = 30)
ax1.tick_params(axis='y', which='both',length=5, labelsize = 30)
#ax1.set_yticks(yticks, ylabels)

