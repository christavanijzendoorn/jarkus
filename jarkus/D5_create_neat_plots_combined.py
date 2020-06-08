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
from jarkus.transects import Transects
import pickle
import numpy as np
import matplotlib.pyplot as plt 
from matplotlib import gridspec

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
        
conversion_ids2alongshore = dict(zip(ids, ids_alongshore))

#%%
##################################
####       PREPARATIONS       ####
##################################
#
#variables = ['max_TWL_topo_dis']
#figure_title = 'Alongshore and temporal variation of the maximum Total Water Level (m)'
#colorbar_label = 'Total water level (m)'
#colormap_var = "Blues"
#file_name = 'TWL'

#variables = ['MHW_y_var']
#figure_title = 'Alongshore and temporal variation of MHW (as included in Jarkus dataset) (m)'
#colorbar_label = 'MHW elevation (m)'
#colormap_var = "Blues"
#file_name = 'mean_high_water_var'

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

#variables = ['DF_der_y']
#figure_title = 'Alongshore and temporal variation of dune foot elevation (m)'
#colorbar_label = 'Dune foot elevation (m)'
#colormap_var = "Greens"
#file_name = 'dune_foot_elevation'

#variables = ['DF_der_x']
#figure_title = 'Alongshore and temporal variation of the cross-shore dune foot location resp. to MSL (m)'
#colorbar_label = 'Cross-shore dune foot location (m)'
#colormap_var = "Greens"
#file_name = 'dune_foot_location'

#variables = ['DFront_der_prim_grad']
#figure_title = 'Alongshore and temporal variation of the dune front gradient (-)'
#colorbar_label = 'Dune front gradient (-)'
#colormap_var = "Reds"
#file_name = 'dune_front_grad'

variables = ['BW_der_var']
figure_title = 'Alongshore and temporal variation of the beach width (m)'
colorbar_label = 'Beach width (m)'
colormap_var = "Greens"
file_name = 'beach_width'

# load dataframe with distribution of values of variable
pickle_file0 = Dir_per_variable + variables[0] + '_dataframe.pickle'    
Variable_values0 = pickle.load(open(pickle_file0, 'rb')) #load pickle of dimension    

# Get transect that have to be visualized
transects_to_visualize = [x for x in ids_filtered if x in ids]

# FILTER HERE TO REDUCE AREA VISUALIZED

# Convert transect codes (i.e. column names) into alongshore decameter values
alongshore_values = np.zeros(len(transects_to_visualize))

for key, val in conversion_ids2alongshore.items():
    alongshore_values[transects_to_visualize == key] = val
    
#index_alongshore = np.where(np.isin(ids, transects_to_visualize))
#alongshore_values = ids_alongshore[index_alongshore]

# Rearrange dataframe based on alongshore decameters
#Variable_values0.columns = alongshore_values
#Variable_values0.sort_index(axis=1, inplace=True)

# Calculate spatial and temporal average
average_through_space = Variable_values0.mean(axis=0)
average_through_time = Variable_values0.mean(axis=1)

# Calculate overall average and stddev, used for range of colorbar
average         = np.nanmean(Variable_values0.values)
stddev          = np.nanstd(Variable_values0.values, ddof=1)
range_value     = 2*stddev
range_value_avg = stddev
vmin            = average - range_value
vmax            = average + range_value
vmin_avg        = average - range_value_avg
vmax_avg        = average + range_value_avg


# Create an array with locations and an array with labels of the ticks
ticks_x = [350, 1100, 1700]
labels_x = ['Wadden Coast', 'Holland Coast', 'Delta Coast']

ticks_y = range(0, len(years_requested))[0::5]
labels_y = [str(yr) for yr in years_requested][0::5]

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
colorplot = plt.pcolor(Variable_values0, vmin=vmin, vmax=vmax, cmap=cmap)
# Set labels and ticks of x and y axis
plt.yticks(ticks_y, labels_y)
plt.tick_params(axis='y', which='both',length=5, labelsize = 16)
plt.xticks(ticks_x, labels_x) #rotation='vertical')
plt.tick_params(axis='x', which='both',length=0, labelsize = 20)
# plot boundaries between coastal regions
plt.axvline(x=686, color='r') # Boundary kustvak 6 en 7, Den Helder, trsct 7000000, decameter 23417
plt.axvline(x=1507, color='r')# Boundary kustvak 9 en 10, Hoek van Holland, trsct 10000000, decameter 35908

# PLOT YEARLY AVERAGE OF VARIABLE
ax2 = fig.add_subplot(gs[1])
plt.plot(average_through_time, average_through_time.index, color=colormap_var[:-1])
#plt.scatter(average_through_time, average_through_time.index, c=average_through_time, cmap=cmap, vmin=vmin, vmax=vmax)
# Set labels and ticks of x and y axis
ticks_y = average_through_time.index[0::5]
plt.xlabel(colorbar_label)
plt.yticks(ticks_y, labels_y)
plt.tick_params(axis='y', which='both',length=5, labelsize = 16)
plt.xlim([vmin_avg, vmax_avg])
plt.tick_params(axis='x', which='both',length=5, labelsize = 16)

# PLOT SPATIAL AVERAGES OF VARIABLE
ax3 = fig.add_subplot(gs[2])
plt.scatter(range(0, len(average_through_space)), average_through_space, c=average_through_space, cmap=cmap, vmin=vmin, vmax=vmax)
# Set labels and ticks of x and y axis
plt.xlim([0, len(average_through_space)])
plt.xticks(ticks_x, labels_x) 
plt.ylabel(colorbar_label)
plt.tick_params(axis='x', which='both',length=0, labelsize = 20)
plt.ylim([vmin, vmax])
plt.tick_params(axis='y', which='both',length=5, labelsize = 16)
plt.axvline(x=686, color='r') # Boundary kustvak 6 en 7, Den Helder, trsct 7000000, decameter 23417
plt.axvline(x=1507, color='r')# Boundary kustvak 9 en 10, Hoek van Holland, trsct 10000000, decameter 35908

# Plot colorbar
cbar = fig.colorbar(colorplot, ax=[ax1, ax2, ax3])
cbar.set_label(colorbar_label,size=18, labelpad = 20)
cbar.ax.tick_params(labelsize=16) 

plt.tight_layout

manager = plt.get_current_fig_manager()
manager.window.showMaximized()
plt.show()

plt.savefig(Dir_neat_plots + file_name + '_variation' + '_plot.png')
pickle.dump(fig, open(Dir_neat_plots + file_name + '_variation' + '_plot.fig.pickle', 'wb'))

#plt.close()



"""

# Plot distribution of variable
plt.figure(figsize=(30,15))

cmap = plt.cm.get_cmap("Blues")
fig = plt.pcolor(Variable_values0, vmin = average-range_value, vmax = average + range_value, cmap=cmap)

plt.axvline(x=686, color='r') # Boundary kustvak 6 en 7, Den Helder, trsct 7000000, decameter 23417
plt.axvline(x=1507, color='r')# Boundary kustvak 9 en 10, Hoek van Holland, trsct 10000000, decameter 35908

plt.title('Alongshore and temporal variation of the maximum Total Water Level (m)', fontsize=26)
ticks_y = range(0, len(years_requested))[0::5]
ticks_x = [350, 1100, 1700]

plt.yticks(ticks_y, labels_y)
plt.tick_params(axis='y', which='both',length=5, labelsize = 16)
plt.xticks(ticks_x, labels_x) #rotation='vertical')
plt.tick_params(axis='x', which='both',length=0, labelsize = 20)

cbar = plt.colorbar()
cbar.set_label('Total Water Level (m)',size=18, labelpad = 20)
cbar.ax.tick_params(labelsize=16) 

#plt.savefig(Dir_neat_plots + 'diff_topo_vs_bathy_dis' + '_plot.png')
#pickle.dump(fig, open(Dir_neat_plots + 'diff_topo_vs_bathy_dis' + '_plot.fig.pickle', 'wb'))

#plt.show()
#plt.close()

"""