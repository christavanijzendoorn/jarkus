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
Created on Thu Nov 28 15:04:10 2019

@author: cijzendoornvan
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
from jarkus.transects import Transects
import pickle
import numpy as np
import matplotlib.pyplot as plt 
from matplotlib import gridspec
import scipy.stats as stats

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

##################################
####       PREPARATIONS       ####
##################################

variables_TWL = ['max_TWL_topo_dis']
figure_title_TWL = 'Alongshore and temporal variation of the maximum Total Water Level (m)'
colorbar_label_TWL = 'Total water level (m)'
colormap_var_TWL = "Blues"
file_name_TWL = 'TWL'

variables_DF = ['DF_der_y']
figure_title_DF = 'Alongshore and temporal variation of dune foot elevation (m)'
colorbar_label_DF = 'Dune foot elevation (m)'
colormap_var_DF = "Greens"
file_name_DF = 'dune_foot'


# load dataframe with distribution of TWL
pickle_file_TWL = Dir_per_variable + variables_TWL[0] + '_dataframe.pickle'    
Variable_values_TWL = pickle.load(open(pickle_file_TWL, 'rb')) #load pickle of dimension    

# load dataframe with distribution of DF
pickle_file_DF = Dir_per_variable + variables_DF[0] + '_dataframe.pickle'    
Variable_values_DF = pickle.load(open(pickle_file_DF, 'rb')) #load pickle of dimension 

# Get transect that have to be visualized
transects_to_visualize = [x for x in ids_filtered if x in ids]

# Convert transect codes (i.e. column names) into alongshore decameter values
index_alongshore = np.where(np.isin(ids, transects_to_visualize))
alongshore_values = ids_alongshore[index_alongshore]

# Rearrange dataframe TWL based on alongshore decameters
Variable_values_TWL.columns = alongshore_values
Variable_values_TWL.sort_index(axis=1, inplace=True)
# Rearrange dataframe DF based on alongshore decameters
Variable_values_DF.columns = alongshore_values
Variable_values_DF.sort_index(axis=1, inplace=True)

# Calculate spatial and temporal average
average_through_time_TWL = Variable_values_TWL.mean(axis=1)

average_through_time_DF = Variable_values_DF.mean(axis=1)

##################################
####       PLOTTING       ####
##################################

#%%
# Set-up of figure
fig1 = plt.figure() 

figure_title = "spatially averaged TWL vs spatially averaged DF"
plt.title(figure_title, fontsize=26)

manager = plt.get_current_fig_manager()
manager.window.showMaximized()

# PLOT SPATIAL AVERAGES OF VARIABLE
plt.scatter(average_through_time_TWL, average_through_time_DF)
# Set labels and ticks of x and y axis
plt.xlim([2, 4])
plt.ylim([2, 4])
plt.xlabel('spatially averaged maximum TWL')
plt.ylabel('spatially averaged dune foot location')
plt.tick_params(axis='x', which='both',length=0, labelsize = 20)
plt.tick_params(axis='y', which='both',length=5, labelsize = 16)

#plt.show()

#plt.savefig(Dir_neat_plots + file_name + '_variation' + '_plot.png')
#pickle.dump(fig, open(Dir_neat_plots + file_name + '_variation' + '_plot.fig.pickle', 'wb'))

#plt.close()

#%%
# Set-up of figure
fig1 = plt.figure() 

figure_title = "all TWL values vs all DF values"
plt.title(figure_title, fontsize=26)

manager = plt.get_current_fig_manager()
manager.window.showMaximized()

# PLOT SPATIAL AVERAGES OF VARIABLE
plt.scatter(Variable_values_TWL, Variable_values_DF)
# Set labels and ticks of x and y axis
plt.xlim([2, 4])
plt.ylim([2, 4])
plt.xlabel('spatially averaged maximum TWL')
plt.ylabel('spatially averaged dune foot location')
plt.tick_params(axis='x', which='both',length=0, labelsize = 20)
plt.tick_params(axis='y', which='both',length=5, labelsize = 16)

#plt.show()

#plt.savefig(Dir_neat_plots + file_name + '_variation' + '_plot.png')
#pickle.dump(fig, open(Dir_neat_plots + file_name + '_variation' + '_plot.fig.pickle', 'wb'))

#plt.close()
