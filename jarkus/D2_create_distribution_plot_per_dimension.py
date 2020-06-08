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
import numpy as np
import pandas as pd
import os.path
from jarkus.transects import Transects
import pickle
import matplotlib.pyplot as plt
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
Dir_variable_plots = settings['Dir_D2']

years_requested = list(range(1965, 2020))
labels_y = [str(yr) for yr in years_requested][0::5]

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

##################################
####      CREATE COLORMAP     ####
##################################

# Get all variables from the columns in the dataframe with all the dimensions
trsct = str(ids_filtered[0])
pickle_file = Dir_per_transect + 'Transect_' + trsct + '_dataframe.pickle'
Dimensions_test = pickle.load(open(pickle_file, 'rb')) #load pickle of dimensions    
variables = list(Dimensions_test.columns)

normalized_variables = [var + '_normalized' for var in variables if '_x' in var and 'change' not in var]

variables = variables + normalized_variables

variables.remove('transect')
variables.remove('topo_date') 
variables.remove('bathy_date')

variables = ['DFront_der_prim_grad'] 

#%%
transects_to_visualize = [str(x) for x in ids_filtered if x in ids]
labels_x = [str(tr) for tr in transects_to_visualize][0::25]

for i, var in enumerate(variables):
    pickle_file = Dir_per_variable + variables[i] + '_dataframe.pickle'    
    if os.path.exists(pickle_file):
        Variable_values = pickle.load(open(pickle_file, 'rb')) #load pickle of dimension    
        plt.figure(figsize=(30,15))
        average = np.nanmean(Variable_values[transects_to_visualize].values)
        stddev = np.nanstd(Variable_values[transects_to_visualize].values, ddof=1)
        range_value = 2*stddev
        fig = plt.pcolor(Variable_values[transects_to_visualize], vmin = average-range_value, vmax = average + range_value)
        plt.title(settings[var])
        ticks_y = range(0, len(years_requested))[0::5]
        ticks_x = range(0, len(transects_to_visualize))[0::25]
        plt.yticks(ticks_y, labels_y)
        plt.xticks(ticks_x, labels_x, rotation='vertical')
        plt.colorbar()
        plt.savefig(Dir_variable_plots + var + '_plot.png')
        pickle.dump(fig, open(Dir_variable_plots + var + '_plot.fig.pickle', 'wb'))
        
        #plt.show()
        #plt.close()
    
    
    #set(ax, 'YTickLabel', years_requested)
    else:
        print(pickle_file + ' was not available.')
        continue
    
#%%
# Only plot Holland coast
transects_to_visualize_HC = [str(x) for x in ids_filtered if x in ids and x < 10000000 and x >= 7000000]
labels_x = [str(tr) for tr in transects_to_visualize_HC][0::25]

for i, var in enumerate(variables):
    pickle_file = Dir_per_variable + variables[i] + '_dataframe.pickle'    
    if os.path.exists(pickle_file):
        Variable_values = pickle.load(open(pickle_file, 'rb')) #load pickle of dimension    
        plt.figure(figsize=(30,15))
        average = np.nanmean(Variable_values[transects_to_visualize_HC].values)
        stddev = np.nanstd(Variable_values[transects_to_visualize_HC].values, ddof=1)
        range_value = 2*stddev
        fig = plt.pcolor(Variable_values[transects_to_visualize_HC], vmin = average-range_value, vmax = average + range_value)
        plt.title(settings[var])
        ticks_y = range(0, len(years_requested))[0::5]
        ticks_x = range(0, len(transects_to_visualize_HC))[0::25]
        plt.yticks(ticks_y, labels_y)
        plt.xticks(ticks_x, labels_x, rotation='vertical')
        plt.colorbar()
        plt.savefig(Dir_variable_plots + var + '_HC_plot.png')
        #pickle.dump(fig, open(Dir_per_variable + var + '_plot.fig.pickle', 'wb'))
        
        #plt.show()
        plt.close()
    
    
    #set(ax, 'YTickLabel', years_requested)
    else:
        print(pickle_file + ' was not available.')
        continue
    
#%%
# Only plot Delta coast
transects_to_visualize_DC = [str(x) for x in ids_filtered if x in ids and x > 10000000]
labels_x = [str(tr) for tr in transects_to_visualize_DC][0::25]

for i, var in enumerate(variables):
    pickle_file = Dir_per_variable + variables[i] + '_dataframe.pickle'    
    if os.path.exists(pickle_file):
        Variable_values = pickle.load(open(pickle_file, 'rb')) #load pickle of dimension    
        plt.figure(figsize=(30,15))
        average = np.nanmean(Variable_values[transects_to_visualize_DC].values)
        stddev = np.nanstd(Variable_values[transects_to_visualize_DC].values, ddof=1)
        range_value = 2*stddev
        fig = plt.pcolor(Variable_values[transects_to_visualize_DC], vmin = average-range_value, vmax = average + range_value)
        plt.title(settings[var])
        ticks_y = range(0, len(years_requested))[0::5]
        ticks_x = range(0, len(transects_to_visualize_DC))[0::25]
        plt.yticks(ticks_y, labels_y)
        plt.xticks(ticks_x, labels_x, rotation='vertical')
        plt.colorbar()
        plt.savefig(Dir_variable_plots + var + '_DC_plot.png')
        #pickle.dump(fig, open(Dir_per_variable + var + '_plot.fig.pickle', 'wb'))
        
        #plt.show()
        plt.close()
    
    
    #set(ax, 'YTickLabel', years_requested)
    else:
        print(pickle_file + ' was not available.')
        continue
    
#%% 
# Only plot Wadden coast
transects_to_visualize_WC = [str(x) for x in ids_filtered if x in ids and x >= 2000101 and x < 7000000]
labels_x = [str(tr) for tr in transects_to_visualize_WC][0::25]

for i, var in enumerate(variables):
    pickle_file = Dir_per_variable + variables[i] + '_dataframe.pickle'    
    if os.path.exists(pickle_file):
        Variable_values = pickle.load(open(pickle_file, 'rb')) #load pickle of dimension    
        plt.figure(figsize=(30,15))
        average = np.nanmean(Variable_values[transects_to_visualize_WC].values)
        stddev = np.nanstd(Variable_values[transects_to_visualize_WC].values, ddof=1)
        range_value = 2*stddev
        fig = plt.pcolor(Variable_values[transects_to_visualize_WC], vmin = average-range_value, vmax = average + range_value)
        plt.title(settings[var])
        ticks_y = range(0, len(years_requested))[0::5]
        ticks_x = range(0, len(transects_to_visualize_WC))[0::25]
        plt.yticks(ticks_y, labels_y)
        plt.xticks(ticks_x, labels_x, rotation='vertical')
        plt.colorbar()
        plt.savefig(Dir_variable_plots + var + '_WC_plot.png')
        #pickle.dump(fig, open(Dir_per_variable + var + '_plot.fig.pickle', 'wb'))
        
        #plt.show()
        plt.close()
    
    
    #set(ax, 'YTickLabel', years_requested)
    else:
        print(pickle_file + ' was not available.')
        continue