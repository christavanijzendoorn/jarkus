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
Dir_variable_plots = settings['Dir_D4']

years_requested = list(range(1965, 2020))
labels_y = [str(yr) for yr in years_requested][0::5]

##################################
####      INITIALISATION      ####
##################################
# Collect the JARKUS data from the server
Jk = Transects(url= settings['url'])
ids = Jk.get_data('id') # ids of all available transects

remove1 = list(range(2000303,2000304))
remove2 = list(range(2000720,2000721))
remove3 = list(range(2002019,2002020))
remove4 = list(range(4002001,4005917,1))
remove5 = list(range(5003300,5004000,1))
remove6 = list(range(6000416,6000900,1))
remove7 = list(range(6003070,6003453,1))
remove8 = list(range(8000000,8005626,1))
remove9 = list(range(9010193,9010194,1))
remove10 = list(range(10000000,10001411,1))
remove11 = list(range(11000660,11000661,1))
remove12 = list(range(11000680,11000681,1))
remove13 = list(range(11000700,11000841,1))
remove14 = list(range(14000000,14000700,1))

remove_all = remove1 + remove2 + remove3 + remove4 + remove5 + remove6 + remove7 + remove8 + remove9 + remove10 + remove11 + remove12 + remove13 + remove14

ids_filtered = [x for x in ids if x not in remove_all]

##################################
####      CREATE COLORMAP     ####
##################################

variables = ['max_TWL_bathy_gen', 'max_TWL_bathy_dis', 'max_TWL_topo_gen', 'max_TWL_topo_dis']

pickle_file0 = Dir_per_variable + variables[0] + '_dataframe.pickle'    
pickle_file1 = Dir_per_variable + variables[1] + '_dataframe.pickle'    
pickle_file2 = Dir_per_variable + variables[2] + '_dataframe.pickle'    
pickle_file3 = Dir_per_variable + variables[3] + '_dataframe.pickle'    

Variable_values0 = pickle.load(open(pickle_file0, 'rb')) #load pickle of dimension    
Variable_values1 = pickle.load(open(pickle_file1, 'rb')) #load pickle of dimension    
Variable_values2 = pickle.load(open(pickle_file2, 'rb')) #load pickle of dimension    
Variable_values3 = pickle.load(open(pickle_file3, 'rb')) #load pickle of dimension    

diff_topo_vs_bathy_gen = Variable_values0 - Variable_values2
diff_topo_vs_bathy_dis = Variable_values1 - Variable_values3
diff_gen_vs_dis_bathy = Variable_values0 - Variable_values1
diff_gen_vs_dis_topo = Variable_values2 - Variable_values3


#%%
transects_to_visualize = [str(x) for x in ids_filtered if x in ids]
labels_x = [str(tr) for tr in transects_to_visualize][0::25]

plt.figure(figsize=(30,15))
average = np.nanmean(diff_topo_vs_bathy_dis.values)
stddev = np.nanstd(diff_topo_vs_bathy_dis.values, ddof=1)
range_value = 2*stddev
fig = plt.pcolor(diff_topo_vs_bathy_dis, vmin = average-range_value, vmax = average + range_value)
plt.title('Difference map topographic dates vs bathymetric dates (dissipative equation)')
ticks_y = range(0, len(years_requested))[0::5]
ticks_x = range(0, len(transects_to_visualize))[0::25]
plt.yticks(ticks_y, labels_y)
plt.xticks(ticks_x, labels_x, rotation='vertical')
plt.colorbar()
plt.savefig(Dir_variable_plots + 'diff_topo_vs_bathy_dis' + '_plot.png')
pickle.dump(fig, open(Dir_variable_plots + 'diff_topo_vs_bathy_dis' + '_plot.fig.pickle', 'wb'))

#plt.show()
plt.close()

#%%
transects_to_visualize = [str(x) for x in ids_filtered if x in ids]
labels_x = [str(tr) for tr in transects_to_visualize][0::25]

plt.figure(figsize=(30,15))
average = np.nanmean(diff_topo_vs_bathy_gen.values)
stddev = np.nanstd(diff_topo_vs_bathy_gen.values, ddof=1)
range_value = 2*stddev
fig = plt.pcolor(diff_topo_vs_bathy_gen, vmin = average-range_value, vmax = average + range_value)
plt.title('Difference map topographic dates vs bathymetric dates (general equation)')
ticks_y = range(0, len(years_requested))[0::5]
ticks_x = range(0, len(transects_to_visualize))[0::25]
plt.yticks(ticks_y, labels_y)
plt.xticks(ticks_x, labels_x, rotation='vertical')
plt.colorbar()
plt.savefig(Dir_variable_plots + 'diff_topo_vs_bathy_gen' + '_plot.png')
pickle.dump(fig, open(Dir_variable_plots + 'diff_topo_vs_bathy_gen' + '_plot.fig.pickle', 'wb'))

#plt.show()
plt.close()

#%%
transects_to_visualize = [str(x) for x in ids_filtered if x in ids]
labels_x = [str(tr) for tr in transects_to_visualize][0::25]

plt.figure(figsize=(30,15))
average = np.nanmean(diff_gen_vs_dis_bathy.values)
stddev = np.nanstd(diff_gen_vs_dis_bathy.values, ddof=1)
range_value = 2*stddev
fig = plt.pcolor(diff_gen_vs_dis_bathy, vmin = average-range_value, vmax = average + range_value)
plt.title('Difference map dissipative vs general equation (bathymetric dates)')
ticks_y = range(0, len(years_requested))[0::5]
ticks_x = range(0, len(transects_to_visualize))[0::25]
plt.yticks(ticks_y, labels_y)
plt.xticks(ticks_x, labels_x, rotation='vertical')
plt.colorbar()
plt.savefig(Dir_variable_plots + 'diff_gen_vs_dis_bathy' + '_plot.png')
pickle.dump(fig, open(Dir_variable_plots + 'diff_gen_vs_dis_bathy' + '_plot.fig.pickle', 'wb'))

#plt.show()
plt.close()

#%%
transects_to_visualize = [str(x) for x in ids_filtered if x in ids]
labels_x = [str(tr) for tr in transects_to_visualize][0::25]

plt.figure(figsize=(30,15))
average = np.nanmean(diff_gen_vs_dis_topo.values)
stddev = np.nanstd(diff_gen_vs_dis_topo.values, ddof=1)
range_value = 2*stddev
fig = plt.pcolor(diff_gen_vs_dis_topo, vmin = average-range_value, vmax = average + range_value)
plt.title('Difference map dissipative vs general equation (topographic dates)')
ticks_y = range(0, len(years_requested))[0::5]
ticks_x = range(0, len(transects_to_visualize))[0::25]
plt.yticks(ticks_y, labels_y)
plt.xticks(ticks_x, labels_x, rotation='vertical')
plt.colorbar()
plt.savefig(Dir_variable_plots + 'diff_gen_vs_dis_topo' + '_plot.png')
pickle.dump(fig, open(Dir_variable_plots + 'diff_gen_vs_dis_topo' + '_plot.fig.pickle', 'wb'))

#plt.show()
plt.close()