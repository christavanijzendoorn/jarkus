# -*- coding: utf-8 -*-
"""
Created on Mon Oct 14 16:58:52 2019

@author: cijzendoornvan
"""
##################################
####          PACKAGES        ####
##################################
import numpy as np
import pandas as pd
import os.path
from jarkus.transects import Transects
import pickle
import matplotlib.pyplot as plt

## %matplotlib auto TO GET WINDOW FIGURE
#################################
####        FUNCTIONS        ####
#################################


##################################
####    USER-DEFINED REQUEST  ####
##################################
        
Dir_per_transect = "C:/Users/cijzendoornvan/Documents/GitHub/jarkus/jarkus/Figures/Dataframes_per_transect/"
Dir_per_variable = "C:/Users/cijzendoornvan/Documents/GitHub/jarkus/jarkus/Figures/Dataframes_per_dimension/"

years_requested = list(range(1965, 2020))
labels_y = [str(yr) for yr in years_requested][0::5]

##################################
####      INITIALISATION      ####
##################################

# Collect the JARKUS data from the server
#Jk = Transects(url='http://opendap.deltares.nl/thredds/catalog/opendap/rijkswaterstaat/jarkus/profiles/catalog.html?dataset=varopendap/rijkswaterstaat/jarkus/profiles/transect_r20190731.nc')
Jk = Transects(url='https://opendap.tudelft.nl/thredds/dodsC/data2/deltares/rijkswaterstaat/jarkus/profiles/transect_r20180914.nc')

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

variables = ['DT_prim_x_normalized', 'DT_prim_y',
 'DT_sec_x_normalized','DT_sec_y',
 'MSL_x_normalized',
 'MLW_x_fix_normalized', 'MLW_x_var_normalized', 'MLW_y_var',
 'MHW_x_fix', 'MHW_x_var', 'MHW_y_var',
 'W_intertidal_fix', 'W_intertidal_var',
 'landward_stable_x_normalized', 'landward_6m_x_normalized', 
 'Bma_x_normalized',
 'seaward_FS_x_all_normalized', 'seaward_ActProf_x_all_normalized',
 'DF_fix_y', 'DF_fix_x_normalized',
 'DF_der_y', 'DF_der_x_normalized',
 'BW_fix', 'BW_var', 'BW_der', 
 'B_grad_fix',
 'DFront_fix_prim_W', 'DFront_der_prim_W',
 'DFront_fix_sec_W', 'DFront_der_sec_W',
 'DFront_fix_prim_grad', 'DFront_fix_sec_grad',
 'DVol_fix', 'DVol_der',
 'Int_grad', 'IntVol_fix', 'IntVol_var',
 'FS_grad', 'FSVol_fix',
 'AP_grad', 'APVol_fix', 'APVol_var']

titles = ['Cross-shore normalized primary dune peak location (m)', 'Primary dune peak elevation (m)',
          'Cross-shore normalized secondary dune peak location (m)', 'Secondary dune peak elevation (m)',
          'Cross-shore normalized mean sea level location (m)',
          'Cross-shore normalized mean low water location (m) (fixed)', 'Cross-shore normalized mean low water location (m) (variable)', 'Mean low water elevation (m) (variable)',
          'Cross-shore normalized mean high water location (m) (fixed)', 'Cross-shore normalized mean high water location (m) (variable)', 'Mean high water elevation (m) (variable)',
          'Width of intertidal zone (m) (fixed)','Width of intertidal zone (m) (variable)',
          'Cross-shore normalized location of the landward stable point (m)', 'Cross-shore normalized location of the landward point based on 6 m elevation (m)',
          'Cross-shore normalized location of the boundary between the marine and aeolian zone (m) based on 2 m elevation',
          'Cross-shore normalized location of the seaward foreshore boundary (m) based on -4 m NAP', 'Cross-shore normalized location of the seaward active profile boundary (m) based on -8 m NAP',
          'Dune foot elevation (m) (fixed)','Cross-shore normalized dune foot location (m) (fixed)', 
          'Dune foot elevation (m) (derivative)','Cross-shore normalized dune foot location (m) (derivative)', 
          'Beach width (m) (fixed)', 'Beach width (m) (variable)', 'Beach width (m) (derivative)',
          'Beach slope (m/m) (fixed)',
          'Primary dune front width (m) (fixed)', 'Primary dune front width (m) (derivative)',
          'Secondary dune front width (m) (fixed)', 'Secondary dune front width (m) (derivative)',
          'Primary dune front slope (m) (fixed)', 'Secondary dune front slope (m) (fixed)',
          'Dune volume (m^3/m) (fixed)', 'Dune volume (m^3/m) (derivative)',
          'Intertidal area slope (m/m)', 'Intertidal area volume (m^3/m) (fixed)', 'Intertidal area volume (m^3/m) (variable)',
          'Foreshore slope (m/m)', 'Foreshore volume (m^3/m) (fixed)',
          'Active profile slope (m/m)', 'Active profile volume (m^3/m) (fixed)', 'Active profile volume (m^3/m) (variable)',]

range_colorbar = [[-1000, 0], [0, 25],
        [-1000, 0], [0, 25],
        [-500, 500],
        [-1000, 1000], [-1000, 1000], [0.5, 2.0],
        [-1000, 1000], [-1000, 1000], [-2.0, -0.5],
        [0, 1000], [0, 1000],
        [-2000, 0], [-2000, 0],
        [-1000, 1000],
        [0, 3000], [0, 3000],
        [0, 5], [-1500, 0],
        [0, 5], [-1500, 0],
        [0, 1000], [0, 1000], [0, 1000],
        [-0.05, 0],
        [0, 250], [0, 250],
        [0, 250], [0, 250],
        [-0.05, 0], [-0.05, 0],
        [0, 2500], [0, 2500],
        [-0.03, 0], [0, 2500], [0, 2500],
        [-0.03, 0], [0, 5000],
        [-0.02, 0], [-0.02, 0], [0, 10000]]
              
transects_to_visualize = [str(x) for x in ids_filtered if x in ids]
labels_x = [str(tr) for tr in transects_to_visualize][0::25]

for i, var in enumerate(variables):
    pickle_file = Dir_per_variable + variables[i] + '_dataframe.pickle'    
    if os.path.exists(pickle_file):
        Variable_values = pickle.load(open(pickle_file, 'rb')) #load pickle of dimension    
        plt.figure(figsize=(30,15))
        fig = plt.pcolor(Variable_values[transects_to_visualize], vmin = range_colorbar[i][0], vmax = range_colorbar[i][1])
        plt.title(titles[i])
        ticks_y = range(0, len(years_requested))[0::5]
        ticks_x = range(0, len(transects_to_visualize))[0::25]
        plt.yticks(ticks_y, labels_y)
        plt.xticks(ticks_x, labels_x, rotation='vertical')
        plt.colorbar()
        plt.savefig(Dir_per_variable + var + '_plot.png')
        #pickle.dump(fig, open(Dir_per_variable + var + '_plot.fig.pickle', 'wb'))
        
        #plt.show()
        plt.close()
    
    
    #set(ax, 'YTickLabel', years_requested)
else:
    pass