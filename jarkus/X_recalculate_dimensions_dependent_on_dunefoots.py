# -*- coding: utf-8 -*-
"""
Created on Thu Nov 21 17:15:36 2019

@author: cijzendoornvan
"""

# -*- coding: utf-8 -*-
"""
Created on Mon Jul 29 11:20:42 2019

@author: cijzendoornvan
"""
##################################
####          PACKAGES        ####
##################################
import json
import numpy as np
import pandas as pd
from scipy.interpolate import griddata
from scipy.signal import find_peaks
import time
import os
import pickle

from analysis_functions import get_volume, get_gradient, find_intersections
from jarkus.transects import Transects
from IPython import get_ipython
get_ipython().run_line_magic('matplotlib', 'auto')

#%%
##################################
####       RETRIEVE DATA      ####
##################################
# load basic settings from .txt file
with open("C:/Users/cijzendoornvan/Documents/GitHub/jarkus/jarkus/Settings.txt") as file:
    settings = json.load(file)

# Collect the JARKUS data from the server
Jk = Transects(url= settings['url'])
ids = Jk.get_data('id') # ids of all available transects

#ids = ids[ids>8008800]

#%%
##################################
####    USER-DEFINED REQUEST  ####
##################################
# Select which dimensions should be calculated from the transects
dune_height_and_location = True
mean_sea_level      = True
mean_low_water      = True
mean_high_water     = True
intertidal_width    = True
landward_point      = True
seaward_point       = True
dune_foot_location  = True
beach_width         = True
beach_gradient      = True
dune_front_width    = True
dune_gradient       = True
dune_volume         = True
intertidal_gradient = True
intertidal_volume   = True
foreshore_gradient  = True
foreshore_volume    = True
active_profile_gradient  = True
active_profile_volume    = True

# Set which years should be analysed
years_requested = list(range(1965, 2020))
#years_requested = [1978, 1979]

# Set whether all transect should be analysed or define a retrieval request
execute_all_transects = False

if execute_all_transects == False:
    # Set the transect and years for retrieval request
    transect_name   = "02_Schiermonnikoog"
    #transect_req = np.arange(17000011, 17001467, 1)
    transect_req        = [2000105]
    idxs = np.isin(transect_req, ids) # check which transect are available of those that were requested
    ids_filtered = np.array(transect_req)[np.nonzero(idxs)[0]]
    Dir_pickles = settings['Dir_B']  
else:
    # Filter out location that are not suitable for analysis. Based on Kustlijnkaarten 2019, RWS.
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
    
    Dir_dataframes = settings['Dir_X']
    Dir_pickles = settings['Dir_X']

years_requested_str = [str(yr) for yr in years_requested]

###################################
####      LOAD DATAFRAMES      ####
###################################
        
for idx in ids_filtered: # For each available transect go though the entire analysis    
    trsct = str(idx)
    pickle_file = Dir_dataframes + 'Transect_' + trsct + '_dataframe.pickle'
    
    # Here the JARKUS filter is set and the data for the requested transect and years is retrieved
    Jk = Transects(url='https://opendap.tudelft.nl/thredds/dodsC/data2/deltares/rijkswaterstaat/jarkus/profiles/transect_r20180914.nc')
    df, years_available = Jk.get_dataframe(idx, years_requested)
    
    # Interpolate x and y along standardized cross shore axis
    cross_shore = list(range(-3000, 9320, 1))
    
    # Convert elevation data for each year of each transect into aray that can be easily analysed
    y_all = [] 
    for i, yr in enumerate(years_requested_str):
        if yr in years_available:
            y = np.array(df.loc[trsct, yr]['y'])
            x = np.array(df.loc[trsct, yr]['x'])
            y_grid = griddata(x,y,cross_shore)
            if i == 0:
                y_all = y_grid
            else:
                y_all = np.column_stack((y_all, y_grid))
                y_grid = griddata(x,y,cross_shore)
        else:
            y_grid = np.empty((len(cross_shore),))
            y_grid[:] = np.nan
            #print(y_grid)
            if i == 0:
                y_all = y_grid
            else:
                y_all = np.column_stack((y_all, y_grid))


#%%
###################################  
#### DEFINE DUNEFOOT LOCATION  ####
###################################
                
    if os.path.exists(pickle_file) and dune_foot_location == True:
        Dimensions = pickle.load(open(pickle_file, 'rb')) #load pickle of transect
            
        Dimensions['BW_der'] = Dimensions['MSL_x'] - Dimensions['DF_der_x'] 
        Dimensions['DFront_der_prim_W'] = Dimensions['DF_der_x'] - Dimensions['DT_prim_x']
        Dimensions['DFront_der_sec_W'] = Dimensions['DF_der_x'] - Dimensions['DT_sec_x']
        Dimensions['DVol_der'] = get_volume(cross_shore, y_all, years_requested, Dimensions['DF_der_x'], Dimensions['landward_stable_x'])
  
    else:
        pass
    

#%%    
###################################
###        SAVE DATAFRAME       ### 
###################################
    # Save dataframe for each transect.
    # Later these can all be loaded to calculate averages for specific sites/sections along the coast
    Dimensions.to_pickle(Dir_pickles + 'Transect_' + trsct + '_dataframe.pickle')
    

    
