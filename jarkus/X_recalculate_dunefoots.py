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
    
    Dir_dataframes = settings['Dir_B']
    Dir_pickles = settings['Dir_X']

years_requested_str = [str(yr) for yr in years_requested]

###################################
####      LOAD DATAFRAMES      ####
###################################

        
#HERE LOAD ALREADY EXISTING DATAFRAME AND USE TO RECALCULATE THE DUNEFOOT
        
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
        
        #### Fixed dunefoot definition ####
        ###################################
        DF_fixed_y = 3 # in m above reference datum
        Dimensions['DF_fix_y'] = np.nan
        Dimensions['DF_fix_x'] = np.nan
        
        for i, yr in enumerate(years_requested_str):
                if yr in years_available:
                    intersect = find_intersections(cross_shore, y_all[:,i], DF_fixed_y)
                    if len(intersect[0]) != 0:
                        idx = intersect[0][-1]
                        Dimensions.loc[yr, 'DF_fix_x'] = cross_shore[idx]
                        Dimensions.loc[yr, 'DF_fix_y'] = DF_fixed_y
            
        ####  Derivative method - E.D. ####
        ###################################
        ## Variable dunefoot definition based on first and second derivative of profile
        Dimensions['DF_der_y'] = np.nan
        Dimensions['DF_der_x'] = np.nan
        for i, yr in enumerate(years_requested_str):
            if yr in years_available:
                # Get seaward boundary
                seaward_x = Dimensions.loc[yr, 'MHW_x_var']
                # Get landward boundary 
                landward_x = Dimensions.loc[yr, 'landward_6m_x']   
        
                # Give nan values to everything outside of boundaries
                y_all_domain = []
                x_all_domain = []
                for xc in range(len(cross_shore)): 
                    if cross_shore[xc] < seaward_x and cross_shore[xc] > landward_x:
                        y_all_domain.append(y_all[xc,i])
                        x_all_domain.append(cross_shore[xc])
            
                # Give nan to years where not enough measurements are available within domain
                if np.count_nonzero(~np.isnan(y_all_domain)) > 5:
                    
                    from scipy import interpolate
                    # smooth the profile so dune foot search is not disturbed by the jagged lines.
                    f = interpolate.splrep(x_all_domain, y_all_domain, s=0.01)
                    cross_shore_new = list(range(min(x_all_domain), max(x_all_domain)+1, 1))
                    y_new = interpolate.splev(cross_shore_new, f, der=0)
                    
                    # Get first and second derivative
                    #y_der1 = np.gradient(y_new, cross_shore_new)    
                    y_der1 = interpolate.splev(cross_shore_new, f, der=1)
                    
                    # Set first derivative values of -0.001 and higher to zero
                    for n in range(len(y_der1)):
                        if y_der1[n] >= -0.001:
                            y_der1[n] = 0
                    
                    y_der2 = np.gradient(y_der1, cross_shore_new)    
                    #y_der2 = interpolate.splev(cross_shore_new, f, der=2)
                
                    # Set second derivative values below 0.01 to zero
                    for n in range(len(y_der2)):
                        #if y_der2[n] <= 0.01:
                        if y_der2[n] <= 0.01:
                            y_der2[n] = 0
                            
                    # Set locations to True where second derivative is above the threshold value
                    dunefoot = np.zeros(len(y_der1))
                    for l in range(len(y_der1)):
                        if y_der2[l] != 0:
                            dunefoot[l] = True
                        
                    # Get most seaward point where the above condition is True
                    if sum(dunefoot) != 0:
                        dunefoot_idx = np.where(dunefoot == True)[0][-1]
                        Dimensions.loc[yr, 'DF_der_x'] = x_all_domain[dunefoot_idx]
                        Dimensions.loc[yr, 'DF_der_y'] = y_all_domain[dunefoot_idx]
                        
                    """
                    # Plot graphs to check the derivatives and placing of the dunefoot
                    import matplotlib.pyplot as plt
                    # Plot the transects and smoothed transects
                    plt.plot(cross_shore_new, y_new, 'r-', cross_shore, y_all[:, i], '--',)
                    # Plot the first and second derivative and the location of the dunefoot
                    plt.plot(cross_shore_new, y_der1, '-', cross_shore_new, y_der2, '--', Dimensions['DF_der_x'], Dimensions['DF_der_y'], 'o')
                    plt.legend(['spline', 'jarkus', 'der1', 'der2', 'DF_der'], loc='best')
                    plt.show()
                    """
    else:
        pass
    

#%%    
###################################
###        SAVE DATAFRAME       ### 
###################################
    # Save dataframe for each transect.
    # Later these can all be loaded to calculate averages for specific sites/sections along the coast
    Dimensions.to_pickle(Dir_pickles + 'Transect_' + trsct + '_dataframe.pickle')
    
    
    
