# -*- coding: utf-8 -*-
"""
Created on Mon Oct 21 09:49:55 2019

@author: cijzendoornvan
"""
import numpy as np
import os.path
from jarkus.transects import Transects
import pickle
from scipy.interpolate import griddata


from analysis_functions import get_volume, get_gradient

##################################
####      INITIALISATION      ####
##################################

Dir_per_transect = "C:/Users/cijzendoornvan/Documents/GitHub/jarkus/jarkus/Figures/Dataframes_per_transect_try2/"

years_requested = list(range(1965, 2020))
years_requested_str = [str(yr) for yr in years_requested]

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
        
for idx in ids_filtered: # For each available transect go though the entire analysis    
    trsct = str(idx)
    pickle_file = Dir_per_transect + 'Transect_' + trsct + '_dataframe.pickle'
    
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

    if os.path.exists(pickle_file):
        Dimensions = pickle.load(open(pickle_file, 'rb')) #load pickle of transect
        
        # Swap DF_x and DF_y
        Dimensions.rename(columns={'DF_fix_x':'DF_fix_y', 'DF_fix_y':'DF_fix_x'}, inplace=True)
        Dimensions.rename(columns={'DF_der_x':'DF_der_y', 'DF_der_y':'DF_der_x'}, inplace=True)
        
        # Calculate BW
        B_seaward_var = (Dimensions['MLW_x_var']+Dimensions['MHW_x_var'])/2 # Base beach width on the varying location of the low and high water line
        Dimensions['BW_fix'] = Dimensions['MSL_x'] - Dimensions['DF_fix_x']
        Dimensions['BW_var'] = B_seaward_var - Dimensions['DF_fix_x']
        Dimensions['BW_der'] = Dimensions['MSL_x'] - Dimensions['DF_der_x'] 
        
        # Calculate DFront width
        Dimensions['DFront_fix_prim_W'] = Dimensions['DF_fix_x'] - Dimensions['DT_prim_x']
        Dimensions['DFront_der_prim_W'] = Dimensions['DF_der_x'] - Dimensions['DT_prim_x']
        Dimensions['DFront_fix_sec_W'] = Dimensions['DF_fix_x'] - Dimensions['DT_sec_x']
        Dimensions['DFront_der_sec_W'] = Dimensions['DF_der_x'] - Dimensions['DT_sec_x']    
        
        # Calculate Beach gradient
        Dimensions['B_grad_fix'] = get_gradient(cross_shore, y_all, years_requested, Dimensions['MSL_x'], Dimensions['DF_fix_x'])
        
        # Calculate dune front gradient            
        Dimensions['DFront_fix_prim_grad'] = get_gradient(cross_shore, y_all, years_requested, Dimensions['DF_fix_x'], Dimensions['DT_prim_x'])
        Dimensions['DFront_fix_sec_grad'] = get_gradient(cross_shore, y_all, years_requested, Dimensions['DF_fix_x'], Dimensions['DT_sec_x'])
        
        # Calculate dune volume
        Dimensions['DVol_fix'] = get_volume(cross_shore, y_all, years_requested, Dimensions['DF_fix_x'], Dimensions['landward_stable_x'])
        Dimensions['DVol_der'] = get_volume(cross_shore, y_all, years_requested, Dimensions['DF_der_x'], Dimensions['landward_stable_x'])
        
        Dimensions.to_pickle(Dir_per_transect + 'Transect_' + trsct + '_dataframe' + '.pickle')
        print('Restored dataframes for transect ' + trsct)