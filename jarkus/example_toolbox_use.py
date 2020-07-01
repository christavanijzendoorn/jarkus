
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 10 17:24:40 2020

@author: cijzendoornvan
"""
######################
# PACKAGES
######################
import json
import numpy as np
import pandas as pd
import Jarkus_Analysis_Toolbox as TB

######################
# LOAD SETTINGS
######################
import time
Time1 = time.time()

# load basic settings from .txt file
with open("C:/Users/cijzendoornvan/Documents/GitHub/jarkus/jarkus/Settings.txt") as file:
    settings = json.load(file)                                                  # include USER-DEFINED settings

DirJarkus = settings['Dir_Jarkus']
DirPlots = settings['Dir_A']
DirDataframes = settings['Dir_B']

######################
# USER-DEFINED REQUEST
######################
start_yr = 1965                                                                 # USER-DEFINED request for years
end_yr = 2020
#trscts_requested = 8009325
#trscts_requested = [8009325, 8009350]
trscts_requested = np.arange(8009000, 8009751, 1)                               # USER-DEFINED request for transects

# Set whether all transect should be analysed or define a retrieval request
execute_all_transects = True

# Select which dimensions should be calculated for the transects
dune_height_and_location    = True

mean_sea_level              = True
mean_low_water_fixed        = True
mean_low_water_variable     = True
mean_high_water_fixed       = True
mean_high_water_variable    = True
mean_sea_level_variable     = True

intertidal_width_fixed      = True
intertidal_width_variable   = True

landward_point_variance     = True
landward_point_derivative   = True
landward_point_bma          = True

seaward_point_foreshore     = True
seaward_point_activeprofile = True

dune_foot_fixed             = True
dune_foot_derivative        = True
dune_foot_pybeach           = True

beach_width_fix             = True
beach_width_var             = True
beach_width_der             = True
beach_width_der_var         = True

beach_gradient_fix          = True
beach_gradient_var          = True
beach_gradient_der          = True

dune_front_width_prim_fix   = True
dune_front_width_prim_der   = True
dune_front_width_sec_fix    = True
dune_front_width_sec_der    = True

dune_front_gradient_prim_fix= True
dune_front_gradient_prim_der= True
dune_front_gradient_sec_fix = True
dune_front_gradient_sec_der = True

dune_volume_fix             = True
dune_volume_der             = True

intertidal_gradient         = True
intertidal_volume_fix       = True
intertidal_volume_var       = True

foreshore_gradient          = True
foreshore_volume            = True

active_profile_gradient     = True
active_profile_volume       = True


######################
# LOAD DATA
######################
# Load jarkus dataset
dataset, variables = TB.get_jarkus_data(DirJarkus)

# Filter for locations that are not suitable for analysis. Based on Kustlijnkaarten 2019, RWS.
with open("C:/Users/cijzendoornvan/Documents/GitHub/jarkus/jarkus/Filter.txt") as file:
        filter_transects = json.load(file)
        
######################
# VISUALISATION
######################
        
if execute_all_transects == False:
    # filter requested transects to make sure only existing transects are used
    trscts_filtered, trscts_filtered_idxs = TB.get_transects_filtered(trscts_requested, variables)
else:
    trscts_requested = variables['id'].values
    
    # Filter out location that are not suitable for analysis. Based on Kustlijnkaarten 2019, RWS.
    with open("C:/Users/cijzendoornvan/Documents/GitHub/jarkus/jarkus/Filter.txt") as file:
        filter_transects = json.load(file)
    
    filter_all = []
    for key in filter_transects:
        filter_all += list(range(int(filter_transects[key]["begin"]), int(filter_transects[key]["eind"])))
        
    # Use this if you want to skip transects, e.g. if your battery died during running...
    skip_transects = list(trscts_requested[0:1643])
    filter_all = filter_all + skip_transects
    
    trscts_filtered = [x for x in trscts_requested if x not in filter_all]
    trscts_filtered_idxs = [idx for idx, x in enumerate(trscts_requested) if x not in filter_all]

for i, trsct in enumerate(trscts_filtered):
    print(trsct)
    #trsct = trscts_filtered[0]
    #i = 0
    
    # extract all years and variables of specific transect into dataframe
    df = TB.get_dataframe_transect(variables, start_yr, end_yr, trsct, trscts_filtered_idxs[i])
    
    # filter years in which elevation data in not available between user-defined values
    min_elevation = -1
    max_elevation = 5
    
    # convert to only elevation data, requires an unindexed dataframe and filter based on min, max values.
    elevation_dataframe = TB.get_elevations_dataframe(df, min_elevation, max_elevation)
    
    # to get single year in dataframe
    df.set_index('year', inplace=True)
    #df_1965 = df[1965]
    
    # saves plots of all yearly profiles of specific transect
    TB.get_transect_plot(elevation_dataframe, trsct, DirPlots)
    
    dimensions = pd.DataFrame({'transect': trsct, 'years':elevation_dataframe.index})
    dimensions.set_index('years', inplace=True)
    
    # extract dimensions
    dimensions = TB.get_dune_height_and_location(dune_height_and_location, elevation_dataframe, dimensions)
    
    dimensions = TB.get_mean_sea_level(mean_sea_level, elevation_dataframe, dimensions)
    dimensions = TB.get_mean_low_water_fixed(mean_low_water_fixed, elevation_dataframe, dimensions)
    dimensions = TB.get_mean_low_water_variable(mean_low_water_variable, elevation_dataframe, df, dimensions)
    dimensions = TB.get_mean_high_water_fixed(mean_high_water_fixed, elevation_dataframe, dimensions)
    dimensions = TB.get_mean_high_water_variable(mean_high_water_variable, elevation_dataframe, df, dimensions)
    dimensions = TB.get_mean_sea_level_variable(mean_sea_level_variable, dimensions)
    
    dimensions = TB.get_intertidal_width_fixed(intertidal_width_fixed, dimensions)
    dimensions = TB.get_intertidal_width_variable(intertidal_width_variable, dimensions)
    
    dimensions = TB.get_landwardpoint_variance(landward_point_variance, elevation_dataframe, dimensions)
    dimensions = TB.get_landwardpoint_derivative(landward_point_derivative, elevation_dataframe, dimensions)
    dimensions = TB.get_landwardpoint_bma(landward_point_bma, elevation_dataframe, dimensions)
    
    dimensions = TB.get_seawardpoint_foreshore(seaward_point_foreshore, elevation_dataframe, dimensions)
    dimensions = TB.get_seawardpoint_activeprofile(seaward_point_activeprofile, elevation_dataframe, dimensions)
    
    dimensions = TB.get_dune_foot_fixed(dune_foot_fixed, elevation_dataframe, dimensions)
    dimensions = TB.get_dune_foot_derivative(dune_foot_derivative, elevation_dataframe, dimensions)    
    dimensions = TB.get_dune_foot_pybeach(dune_foot_pybeach, elevation_dataframe, dimensions) 
    
    dimensions = TB.get_beach_width_fix(beach_width_fix, elevation_dataframe, dimensions)
    dimensions = TB.get_beach_width_var(beach_width_var, elevation_dataframe, dimensions)
    dimensions = TB.get_beach_width_der(beach_width_der, elevation_dataframe, dimensions)
    dimensions = TB.get_beach_width_der_var(beach_width_der_var, elevation_dataframe, dimensions)
    
    dimensions = TB.get_beach_gradient_fix(beach_gradient_fix, elevation_dataframe, dimensions)
    dimensions = TB.get_beach_gradient_var(beach_gradient_var, elevation_dataframe, dimensions)
    dimensions = TB.get_beach_gradient_der(beach_gradient_der, elevation_dataframe, dimensions)
    
    dimensions = TB.get_dune_front_width_prim_fix(dune_front_width_prim_fix, elevation_dataframe, dimensions)
    dimensions = TB.get_dune_front_width_prim_der(dune_front_width_prim_der, elevation_dataframe, dimensions)
    dimensions = TB.get_dune_front_width_sec_fix(dune_front_width_sec_fix, elevation_dataframe, dimensions)
    dimensions = TB.get_dune_front_width_sec_der(dune_front_width_sec_der, elevation_dataframe, dimensions)
    
    dimensions = TB.get_dune_front_gradient_prim_fix(dune_front_gradient_prim_fix, elevation_dataframe, dimensions)
    dimensions = TB.get_dune_front_gradient_prim_der(dune_front_gradient_prim_der, elevation_dataframe, dimensions)
    dimensions = TB.get_dune_front_gradient_sec_fix(dune_front_gradient_sec_fix, elevation_dataframe, dimensions)
    dimensions = TB.get_dune_front_gradient_sec_der(dune_front_gradient_sec_der, elevation_dataframe, dimensions)    
    
    dimensions = TB.get_dune_volume_fix(dune_volume_fix, elevation_dataframe, dimensions)
    dimensions = TB.get_dune_volume_der(dune_volume_der, elevation_dataframe, dimensions)
    
    dimensions = TB.get_intertidal_gradient(intertidal_gradient, elevation_dataframe, dimensions)
    dimensions = TB.get_intertidal_volume_fix(intertidal_volume_fix, elevation_dataframe, dimensions)
    dimensions = TB.get_intertidal_volume_var(intertidal_volume_var, elevation_dataframe, dimensions)
    
    dimensions = TB.get_foreshore_gradient(foreshore_gradient, elevation_dataframe, dimensions)
    dimensions = TB.get_foreshore_volume(foreshore_volume, elevation_dataframe, dimensions)
    
    dimensions = TB.get_active_profile_gradient(active_profile_gradient, elevation_dataframe, dimensions)
    dimensions = TB.get_active_profile_volume(active_profile_volume, elevation_dataframe, dimensions)
    
    #%%    
    ###################################
    ###        SAVE DATAFRAME       ### 
    ###################################
    
    TB.save_dimensions_dataframe(dimensions, DirDataframes)

# REOPEN FIGURE EXAMPLE


Time2 = time.time()
print('Extracting time ' + str(Time2 - Time1) + ' seconds')