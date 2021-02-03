
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 10 17:24:40 2020

@author: cijzendoornvan
"""
######################
# PACKAGES
######################
import json
import os
import pickle
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
DirDimensions = settings['Dir_D1']
DirVarPlots = settings['Dir_D2']

######################
# USER-DEFINED REQUEST
######################
start_yr = 1965                                                                # USER-DEFINED request for years
end_yr = 2020
trscts_requested = 8009325
#trscts_requested = [8009325, 8009350]
#trscts_requested = np.arange(8009000, 8009751, 1)                               # USER-DEFINED request for transects

# Set whether all transect should be analysed or define a retrieval request
execute_all_transects = False

# Select which dimensions should be calculated for the transects
dune_height_and_location    = False

mean_sea_level              = False
mean_low_water_fixed        = False
mean_low_water_variable     = False
mean_high_water_fixed       = False
mean_high_water_variable    = False
mean_sea_level_variable     = False

intertidal_width_fixed      = False
intertidal_width_variable   = False

landward_point_variance     = False
landward_point_derivative   = False
landward_point_bma          = False

seaward_point_foreshore     = False
seaward_point_activeprofile = False
seaward_point_doc           = False

dune_foot_fixed             = False
dune_foot_derivative        = False
dune_foot_pybeach           = False

beach_width_fix             = False
beach_width_var             = False
beach_width_der             = False
beach_width_der_var         = False

beach_gradient_fix          = False
beach_gradient_var          = False
beach_gradient_der          = True

dune_front_width_prim_fix   = False
dune_front_width_prim_der   = False
dune_front_width_sec_fix    = False
dune_front_width_sec_der    = False

dune_front_gradient_prim_fix= False
dune_front_gradient_prim_der= False
dune_front_gradient_sec_fix = False
dune_front_gradient_sec_der = False

dune_volume_fix             = False
dune_volume_der             = True

intertidal_gradient         = False
intertidal_volume_fix       = False
intertidal_volume_var       = False

foreshore_gradient          = False
foreshore_volume            = False

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
# TRSCT FILTERING
######################
        
if execute_all_transects == False:
    # filter requested transects to make sure only existing transects are used
    trscts_filtered, trscts_filtered_idxs = TB.get_transects_filtered(trscts_requested, variables)
else:
    trscts_requested = variables['id'].values
    trscts_filtered, trscts_filtered_idxs = TB.get_transects_filtered(trscts_requested, variables)
    
    # # Use this if you want to skip transects, e.g. if your battery died during running...
    # trscts_filtered = trscts_filtered[1379:]
    # trscts_filtered_idxs = trscts_filtered_idxs[1379:]
#%%
######################
# LOOPING through TRSCTS
######################

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
    
######################
# VISUALISATION
######################
    
    # saves plots of all yearly profiles of specific transect
    TB.get_transect_plot(elevation_dataframe, trsct, DirPlots)
    
######################
# DIMENSION EXTRACTION
######################
    
    trsct = str(trsct)
    file_name = 'Transect_' + trsct + '_dataframe.pickle'
    pickle_file = DirDataframes + file_name
    
    if file_name in os.listdir(DirDataframes):
        dimensions = pickle.load(open(pickle_file, 'rb')) #load pickle of dimension    
    else:
        dimensions = pd.DataFrame({'transect': trsct, 'years':elevation_dataframe.index})
        dimensions.set_index('years', inplace=True)
    
    # extract dimensions
    # dimensions = TB.get_dune_height_and_location(dune_height_and_location, elevation_dataframe, dimensions)
    
    # dimensions = TB.get_mean_sea_level(mean_sea_level, elevation_dataframe, dimensions)
    # dimensions = TB.get_mean_low_water_fixed(mean_low_water_fixed, elevation_dataframe, dimensions)
    # dimensions = TB.get_mean_low_water_variable(mean_low_water_variable, elevation_dataframe, df, dimensions)
    # dimensions = TB.get_mean_high_water_fixed(mean_high_water_fixed, elevation_dataframe, dimensions)
    # dimensions = TB.get_mean_high_water_variable(mean_high_water_variable, elevation_dataframe, df, dimensions)
    # dimensions = TB.get_mean_sea_level_variable(mean_sea_level_variable, dimensions)
    
    # dimensions = TB.get_intertidal_width_fixed(intertidal_width_fixed, dimensions)
    # dimensions = TB.get_intertidal_width_variable(intertidal_width_variable, dimensions)
    
    # dimensions = TB.get_landwardpoint_variance(landward_point_variance, elevation_dataframe, dimensions)
    # dimensions = TB.get_landwardpoint_derivative(landward_point_derivative, elevation_dataframe, dimensions)
    # dimensions = TB.get_landwardpoint_bma(landward_point_bma, elevation_dataframe, dimensions)
    
    # dimensions = TB.get_seawardpoint_foreshore(seaward_point_foreshore, elevation_dataframe, dimensions)
    # dimensions = TB.get_seawardpoint_activeprofile(seaward_point_activeprofile, elevation_dataframe, dimensions)
    # dimensions = TB.get_seawardpoint_depthofclosure(seaward_point_doc, elevation_dataframe, dimensions)
    
    # dimensions = TB.get_dune_foot_fixed(dune_foot_fixed, elevation_dataframe, dimensions)
    # dimensions = TB.get_dune_foot_derivative(dune_foot_derivative, elevation_dataframe, dimensions)    
    # dimensions = TB.get_dune_foot_pybeach(dune_foot_pybeach, elevation_dataframe, dimensions) 
    
    # dimensions = TB.get_beach_width_fix(beach_width_fix, elevation_dataframe, dimensions)
    # dimensions = TB.get_beach_width_var(beach_width_var, elevation_dataframe, dimensions)
    # dimensions = TB.get_beach_width_der(beach_width_der, elevation_dataframe, dimensions)
    # dimensions = TB.get_beach_width_der_var(beach_width_der_var, elevation_dataframe, dimensions)
    
    # dimensions = TB.get_beach_gradient_fix(beach_gradient_fix, elevation_dataframe, dimensions)
    # dimensions = TB.get_beach_gradient_var(beach_gradient_var, elevation_dataframe, dimensions)
    dimensions = TB.get_beach_gradient_der(beach_gradient_der, elevation_dataframe, dimensions)
    
    # dimensions = TB.get_dune_front_width_prim_fix(dune_front_width_prim_fix, elevation_dataframe, dimensions)
    # dimensions = TB.get_dune_front_width_prim_der(dune_front_width_prim_der, elevation_dataframe, dimensions)
    # dimensions = TB.get_dune_front_width_sec_fix(dune_front_width_sec_fix, elevation_dataframe, dimensions)
    # dimensions = TB.get_dune_front_width_sec_der(dune_front_width_sec_der, elevation_dataframe, dimensions)
    
    # dimensions = TB.get_dune_front_gradient_prim_fix(dune_front_gradient_prim_fix, elevation_dataframe, dimensions)
    # dimensions = TB.get_dune_front_gradient_prim_der(dune_front_gradient_prim_der, elevation_dataframe, dimensions)
    # dimensions = TB.get_dune_front_gradient_sec_fix(dune_front_gradient_sec_fix, elevation_dataframe, dimensions)
    # dimensions = TB.get_dune_front_gradient_sec_der(dune_front_gradient_sec_der, elevation_dataframe, dimensions)    
    
    # dimensions = TB.get_dune_volume_fix(dune_volume_fix, elevation_dataframe, dimensions)
    dimensions = TB.get_dune_volume_der(dune_volume_der, elevation_dataframe, dimensions)
    
    # dimensions = TB.get_intertidal_gradient(intertidal_gradient, elevation_dataframe, dimensions)
    # dimensions = TB.get_intertidal_volume_fix(intertidal_volume_fix, elevation_dataframe, dimensions)
    # dimensions = TB.get_intertidal_volume_var(intertidal_volume_var, elevation_dataframe, dimensions)
    
    # dimensions = TB.get_foreshore_gradient(foreshore_gradient, elevation_dataframe, dimensions)
    # dimensions = TB.get_foreshore_volume(foreshore_volume, elevation_dataframe, dimensions)
    
    dimensions = TB.get_active_profile_gradient(active_profile_gradient, elevation_dataframe, dimensions)
    dimensions = TB.get_active_profile_volume(active_profile_volume, elevation_dataframe, dimensions)
    
######################
# SAVE DATAFRAME
######################
    
    TB.save_dimensions_dataframe(dimensions, DirDataframes)
    
# END OF LOOP

#%%
######################
# CREATE VARIABLE DF
######################
import pickle

trsct = 7000000
pickle_file = DirDataframes + 'Transect_' + str(trsct) + '_dataframe.pickle'
dimensions = pickle.load(open(pickle_file, 'rb')) #load pickle of dimension   

variables = list(dimensions.columns)
variables = ['Active_profile_volume', 'Active_profile_gradient', 'Seaward_x_DoC', 'DuneVol_der', 'Beach_gradient_der']

# Convert dataframes with variables per transect to dataframes with values per transect for each variable
for i in range(len(variables)):    
    TB.extract_variable(variables[i], DirDataframes, DirDimensions, trscts_filtered, start_yr, end_yr)

######################
# NORMALIZE VARIABLES
######################

# Each cross-shore dimension is normalized based on the x-location of the MSL in 1999.
# This is done to remove unnecessary variations from the distribution plot, making it more readable
norm_variable = 'MSL_x'
norm_year = 1999

normalized_variables = TB.normalize_variables(DirDimensions, variables, norm_variable, norm_year)

#%%
######################
# VARIABLE PLOTS
######################
import os
import pickle
import matplotlib.pyplot as plt

trsct = 2000100
pickle_file = DirDataframes + 'Transect_' + str(trsct) + '_dataframe.pickle'
dimensions = pickle.load(open(pickle_file, 'rb')) #load pickle of dimension    

variables = list(dimensions.columns)
variables = ['Active_profile_volume', 'Active_profile_gradient', 'Seaward_x_DoC', 'DuneVol_der', 'Beach_gradient_der']

trscts_vis = trscts_filtered

years_requested = list(range(start_yr, end_yr))
labels_y = [str(yr) for yr in years_requested][0::5]
labels_x = [str(tr) for tr in trscts_filtered][0::25]

for i, var in enumerate(variables):
    print(var)
    pickle_file = DirDimensions + variables[i] + '_dataframe.pickle'    
    if os.path.exists(pickle_file):
        variable_df = pickle.load(open(pickle_file, 'rb')) #load pickle of dimension    
        plt.figure(figsize=(30,15))
        average = np.nanmean(variable_df[trscts_vis].values)
        stddev = np.nanstd(variable_df[trscts_vis].values, ddof=1)
        range_value = 2*stddev
        fig = plt.pcolor(variable_df[trscts_vis], vmin = average-range_value, vmax = average + range_value)
        plt.title(settings[var])
        ticks_y = range(0, len(years_requested))[0::5]
        ticks_x = range(0, len(trscts_vis))[0::25]
        plt.yticks(ticks_y, labels_y)
        plt.xticks(ticks_x, labels_x, rotation='vertical')
        plt.colorbar()
        plt.savefig(DirVarPlots + var + '_plot.png')
        pickle.dump(fig, open(DirVarPlots + var + '_plot.fig.pickle', 'wb'))
        
        #plt.show()
        plt.close()
    
    #set(ax, 'YTickLabel', years_requested)
    else:
        print(pickle_file + ' was not available.')
        continue
    

# REOPEN FIGURE EXAMPLE


Time2 = time.time()
print('Extracting time ' + str(Time2 - Time1) + ' seconds')