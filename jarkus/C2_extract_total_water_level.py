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
Created on Thu Oct 24 16:47:28 2019

@author: cijzendoornvan
"""

##################################
####          PACKAGES        ####
##################################
import json
import pandas as pd
import numpy as np
import datetime
from scipy.interpolate import griddata
from analysis_functions import find_intersections
import os.path
from jarkus.transects import Transects
from netCDF4 import num2date
import pickle
import matplotlib.pyplot as plt
from IPython import get_ipython
get_ipython().run_line_magic('matplotlib', 'auto') ## %matplotlib auto TO GET WINDOW FIGURE

#################################
####        FUNCTIONS        ####
#################################
def convert2datetime(value):
    t_units = "days since 1970-01-01"    
    date = str(num2date(value, t_units))
    yr = int(date[:4])
    mnth = int(date[5:7])
    day = int(date[8:10])
    dt = datetime.datetime(yr, mnth, day)
    return dt

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

##################################
####    USER-DEFINED REQUEST  ####
##################################
# Set the transect and years for retrieval request
years_requested = list(range(1965, 2020))

# Set whether all transect should be analysed or define a retrieval request
execute_all_transects = True

if execute_all_transects == False:
    # Set the transect and years for retrieval request
    transect_name   = "02_Schiermonnikoog"
    transect_req = [2000103] #np.arange(6002521, 6003100, 1)
    #transect_req = np.arange(8009325, 8009750, 1)
    idxs = np.isin(transect_req, ids) # check which transect are available of those that were requested
    ids_filtered = np.array(transect_req)[np.nonzero(idxs)[0]]
    Dir_plots = settings['Dir_figures'] + transect_name.replace(" ","") + "/"
else:
    # Filter out location that are not suitable for analysis. Based on Kustlijnkaarten 2019, RWS.

    with open("C:/Users/cijzendoornvan/Documents/GitHub/jarkus/jarkus/Filter.txt") as file:
        filter_transects = json.load(file)

    filter_all = []
    for key in filter_transects:
        filter_all += list(range(int(filter_transects[key]["begin"]), int(filter_transects[key]["eind"])))

    ids_filtered = [x for x in ids if x not in filter_all]
    #ids_filtered = [idx for idx in ids_filtered if idx >= 8006450]
    
    Dir_plots = settings['Dir_A']    

Dir_csv = settings['Dir_csv']
Dir_per_transect = settings['Dir_B']
Dir_env_cond = settings['Dir_C1']
Dir_pickles = settings['Dir_C2']

#%%
##################################
####   GET MEASUREMENT DATES  ####
##################################
# Get the dates that correspond to the collection of each yearly measurment per transect
years_requested_str = [str(yr) for yr in years_requested]

for idx in ids_filtered:
    print(idx)    
    # Collect the JARKUS data from the server
    Jk = Transects(url= settings['url'])
    ids = Jk.get_data('id') # ids of all available transects
    
    df, years_available = Jk.get_dataframe(idx, years_requested)
    trsct = str(idx)
    
    pickle_file = Dir_per_transect + 'Transect_' + trsct + '_dataframe.pickle'
    Dimensions = pickle.load(open(pickle_file, 'rb')) #load pickle of transect
        
    # Intialize empty array for saving dates
    topo_date = []
    bathy_date = []
    for i, yr in enumerate(Dimensions.index):
        if yr not in years_available: 
            topo_date.append(np.nan)
            bathy_date.append(np.nan)            
        elif np.isnan(df.loc[trsct, yr]['time_topo'][0]) == False and np.isnan(df.loc[trsct, yr]['time_bathy'][0]) == False:
            # Get dates of Jarkus data collection to define time windows for water level and wave analysis.
            # Get date related to certain transect and year, then convert to datetime
            topo_date.append(convert2datetime(df.loc[trsct, yr]['time_topo'][0])) 
            bathy_date.append(convert2datetime(df.loc[trsct, yr]['time_bathy'][0]))
        elif np.isnan(df.loc[trsct, yr]['time_topo'][0]) == False and np.isnan(df.loc[trsct, yr]['time_topo'][0]) == True:
            topo_date.append(convert2datetime(df.loc[trsct, yr]['time_topo'][0])) 
            bathy_date.append(np.nan)
        elif np.isnan(df.loc[trsct, yr]['time_topo'][0]) == True and np.isnan(df.loc[trsct, yr]['time_topo'][0]) == False:
            topo_date.append(np.nan)
            bathy_date.append(convert2datetime(df.loc[trsct, yr]['time_bathy'][0]))
        else:
            topo_date.append(np.nan)
            bathy_date.append(np.nan)                
            
    Dimensions['topo_date'] = topo_date
    Dimensions['bathy_date'] = bathy_date
  
#%%
##################################
####    CLIP BETWEEN DATES    ####
##################################
# The bathymetry and topographic measurements are taken at different times. To relate the environmental conditions to changes in dimensions along the coast, they have to be clipped using the measurements dates.
# In this case, I want to first focus on dunefoot dynamics, which is recorded in the topographic measurments. So I will relate these dynamics to the environmental conditions that occur between the topographic measurement dates.
    
    # Load environmental data
    wave_height_data = pickle.load(open(Dir_env_cond + 'Wave_heigth_hourly.pickle', 'rb'))
    wave_period_data = pickle.load(open(Dir_env_cond + 'Wave_period_hourly.pickle', 'rb'))
    if idx >= 17000791:
        water_level_data = pickle.load(open(Dir_env_cond + 'Water_level_Cadzand_hourly.pickle', 'rb'))
    elif idx < 17000791 and idx >= 17000000:
        water_level_data = pickle.load(open(Dir_env_cond + 'Water_level_Breskens_hourly.pickle', 'rb'))    
    elif idx < 17000000 and idx >= 16002990:
        water_level_data = pickle.load(open(Dir_env_cond + 'Water_level_Vlissingen_hourly.pickle', 'rb'))    
    elif idx < 16002990 and idx >= 16000900:
        water_level_data = pickle.load(open(Dir_env_cond + 'Water_level_Westkapelle_hourly.pickle', 'rb'))    
    elif idx < 16000900 and idx >= 14000000:
        water_level_data = pickle.load(open(Dir_env_cond + 'Water_level_Roompotbuiten_hourly.pickle', 'rb'))
    elif idx < 14000000 and idx >= 13001465:
        water_level_data = pickle.load(open(Dir_env_cond + 'Water_level_Oosterschelde04_hourly.pickle', 'rb'))
    elif idx < 13001465 and idx >= 13000574:
        water_level_data = pickle.load(open(Dir_env_cond + 'Water_level_Oosterschelde14_hourly.pickle', 'rb'))
    elif idx < 13000574 and idx >= 12001200:
        water_level_data = pickle.load(open(Dir_env_cond + 'Water_level_BrouwershavenscheGat08_hourly.pickle', 'rb'))
    elif idx < 12001200 and idx >= 11000000:
        water_level_data = pickle.load(open(Dir_env_cond + 'Water_level_Stellendambuiten_hourly.pickle', 'rb'))
    elif idx < 11000000 and idx >= 9011221:
        water_level_data = pickle.load(open(Dir_env_cond + 'Water_level_HoekvanHolland_hourly.pickle', 'rb'))
    elif idx < 9011221 and idx >= 8008625:
        water_level_data = pickle.load(open(Dir_env_cond + 'Water_level_Scheveningen_hourly.pickle', 'rb'))
    elif idx < 8008625 and idx >= 7002629:
        water_level_data = pickle.load(open(Dir_env_cond + 'Water_level_IJmuidenbuitenhaven_hourly.pickle', 'rb'))
    elif idx < 7002629 and idx >= 7000000:
        water_level_data = pickle.load(open(Dir_env_cond + 'Water_level_DenHelder_hourly.pickle', 'rb'))
    elif idx < 7000000 and idx >= 5000000:
        water_level_data = pickle.load(open(Dir_env_cond + 'Water_level_TexelNoordzee_hourly.pickle', 'rb'))
    elif idx < 5000000:
        water_level_data = pickle.load(open(Dir_env_cond + 'Water_level_TerschellingNoordzee_hourly.pickle', 'rb'))
    else:
        print("Found no water level file")

    Dimensions['max_TWL_bathy_dis'] = np.nan
    Dimensions['max_TWL_bathy_gen'] = np.nan
    Dimensions['max_TWL_topo_dis'] = np.nan
    Dimensions['max_TWL_topo_gen'] = np.nan

    for i, yr in enumerate(Dimensions.index):
        if i != len(Dimensions.index) - 1:
            end_yr = str(int(yr) + 1)                        

            index_notnull = Dimensions['bathy_date'].notnull()
            if index_notnull[yr] == True and index_notnull[end_yr] == True: # only perform calcualtions if for both years a topography measurement date is available
                # get the measurement date for the current year (in the loop) and the next year
                cutoff_begin = Dimensions['bathy_date'][yr]
                cutoff_end = Dimensions['bathy_date'][end_yr]
                
                # slice environmental data to measurement dates
                WaterLevel_slice = water_level_data.loc[(water_level_data.index >= cutoff_begin) & (water_level_data.index < cutoff_end)]
                WaveHeight_slice = wave_height_data.loc[(wave_height_data.index >= cutoff_begin) & (wave_height_data.index < cutoff_end)]
                WavePeriod_slice = wave_period_data.loc[(wave_period_data.index >= cutoff_begin) & (wave_period_data.index < cutoff_end)]
                
                time = WaterLevel_slice.index
                h = WaterLevel_slice['NUMERIEKEWAARDE']/100
                h[h > 1500] = np.nan # Filter nan values
                H0 = WaveHeight_slice['NUMERIEKEWAARDE']/100
                H0[H0 > 1500] = np.nan # Filter nan values
                T0 = WavePeriod_slice['NUMERIEKEWAARDE']
                T0[T0 > 1500] = np.nan # Filter nan values
                
                if len(H0) < len(h) and len(T0) < len(h):
                    print('TWL cannot be calculated for period ' + yr + '-' + end_yr + ' because values of both the wave height and wave period are missing')    
                elif len(H0) < len(h) and len(T0) == len(h):
                    print('TWL cannot be calculated for period ' + yr + '-' + end_yr + ' because values of the wave height are missing')    
                elif len(H0) == len(h) and len(T0) < len(h):
                    print('TWL cannot be calculated for period ' + yr + '-' + end_yr + ' because values of the wave period are missing')    
                else:
                    # calculation of wave length for deep water
                    g = 9.81;                           # gravitational acceleration
                    L0 = g * T0**2 / (2 * np.pi)        # this method was compared to that of Fenton (1988) and they gave similar results.
                    
                    # get the beach slope
                    beach_slope = abs(Dimensions['B_grad_fix'].mean()) # THIS CAN BE CHANGED TO USE BEACH SLOPE PER YEAR

                    # calculate eta to test whether beach is dissipative or not
                    eta = np.nanmean(beach_slope / (H0/L0)**(0.5)) # Now the mean is used to show that the beach is dissipative on average.
            
                    # calculate R_2% based on Stockdon et al. (2006), corresponding to the 2% exceedance percentil of extreme runup maxima on sandy beaches
                    runup_diss = 0.043*(H0*L0)**(0.5) # if eta < 0.3 -> dissipative beach
                    #runup = H0*0.18/0.85 # Based on values of Ruessink et al. (1998) on Terschelling. Gives similar results as dissipative formula from Stockdon.
                    # for all natural beaches, gives smaller vallues than the equation for dissipative beaches.
                    runup_general = 1.1 * (0.35 * beach_slope * (H0*L0)**(0.5) + 0.5 * (H0*L0*(0.563*beach_slope**2 + 0.002))**(0.5))
            
                    # calculate total water level based on the local water level (i.e. a measurement station close to the transect) and the runup
                    TWL_diss = h + runup_diss
                    TWL_general = h + runup_general

                    # save TWL in dataframe
                    Dimensions.loc[yr, 'max_TWL_bathy_dis'] = TWL_diss.max()
                    Dimensions.loc[yr, 'max_TWL_bathy_gen'] = TWL_general.max()
            
            elif index_notnull[yr] == False and index_notnull[end_yr] == True:
                print('TWL is not calculated for period ' + yr + '-' + end_yr + ' because elevation data is missing for ' + yr)
            elif index_notnull[yr] == True and index_notnull[end_yr] == False:
                print('TWL is not calculated for period ' + yr + '-' + end_yr + ' because elevation data is missing for ' + end_yr)
            else:
                print('TWL is not calculated for period ' + yr + '-' + end_yr + ' because elevation data is missing for ' + yr + ' and ' + end_yr)
            
            index_notnull = Dimensions['topo_date'].notnull()
            if index_notnull[yr] == True and index_notnull[end_yr] == True: # only perform calcualtions if for both years a topography measurement date is available
                # get the measurement date for the current year (in the loop) and the next year
                cutoff_begin = Dimensions['topo_date'][yr]
                cutoff_end = Dimensions['topo_date'][end_yr]
                
                # slice environmental data to measurement dates
                WaterLevel_slice = water_level_data.loc[(water_level_data.index >= cutoff_begin) & (water_level_data.index < cutoff_end)]
                WaveHeight_slice = wave_height_data.loc[(wave_height_data.index >= cutoff_begin) & (wave_height_data.index < cutoff_end)]
                WavePeriod_slice = wave_period_data.loc[(wave_period_data.index >= cutoff_begin) & (wave_period_data.index < cutoff_end)]
                
                time = WaterLevel_slice.index
                h = WaterLevel_slice['NUMERIEKEWAARDE']/100
                h[h > 1500] = np.nan # Filter nan values
                H0 = WaveHeight_slice['NUMERIEKEWAARDE']/100
                H0[H0 > 1500] = np.nan # Filter nan values
                T0 = WavePeriod_slice['NUMERIEKEWAARDE']
                T0[T0 > 1500] = np.nan # Filter nan values
                
                if len(H0) < len(h) and len(T0) < len(h):
                    print('TWL cannot be calculated for period ' + yr + '-' + end_yr + ' because values of both the wave height and wave period are missing')    
                elif len(H0) < len(h) and len(T0) == len(h):
                    print('TWL cannot be calculated for period ' + yr + '-' + end_yr + ' because values of the wave height are missing')    
                elif len(H0) == len(h) and len(T0) < len(h):
                    print('TWL cannot be calculated for period ' + yr + '-' + end_yr + ' because values of the wave period are missing')    
                else:
                    # calculation of wave length for deep water
                    g = 9.81;                           # gravitational acceleration
                    L0 = g * T0**2 / (2 * np.pi)        # this method was compared to that of Fenton (1988) and they gave similar results.
                    
                    # get the beach slope
                    beach_slope = abs(Dimensions['B_grad_fix'].mean()) # THIS CAN BE CHANGED TO USE BEACH SLOPE PER YEAR

                    # calculate eta to test whether beach is dissipative or not
                    eta = np.nanmean(beach_slope / (H0/L0)**(0.5)) # Now the mean is used to show that the beach is dissipative on average.
            
                    # calculate R_2% based on Stockdon et al. (2006), corresponding to the 2% exceedance percentil of extreme runup maxima on sandy beaches
                    runup_diss = 0.043*(H0*L0)**(0.5) # if eta < 0.3 -> dissipative beach
                    #runup = H0*0.18/0.85 # Based on values of Ruessink et al. (1998) on Terschelling. Gives similar results as dissipative formula from Stockdon.
                    # for all natural beaches, gives smaller vallues than the equation for dissipative beaches.
                    runup_general = 1.1 * (0.35 * beach_slope * (H0*L0)**(0.5) + 0.5 * (H0*L0*(0.563*beach_slope**2 + 0.002))**(0.5))
                    
                    # calculate total water level based on the local water level (i.e. a measurement station close to the transect) and the runup
                    TWL_diss = h + runup_diss
                    TWL_general = h + runup_general
                    
                    # save TWL in dataframe
                    Dimensions.loc[yr, 'max_TWL_topo_dis'] = TWL_diss.max()
                    Dimensions.loc[yr, 'max_TWL_topo_gen'] = TWL_general.max()
            
            elif index_notnull[yr] == False and index_notnull[end_yr] == True:
                print('TWL is not calculated for period ' + yr + '-' + end_yr + ' because elevation data is missing for ' + yr)
            elif index_notnull[yr] == True and index_notnull[end_yr] == False:
                print('TWL is not calculated for period ' + yr + '-' + end_yr + ' because elevation data is missing for ' + end_yr)
            else:
                print('TWL is not calculated for period ' + yr + '-' + end_yr + ' because elevation data is missing for ' + yr + ' and ' + end_yr)
            
        else: 
            print('TWL is not calculated because ' + yr + ' is the last year available')
            
#%%
###################################
###      Load elevation data    ### 
################################### 
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
###      Get TWL x-location     ### 
###################################     
    Dimensions['max_TWL_bathy_dis_x'] = np.nan
    Dimensions['max_TWL_topo_dis_x'] = np.nan
    Dimensions['max_TWL_bathy_gen_x'] = np.nan
    Dimensions['max_TWL_topo_gen_x'] = np.nan
    for i, yr in enumerate(years_requested_str):
        TWL_y_topo_dis   = Dimensions['max_TWL_topo_dis'][yr]
        TWL_y_topo_gen   = Dimensions['max_TWL_topo_gen'][yr]
        TWL_y_bathy_dis   = Dimensions['max_TWL_bathy_dis'][yr]
        TWL_y_bathy_gen   = Dimensions['max_TWL_bathy_gen'][yr]
        
        if np.isnan(TWL_y_topo_dis) == False and yr in years_available:
            intersect_idx_TWL = find_intersections(cross_shore, y_all[:,i], TWL_y_topo_dis)
            if len(intersect_idx_TWL[0]) != 0:
                idx = intersect_idx_TWL[0][-1]
                Dimensions.loc[yr, 'max_TWL_topo_dis_x'] = cross_shore[idx]
                
        if np.isnan(TWL_y_topo_gen) == False and yr in years_available:
            intersect_idx_TWL = find_intersections(cross_shore, y_all[:,i], TWL_y_topo_gen)
            if len(intersect_idx_TWL[0]) != 0:
                idx = intersect_idx_TWL[0][-1]
                Dimensions.loc[yr, 'max_TWL_topo_gen_x'] = cross_shore[idx]
            
        if np.isnan(TWL_y_bathy_dis) == False and yr in years_available:
            intersect_idx_TWL = find_intersections(cross_shore, y_all[:,i], TWL_y_bathy_dis)
            if len(intersect_idx_TWL[0]) != 0:
                idx = intersect_idx_TWL[0][-1]
                Dimensions.loc[yr, 'max_TWL_bathy_dis_x'] = cross_shore[idx]
                
        if np.isnan(TWL_y_bathy_gen) == False and yr in years_available:
            intersect_idx_TWL = find_intersections(cross_shore, y_all[:,i], TWL_y_bathy_gen)
            if len(intersect_idx_TWL[0]) != 0:
                idx = intersect_idx_TWL[0][-1]
                Dimensions.loc[yr, 'max_TWL_bathy_gen_x'] = cross_shore[idx]
                
        else:
            pass

###################################
###        SAVE DATAFRAME       ### 
###################################
# Save dataframe for each transect. Later the added info cna be used to relate morphological changes and environmental conditions.
    Dimensions.to_pickle(Dir_pickles + 'Transect_' + trsct + '_dataframe.pickle')
""" 

plt.plot(time, h)
plt.plot(time, H0)
#plt.plot(time, T0)
#plt.plot(time, L0)
plt.plot(time, runup)
plt.plot(time, TWL)
plt.legend(['water level', 'wave heigth', 'runup', 'TWL'])
"""      

  
