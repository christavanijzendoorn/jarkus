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
import matplotlib.dates as mdates
import datetime
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

with open("C:/Users/cijzendoornvan/Documents/GitHub/jarkus/jarkus/Filter.txt") as file:
    filter_transects = json.load(file)

filter_all = []
for key in filter_transects:
    filter_all += list(range(int(filter_transects[key]["begin"]), int(filter_transects[key]["eind"])))

ids_filtered = [x for x in ids if x not in filter_all]

Dir_csv = settings['Dir_csv']
Dir_env_cond = settings['Dir_C1']
Dir_per_variable = settings['Dir_D1']

#%%
##################################
####     Load wave data       ####
##################################

# Load environmental data
wave_height_data = pickle.load(open(Dir_env_cond + 'Wave_heigth_hourly.pickle', 'rb'))
wave_period_data = pickle.load(open(Dir_env_cond + 'Wave_period_hourly.pickle', 'rb'))

##################################
####   EXTRACT BEACH SLOPE    ####
##################################
   
# load dataframe with distribution of values of variable
variables = ['B_grad_fix']
pickle_file = Dir_per_variable + variables[0] + '_dataframe.pickle'    

#### Filtering based on water level station ####
StationTransect = pd.read_excel("C:/Users/cijzendoornvan/Documents/Data/JARKUS/WaterLevelLocations.xlsx")
region_remove = []
for index, row in StationTransect.iterrows():
    VariableValues = pickle.load(open(pickle_file, 'rb')) #load pickle of dimension    
    code_beginraai = row['BeginRaai']
    code_eindraai = row['EindRaai']
    station_region = [str(i) for i in ids_filtered if i >= code_beginraai and i < code_eindraai]
    
    # Extra beach slope only for region corresponding to the water level station location
    for c in VariableValues.columns: # for each column check whether it is in the remove list, if it is set all values to nan
        if c not in station_region:
            VariableValues[c] = np.nan
    
    # get the beach slope
    beach_slope =  abs(VariableValues.mean(axis = 1, skipna = True).mean(skipna = True)) # THIS CAN BE CHANGED TO USE BEACH SLOPE PER YEAR
    
    location_name = row['Locatie']
    water_level_data = pickle.load(open(Dir_env_cond + 'Water_level_' + location_name + '_hourly.pickle', 'rb')) 
            
    start = max([wave_height_data.index[1], wave_period_data.index[1], water_level_data.index[1]])
    end = min([wave_height_data.index[-1], wave_period_data.index[-1], water_level_data.index[-1]])
    
    WaterLevel_slice = water_level_data.loc[(water_level_data.index >= start) & (water_level_data.index < end)]
    WaveHeight_slice = wave_height_data.loc[(wave_height_data.index >= start) & (wave_height_data.index < end)]
    WavePeriod_slice = wave_period_data.loc[(wave_period_data.index >= start) & (wave_period_data.index < end)]
         
    time = WaterLevel_slice.index
    h = WaterLevel_slice['NUMERIEKEWAARDE']/100
    h[h > 1500] = np.nan # Filter nan values
    H0 = WaveHeight_slice['NUMERIEKEWAARDE']/100
    H0[H0 > 1500] = np.nan # Filter nan values
    T0 = WavePeriod_slice['NUMERIEKEWAARDE']
    T0[T0 > 1500] = np.nan # Filter nan values
    
    ##################################
    ####      CALCULATE TWL       ####
    ##################################

    # calculation of wave length for deep water
    g = 9.81;                           # gravitational acceleration
    L0 = g * T0**2 / (2 * np.pi)        # this method was compared to that of Fenton (1988) and they gave similar results.

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

    ###################################
    ###        SAVE DATAFRAME       ### 
    ###################################
    # Save dataframe for water level station location
    TWL_general.to_pickle(Dir_env_cond + 'TWL_' + location_name + '_hourly.pickle')


    

