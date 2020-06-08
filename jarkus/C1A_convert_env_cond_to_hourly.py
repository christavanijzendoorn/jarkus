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
import datetime
import os.path
from jarkus.transects import Transects
from netCDF4 import num2date
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

def parser(dates, hours):
    date_time = []
    for i in range(len(dates)-1):
        yr = int(dates[i][-4:])
        m = int(dates[i][3:5])
        d = int(dates[i][:2])
        hr = int(hours[i][:2])
        mn = int(hours[i][3:5])
        s = int(hours[i][6:8])
        date = datetime.datetime(yr, m, d, hr, mn, s)
        date_time.append(date)
    return pd.DataFrame({'date_time': date_time})

def csv2hourly_data(csv_file_location):
    try:
        data = pd.read_csv(csv_file_location, encoding = "ISO-8859-1", sep = ';')
    except:
        data = pd.read_csv(csv_file_location, encoding = "ISO-8859-1", sep = ',')
    date_time = parser(data['WAARNEMINGDATUM'], data['WAARNEMINGTIJD']) # Convert date and time into one value for the measurement time
    if "period" in csv_file_location:
        data.loc['NUMERIEKEWAARDE'] = data.loc['NUMERIEKEWAARDE'].str.replace(',','.').astype(float)
    data_df = pd.concat([data['NUMERIEKEWAARDE'],date_time], axis = 1) # Combine the values and measurement time in one dataframe
    data_df.set_index('date_time', inplace=True) # Set the time column as the index
    data_df_filt = data_df.loc[~data_df.index.duplicated(keep='first')] # Remove duplicated rows
    # Convert to hourly data, so calculation using this and the other environmental conditions is possible.
    data_hourly = data_df_filt.resample(rule = 'H', closed = 'left', label = 'left').interpolate(method = 'linear') # 'Resample' provides space in dataframe where necessary, 'interpolate' fills in empty values
    print("Converted " + csv_file_location)
    return data_hourly

def WaterLevel_convert(Directory_pickles, Directory_csv, location_name):
    if os.path.exists(Directory_pickles + 'Water_level_' + location_name + '_hourly.pickle'):
        print(Directory_pickles + 'Water_level_' + location_name + '_hourly.pickle' + ' already exists')
    else:
        water_level_data = csv2hourly_data(Directory_csv + "water_level_" + location_name + ".csv")
        water_level_data.to_pickle(Directory_pickles + 'Water_level_' + location_name + '_hourly.pickle')   

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

Dir_csv = settings['Dir_csv']
Dir_pickles = settings['Dir_C1']

  
#%%        
##################################
####   SPLIT UP PER STATION   ####
##################################
# Execute this part of the script if a large csv file with multiple stations in it is used.
# Splitting the large file makes sure that later only the necessary parts have to be loaded.
"""
all_data_csv = Dir_csv + "water_level_signals.csv"
#all_data_csv = Dir_csv + "wave_height_signals.csv"
#all_data_csv = Dir_csv + "wave_period_signals.csv"

# For all unique station names in the file create a new csv file that contains the corresponding time series
all_data = pd.read_csv(all_data_csv, encoding = "ISO-8859-1", sep = ';')
all_data_locations = all_data['MEETPUNT_IDENTIFICATIE'].unique()
all_data_locs = all_data_locations.tolist()

for i,value in enumerate(all_data_locs):
    all_data[all_data['MEETPUNT_IDENTIFICATIE'] == value].to_csv(all_data_csv[:-11] + str(value) + '.csv',index = False, na_rep = 'N/A')
"""
#%%
##################################
####    CONVERT TIME SERIES   ####
##################################
wave_height_data_csv = Dir_csv + "wave_height_Europlatform.csv"
wave_period_data_csv = Dir_csv + "wave_period_Europlatform.csv"

water_level_station = ['TerschellingNoordzee','TexelNoordzee','DenHelder','IJmuidenbuitenhaven','Scheveningen','HoekvanHolland','BrouwershavenscheGat08','Stellendambuiten','Oosterschelde14','Oosterschelde04','Roompotbuiten','Westkapelle','Vlissingen','Breskens','Cadzand']
water_level_station_transect = [4001460, 60002051, 70000170, 8005700, 9010200, 90011800, 13000084, 13000300, 13001084, 13001706, 15000140, 16002195, 16003458, 17000071, 17001354]

# Load hourly offshore wave height (H0) time series from csv file
if os.path.exists(Dir_pickles + 'Wave_heigth_hourly.pickle'):
    print(Dir_pickles + 'Wave_heigth_hourly.pickle' + ' already exists')
else:
    wave_height_data = csv2hourly_data(wave_height_data_csv)
    wave_height_data.to_pickle(Dir_pickles + 'Wave_heigth_hourly.pickle')

# Load hourly offshore wave period (Tm02) time series from csv file
if os.path.exists(Dir_pickles + 'Wave_period_hourly.pickle'):
    print(Dir_pickles + 'Wave_period_hourly.pickle' + ' already exists')
else:
    wave_period_data = csv2hourly_data(wave_period_data_csv)
    wave_period_data.to_pickle(Dir_pickles + 'Wave_period_hourly.pickle')

# Load, convert and save water level measurements for all stations
for loc_name in water_level_station:
    WaterLevel_convert(Dir_pickles, Dir_csv, loc_name)


 
        

  
