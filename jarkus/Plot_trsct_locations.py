# -*- coding: utf-8 -*-
"""
Created on Mon Jul 13 18:21:47 2020

@author: cijzendoornvan
"""

import xarray as xr
import matplotlib.pyplot as plt
import json
import Jarkus_Analysis_Toolbox as TB
import math
import simplekml
import pandas as pd

# load basic settings from .txt file
with open("C:/Users/cijzendoornvan/Documents/GitHub/jarkus/jarkus/Settings.txt") as file:
    settings = json.load(file)                                                  # include USER-DEFINED settings
    
# load basic settings from .txt file
with open("C:/Users/cijzendoornvan/Documents/GitHub/jarkus/jarkus/Filter2.txt") as ffile:
    filter_file = json.load(ffile)   

DirJarkus = settings['Dir_Jarkus']
DirDFAnalysis = r"C:\Users\cijzendoornvan\Documents\DuneForce\JARKUS\ANALYSIS\DF_analysis\\"    
DirDF = r"C:\Users\cijzendoornvan\Documents\DuneForce\JARKUS\Comparison_methods\Derivative_Diamantidou\DF_2nd_deriv.nc"
DirDimensions = settings['Dir_D1']

######################
# USER-DEFINED REQUEST
######################
start_yr = 1965                                                                # USER-DEFINED request for years
end_yr = 2020

######################
# LOAD DATA
######################
# Load jarkus dataset
dataset, variables = TB.get_jarkus_data(DirJarkus)

# Filter for locations that are not suitable for analysis. Based on Kustlijnkaarten 2019, RWS.
with open("C:/Users/cijzendoornvan/Documents/GitHub/jarkus/jarkus/Filter.txt") as file:
        filter_transects = json.load(file)
        
# Load trsct locations
trscts = dataset['id'].values    
lat = dataset['lat'].values # x position of profile point in WGS84 projected coordinates
lon = dataset['lon'].values

kml=simplekml.Kml()

for i, idx in enumerate(trscts):
  kml.newlinestring(name=str(idx), coords=[(lon[i, 0],lat[i,0]),(lon[i,-1],lat[i,-1])])

kml.save(DirDFAnalysis + 'transects.kml')

#%%
Nourishments = pd.read_excel("C:/Users/cijzendoornvan/Documents/Duneforce/JARKUS/Suppletiedatabase.xlsx")
filtered = []
for index, row in Nourishments.iterrows():
    if math.isnan(row['BeginRaai']) or math.isnan(row['EindRaai']):# or row['Volume/m'] > 50: # or row['JaarBeginUitvoering'] < 2010: 
        continue
    else:
        code_beginraai = int(row['KustVakNummer'] * 1000000 + row['BeginRaai'] * 100)
        code_eindraai = int(row['KustVakNummer'] * 1000000 + row['EindRaai'] * 100)
        nourished_transects = [i for i in trscts if i >= code_beginraai and i <= code_eindraai]
        filtered.extend(nourished_transects)
filtered = set(filtered)

# Filter dataframe
remove_transects = []
for i, trsct in enumerate(not_nourished_transects):
    for key in filter_file.keys():
        if trsct >= int(filter_file[key]['begin']) and trsct <= int(filter_file[key]['eind']):
            remove_transects.append(trsct)
            
not_nourished_transects = [i for i in trscts if i not in filtered and i not in remove_transects]
not_nourished_idx = [list(trscts).index(i) for i in not_nourished_transects]

kml_not_nourished=simplekml.Kml()

for i, idx in enumerate(not_nourished_transects):
  kml_not_nourished.newlinestring(name=str(idx), coords=[(lon[not_nourished_idx[i], 0],lat[not_nourished_idx[i],0]),(lon[not_nourished_idx[i],-1],lat[not_nourished_idx[i],-1])])

kml_not_nourished.save(DirDFAnalysis + 'not_nourished_transects.kml')

