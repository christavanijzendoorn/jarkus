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
Created on Tue Jan 21 12:29:16 2020

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
# Calculate and plot regression line 
def timeseries_regression(Series):
    mask = ~np.isnan(Series)
    dates = Series.index
    X = mdates.date2num(dates)[mask]
    labels = mdates.num2date(X)
    Y = Series.values[mask]
    a,b = np.polyfit(X, Y, 1)
    print('The increase of the TWL is ' + str(a*24*365.25*1000) + ' mm/yr')
    regression_line =a*X + b
    plt.plot(labels, Y, 'blue', X, regression_line, 'r--')
    return labels

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

Dir_csv = settings['Dir_csv']
Dir_env_cond = settings['Dir_C1']

#%%
##################################
####     Load wave data       ####
##################################

#### Filtering based on water level station ####
StationTransect = pd.read_excel("C:/Users/cijzendoornvan/Documents/Data/JARKUS/WaterLevelLocations.xlsx")

location_name = 'Scheveningen'
    
# Load TWL from pickle file
TWL = pickle.load(open(Dir_env_cond + 'TWL_' + location_name + '_hourly.pickle', 'rb'))

###################################
####         VISUALIZE         ####
###################################
  
labels = timeseries_regression(TWL)
plt.close()

mask = ~np.isnan(TWL)   
daily_avg = TWL[mask].resample('D').mean()
labels = timeseries_regression(daily_avg)

daily_avg_sorted = daily_avg.sort_values()
n = daily_avg_sorted.shape[0]

daily_max = TWL.resample('D').max()
labels = timeseries_regression(daily_max)
    

