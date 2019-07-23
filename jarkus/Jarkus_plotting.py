# -*- coding: utf-8 -*-
"""
Created on Thu Jul  18 15:54:23 2019

@author: cijzendoornvan
"""

from jarkus.transects import Transects
import numpy as np
## %matplotlib auto TO GET WINDOW FIGURE

# Collect the JARKUS data from the server
Jk = Transects(url='http://opendap.deltares.nl/thredds/dodsC/opendap/rijkswaterstaat/jarkus/profiles/transect_r20180914.nc')
ids = Jk.get_data('id')

# Set the transect and years for retrieval request
"""
EXAMPLES of TRANSECTS
Zeeuws-Vlaandere = 17000011 - 17001487
Vrouwenpolder - Vlissingen = 16000540 - 16003750
Noord-Beveland = 15000000 - 15000520
Westenschouwen = 13001196 - 13001548 - 13001800
Schouwen-Duiveland = 13000074 - 13001800
Goeree-Overflakkee = 12000280 - 12002525
Voorne-Putten = 11000400 - 11001830
Zandmotor = 9010592 - 9010883 - 90011221
Den Haag - Hoek van Holland = 9009740 - 9011850
IJmuiden - Scheveningen = 8005625 - 8009725
Meijendel = 8009300 - 8009750
Schoorlse Duinen - Velsen = 7002629 - 7005500
Hondsbossche = 7002041 - 7002606
Den Helder - Petten = 7000000 - 7002023

Texel = 6000416 - 6003452
Vlieland = 5003300 - 5005480
Terschelling = 4000000 - 4005916
Ameland = 3000100 - 3004966
Schier = 2000101 - 2002019
"""
transect_name = "Terschelling"
transect = np.arange(4000000, 4005916, 1)
years_requested = range(1965, 2020)

# Set x and y limit of plots - Leave lists empty for automatic axis limits
xlimit = [-500, 1500] # EXAMPLE: [-400,1000]
ylimit = [-10, 25] # EXAMPLE: [-10,22]

# Set directory for saving images
Dir_plots = "C:/Users/cijzendoornvan/Documents/GitHub/jarkus/jarkus/Figures/Terschelling/" 

# Set the JARKUS filter and retrieve the data for each requested id and year
a = []
idxs = np.isin(ids, transect)
for idx in np.nonzero(idxs)[0]:
    Jk.set_filter(alongshore=idx, year=years_requested)
    a = Jk.get_jrk()

    # Convert retrieved data for easy visualisation
    x_values = []
    y_values = []  
    years_included = []
    for i in range(len(a)):
        years_included.append(a[i][1])
        
        x = np.array([i[0] for i in a[i][2]])
        x[x == 99999] = np.nan # Convert 99999 to nan values, so they are not included in visualisation
        x_values.append(x)
        
        y = np.array([i[1] for i in a[i][2]])
        y[y == 99999] = np.nan # Convert 99999 to nan values, so they are not included in visualisation
        y_values.append(y)
    
    # Find difference between years included and those requested
    years_missing = list(set(years_requested) - set(years_included))
    # If all years were available do nothing,
    if len(years_missing) == 0:
        print("All requested years were available")
        # If years were missing, show error
    else: 
        print("ERROR - For transect {} the following year(s) were not available:".format(a[0][0]))
        print(years_missing)
    
    from visualisation import multilineplot
    cross_shore = x_values
    elevation = y_values
    multilineplot(cross_shore, elevation, years_included, "Cross shore distance [m]", "Elevation [m to datum]" , "Transect {}".format(a[0][0]), xlimit, ylimit, Dir_plots)
    
##########################################################
# Reopening a pickled figure, for interactive editting
from visualisation import reopen_pickle
reopen_pickle("Transect {}".format(a[0][0]), Dir_plots)
##########################################################

"""
# Dummy data for plotting
# years = [1965, 1966, 1967]
# cross_shore = [np.array([-50, 0, 50, 100, 150]), np.array([-50, 0, 50, 100, 150]), np.array([-50, 0, 50, 100, 150])]
# elevation = [np.array([-5, 0, 4, 9, 15]), np.array([-3, 0, 3, 7, 12]), np.array([-4, 0, 4, 8, 14])]
""" 