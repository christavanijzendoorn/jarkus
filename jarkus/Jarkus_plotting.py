# -*- coding: utf-8 -*-
"""
Created on Thu Jul  18 15:54:23 2019

@author: cijzendoornvan
"""
from jarkus.transects import Transects
from visualisation import multilineplot
import numpy as np
## %matplotlib auto TO GET WINDOW FIGURE

# Collect the JARKUS data from the server
Jk = Transects(url='http://opendap.deltares.nl/thredds/dodsC/opendap/rijkswaterstaat/jarkus/profiles/transect_r20180914.nc')
ids = Jk.get_data('id')

# Set the transect and years for retrieval request
transect_name = "04_Terschelling"
transect = 8009325#np.arange(17000011, 17001487, 1)
years_requested = range(1960, 1970)

# Set x and y limit of plots - Leave lists empty (i.e. []) for automatic axis limits
xlimit = [-500, 1500] # EXAMPLE: [-400,1000]
ylimit = [-10, 25] # EXAMPLE: [-10,22]

# Set directory for saving images
Dir_plots = "C:/Users/cijzendoornvan/Documents/GitHub/jarkus/jarkus/Figures/" + transect_name.replace(" ","") + "/"

# Here the JARKUS filter is set and the data for each requested id and year are retrieved
idxs = np.isin(ids, transect)
for idx in np.nonzero(idxs)[0]:
    a, x_values, y_values, years_included = Jk.filter_jrk(idx,years_requested)
    
    # Here for each transect a multilineplot is made and saved in the predefined directory
    multilineplot(x_values, y_values, years_included, "Cross shore distance [m]", "Elevation [m to datum]" , "Transect {}".format(a[0][0]), xlimit, ylimit, Dir_plots)
    
##########################################################
# Reopening a pickled figure, for interactive editting
#%matplotlib auto
from visualisation import reopen_pickle
fig_transect = str(13001668)
Dir_fig = "C:\\Users\\cijzendoornvan\\Documents\\GitHub\\jarkus\\jarkus\\Figures\\13_Westenschouwen\\"

#reopen_pickle(Dir_fig, fig_transect)
##########################################################

"""
# Dummy data for plotting
# years = [1965, 1966, 1967]
# cross_shore = [np.array([-50, 0, 50, 100, 150]), np.array([-50, 0, 50, 100, 150]), np.array([-50, 0, 50, 100, 150])]
# elevation = [np.array([-5, 0, 4, 9, 15]), np.array([-3, 0, 3, 7, 12]), np.array([-4, 0, 4, 8, 14])]
""" 

