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
Created on Thu Jul  18 15:54:23 2019

@author: cijzendoornvan
"""

##################################
####          PACKAGES        ####
##################################
import json
from jarkus.transects import Transects
from visualisation import multilineplot
import numpy as np
from scipy.interpolate import griddata
from IPython import get_ipython
import matplotlib.pyplot as plt
import pickle
get_ipython().run_line_magic('matplotlib', 'auto')

##################################
####       RETRIEVE DATA      ####
##################################
# load basic settings from .txt file
with open("C:/Users/cijzendoornvan/Documents/GitHub/jarkus/jarkus/Settings.txt") as file:
    settings = json.load(file)

# Collect the JARKUS data from the server/local file
Jk = Transects(url= settings['url'])
ids = Jk.get_data('id')

##################################
####    USER-DEFINED REQUEST  ####
##################################
# Set whether all transect should be analysed or define a retrieval request
execute_all_transects = False

# Set which years should be analysed
years_requested = list(range(1980, 1982))

if execute_all_transects == False:
    # Set the transect and years for retrieval request
    transect_name   = "08_Meijendel"
    #transect_req = np.arange(8009325, 8009750, 1)
    transect_req        = [8006300]
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
        
    # Use this if you want to skip transects, e.g. if your battery died during running...
    #skip_transects = list(ids[1:109])
    #filter_all = filter_all + skip_transects
        
    ids_filtered = [x for x in ids if x not in filter_all]
    Dir_plots = settings['Dir_A']

##################################
####   CREATE AND SAVE PLOTS  ####
##################################
# Set x and y limit of plots - Leave lists empty (i.e. []) for automatic axis limits
xlimit = [-2500,150] # EXAMPLE: [-400,1000]
ylimit = [-10, 25] # EXAMPLE: [-10,22]

years_requested_str = [str(yr) for yr in years_requested]

# Interpolate x and y along standardized cross shore axis
cross_shore = list(range(-3000, 9320, 1))

# Here the JARKUS filter is set and the data for each requested id and year are retrieved
for idx in ids_filtered:
    print(idx)
    trsct = str(idx)

    Jk = Transects(url= settings['url'])
    df, years_available = Jk.get_dataframe(idx, years_requested)
    
    # Convert elevation data for each year of the transect into array that can easily be plotted
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
            if i == 0:
                y_all = y_grid
            else:
                y_all = np.column_stack((y_all, y_grid))
    
    # Here for each transect a multilineplot is made and saved in the predefined directory
    title =  "Transect {}".format(str(idx))
    fig = multilineplot(cross_shore, y_all, years_requested, title, "Cross shore distance [m]", "Elevation [m to datum]", xlimit, ylimit)
    
    # Show the figure    
    #plt.show()
    
    # Save figure as png in predefined directory
    #plt.savefig(Dir_plots + 'Transect_' + title[9:] + '.png')
    #pickle.dump(fig, open(Dir_plots + 'Transect_' + title[9:] + '.fig.pickle', 'wb'))
    
    #plt.close()
    

"""
# Dummy data for plotting
# years = [1965, 1966, 1967]
# cross_shore = [np.array([-50, 0, 50, 100, 150]), np.array([-50, 0, 50, 100, 150]), np.array([-50, 0, 50, 100, 150])]
# elevation = [np.array([-5, 0, 4, 9, 15]), np.array([-3, 0, 3, 7, 12]), np.array([-4, 0, 4, 8, 14])]
""" 

