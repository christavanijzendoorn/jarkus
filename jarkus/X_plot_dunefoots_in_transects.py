# -*- coding: utf-8 -*-
"""
Created on Thu Nov 21 09:31:38 2019

@author: cijzendoornvan
"""

# -*- coding: utf-8 -*-
"""
Created on Thu Jul  18 15:54:23 2019

@author: cijzendoornvan
"""

##################################
####          PACKAGES        ####
##################################
import json
import os
import pickle
from jarkus.transects import Transects
import numpy as np
from scipy.interpolate import griddata
from IPython import get_ipython
get_ipython().run_line_magic('matplotlib', 'auto')

#################################
####        FUNCTIONS        ####

def transects_with_DF(Dimensions, x_data, y_data, time, x_label="", y_label="", title="", xlim=[], ylim=[], plots_dir=""):

    import matplotlib.pyplot as plt
    import matplotlib.colors as colors
    import matplotlib.cm as cmx
    import pickle
   
    fig = plt.figure(figsize=(30,15))
    ax = fig.add_subplot(111)
    
    jet = plt.get_cmap('jet') 
    cNorm  = colors.Normalize(vmin=min(time), vmax=max(time))
    scalarMap = cmx.ScalarMappable(norm=cNorm, cmap=jet)
    
    lines = []
    for i, yr in enumerate(time):
        x = x_data
        y = y_data[:,i]
        colorVal = scalarMap.to_rgba(yr)
        colorText = (
                '%i'%(yr)
                )
        retLine, = ax.plot(x, y,
                           color=colorVal,
                           label=colorText)
        lines.append(retLine)
        
        plt.scatter(Dimensions.loc[str(yr), 'DF_der_x'], Dimensions.loc[str(yr), 'DF_der_y'], marker = 'o')
        plt.scatter(Dimensions.loc[str(yr), 'DF_fix_x'], Dimensions.loc[str(yr), 'DF_fix_y'], marker = 'x')
        
    #added this to get the legend to work
    handles,labels = ax.get_legend_handles_labels()
    ax.legend(handles, labels, loc='upper left',ncol=2, fontsize = 18)
    
    # Label the axes and provide a title
    ax.set_title(title, fontsize = 24)
    ax.set_xlabel(x_label, fontsize = 20)
    ax.set_ylabel(y_label, fontsize = 20)
    if len(xlim) != 0:
        ax.set_xlim(xlim)
    if len(ylim) != 0:
        ax.set_ylim(ylim)
    ax.grid()
    ax.invert_xaxis()
    

    # Show the figure    
    #plt.show()
    
    # Save figure as png in predefined directory
    #plt.savefig(plots_dir + 'Transect_' + title[9:] + '.png')
    #pickle.dump(fig, open(plots_dir + 'Transect_' + title[9:] + '.fig.pickle', 'wb'))
    
    #plt.close()


##################################
####       RETRIEVE DATA      ####
##################################
# load basic settings from .txt file
with open("C:/Users/cijzendoornvan/Documents/GitHub/jarkus/jarkus/Settings.txt") as file:
    settings = json.load(file)

# Collect the JARKUS data from the server
Jk = Transects(url= settings['url'])
ids = Jk.get_data('id')

##################################
####  GET DF from dataframes  ####
##################################
Dir_pickles = settings['Dir_X']


##################################
####    USER-DEFINED REQUEST  ####
##################################
# Set the transect and years for retrieval request
transect_name = "06_Texel"
transect_req = np.arange(6002521, 6003100, 1)
years_requested = list(range(1965, 2020))

# Set whether all transect should be analysed or define a retrieval request
execute_all_transects = False

# Set which years should be analysed
years_requested = list(range(1965, 2020))

if execute_all_transects == False:
    # Set the transect and years for retrieval request
    transect_name   = "08_Meijendel"
    #transect_req = np.arange(8009325, 8009750, 1)
    transect_req        = [8009325]
    idxs = np.isin(transect_req, ids) # check which transect are available of those that were requested
    ids_filtered = np.array(transect_req)[np.nonzero(idxs)[0]]
    Dir_plots = settings['Dir_figures'] + transect_name.replace(" ","") + "/"
else:
    # Filter out location that are not suitable for analysis. Based on Kustlijnkaarten 2019, RWS.
    remove1 = list(range(2000303,2000304))
    remove2 = list(range(2000720,2000721))
    remove3 = list(range(2002019,2002020))
    remove4 = list(range(4002001,4005917,1))
    remove5 = list(range(5003300,5004000,1))
    remove6 = list(range(6000416,6000900,1))
    remove7 = list(range(6003070,6003453,1))
    remove8 = list(range(8000000,8005626,1))
    remove9 = list(range(9010193,9010194,1))
    remove10 = list(range(10000000,10001411,1))
    remove11 = list(range(11000660,11000661,1))
    remove12 = list(range(11000680,11000681,1))
    remove13 = list(range(11000700,11000841,1))
    remove14 = list(range(14000000,14000700,1))
    
    remove_all = remove1 + remove2 + remove3 + remove4 + remove5 + remove6 + remove7 + remove8 + remove9 + remove10 + remove11 + remove12 + remove13 + remove14
    
    ids_filtered = [x for x in ids if x not in remove_all]
    Dir_plots = settings['Dir_A']

##################################
####   CREATE AND SAVE PLOTS  ####
##################################
# Set x and y limit of plots - Leave lists empty (i.e. []) for automatic axis limits
xlimit = [-200,500] # EXAMPLE: [-400,1000]
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
    
    filename = 'Transect_' + trsct + '_dataframe.pickle'
    pickle_file = os.path.join(Dir_pickles, filename)
    Dimensions = pickle.load(open(pickle_file, 'rb')) #load pickle of transect
    
    # Here for each transect a multilineplot is made and saved in the predefined directory
    transects_with_DF(Dimensions, cross_shore, y_all, years_requested, "Cross shore distance [m]", "Elevation [m to datum]" , "Transect {}".format(str(idx)), xlimit, ylimit, Dir_plots)