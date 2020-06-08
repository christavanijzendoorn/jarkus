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
Created on Mon Oct 21 09:49:55 2019

@author: cijzendoornvan
"""

# This script was created to plot the development of several dimensions through time in one figure. This is done for 1 chosen transect and a user requested amount of years.

##################################
####         PACKAGES         ####
##################################
import json
from jarkus.transects import Transects
import pickle
import matplotlib.pyplot as plt
from IPython import get_ipython
get_ipython().run_line_magic('matplotlib', 'auto') ## %matplotlib auto TO GET WINDOW FIGURE

##################################
####      INITIALISATION      ####
##################################
# load basic settings from .txt file
with open("C:/Users/cijzendoornvan/Documents/GitHub/jarkus/jarkus/Settings.txt") as file:
    settings = json.load(file)
    
Dir_per_transect = settings['Dir_B']
Dir_plots_per_transect = settings['Dir_B2']

years_requested = list(range(1965, 2020))
years_requested_str = [str(yr) for yr in years_requested]

# Collect the JARKUS data from the server
Jk = Transects(url= settings['url'])
ids = Jk.get_data('id') # ids of all available transects

with open("C:/Users/cijzendoornvan/Documents/GitHub/jarkus/jarkus/Filter.txt") as file:
    filter_transects = json.load(file)

filter_all = []
for key in filter_transects:
    filter_all += list(range(int(filter_transects[key]["begin"]), int(filter_transects[key]["eind"])))

ids_filtered = [x for x in ids if x not in filter_all]
        
###################################
####         VISUALIZE         ####
###################################
for idx in ids_filtered: # For each available transect visualize the dimensions
    trsct = str(idx)
    pickle_file = Dir_per_transect + 'Transect_' + trsct + '_dataframe.pickle'
    Dimensions = pickle.load(open(pickle_file, 'rb')) #load pickle of transect
    
    Dimensions_plot = Dimensions.astype(float) # To make sure all values are converted to float, otherwise error during plotting
    
    fig, axs = plt.subplots(nrows=2, ncols=3, figsize=(30,15)) #sharex = True
    fig.suptitle('Dimensions of Jarkus transect ' + trsct)
    
    # Plotting the seaward boundary
    keys_plot1 = ['MLW_x_fix', 'MLW_x_var', 'MSL_x', 'MHW_x_fix', 'MHW_x_var', 'DF_fix_x', 'DF_der_x']
    colors_plot1 = ['black', 'black', 'blue', 'red', 'red', 'orange', 'orange']
    linest_plot1 = ['-','--','-','-','--', '-', '--']
    
    for i in range(len(keys_plot1)):
        Dimensions_plot.reset_index().plot(kind='line', x = 'years', y = keys_plot1[i], color = colors_plot1[i], linestyle = linest_plot1[i], marker = 'o', ax=axs[0,0])
    
    axs[0,0].set(xlabel='time (yr)', ylabel='cross shore distance (m)',
       title='Cross shore movement of MLW, MSL, MHW and DF')
    #axs[0,0].set_xlim([1965, 2020])
    #axs[0,0].set_ylim([0, 300])
    axs[0,0].grid()
    
    # Plotting beach width and intertidal width
    keys_plot2 = ['W_intertidal_fix', 'W_intertidal_var', 'BW_fix', 'BW_var', 'BW_der']
    # Variation due to definition of the MHW and MSL (fixed vs jarkus data)
    # Variation due to definition of dune foot (fixed vs derivative) and the water line (MSL vs MLW-MHW-combo based on the jarkus data)
    colors_plot2 = ['green', 'green', 'cyan', 'cyan', 'cyan']
    linest_plot2 = ['-', '--', '-', '--', 'dotted']
        
    for i in range(len(keys_plot2)):
        Dimensions_plot.reset_index().plot(kind='line', x = 'years', y = keys_plot2[i], color = colors_plot2[i], linestyle = linest_plot2[i], marker = 'o', ax=axs[1,0])
    
    axs[1,0].set(xlabel='time (yr)', ylabel='width (m)',
       title='Width of the intertidal area and beach')
    #axs[1,0].set_xlim([1965, 2020])
    #axs[1,0].set_ylim([0, 200])
    axs[1,0].grid()
    
    # Plotting beach gradient
    keys_plot3 = ['B_grad_fix']#, 'B_grad_var', 'B_grad_der']
    # Variation due to definition of the dune foot (fixed vs derivative) and the water line (MSL vs MLW-MHW-combo)
    colors_plot3 = ['darkblue', 'darkblue', 'darkblue']
    linest_plot3 = ['-', '--', 'dotted']
    
    for i in range(len(keys_plot3)):
        Dimensions_plot.reset_index().plot(kind='line', x = 'years', y = keys_plot3[i], color = colors_plot3[i], linestyle = linest_plot3[i], marker = 'o', ax=axs[1,2])
    
    axs[1,2].set(xlabel='time (yr)', ylabel='slope (m/m)',
       title='Beach gradient')
    #axs[1,2].set_xlim([1965, 2020])
    #axs[1,2].set_ylim([-0.05, -0.01])
    axs[1,2].grid()
    
    # Plotting dune gradient (both primary and secondary)
    #keys_plot4 = ['DFront_fix_prim_grad', 'DFront_der_prim_grad', 'DFront_fix_sec_grad', 'DFront_der_sec_grad']
    keys_plot4 = ['DFront_fix_prim_grad', 'DFront_fix_sec_grad']
    # Variation due to definition of the dune foot (fixed vs derivative)
    #colors_plot4 = ['purple', 'purple', 'pink', 'pink']
    colors_plot4 = ['purple', 'pink']
    linest_plot4 = ['-', 'dotted', '-', 'dotted']
    
    for i in range(len(keys_plot4)):
        Dimensions_plot.reset_index().plot(kind='line', x = 'years', y = keys_plot4[i], color = colors_plot4[i], linestyle = linest_plot4[i], marker = 'o', ax=axs[1,1])
    
    axs[1,1].set(xlabel='time (yr)', ylabel='slope (m/m)',
       title='Dune front gradient')
    #axs[1,1].set_xlim([1965, 2020])
    #axs[1,1].set_ylim([-0.6, -0.1])
    axs[1,1].grid()
    
    # Plotting dune height (both primary and secondary)
    keys_plot5 = ['DT_prim_y', 'DT_sec_y']
    colors_plot5 = ['grey', 'red']
    linest_plot5 = ['-', '-']
    
    for i in range(len(keys_plot5)):
        Dimensions_plot.reset_index().plot(kind='line', x = 'years', y = keys_plot5[i], color = colors_plot5[i], linestyle = linest_plot5[i], marker = 'o', ax=axs[0,2])
    
    axs[0,2].set(xlabel='time (yr)', ylabel='elevation (m)',
       title='Dune height')
    #axs[0,2].set_xlim([1965, 2020])
    #axs[0,2].set_ylim([5, 25])
    axs[0,2].grid()
    
    # Plotting dune volume
    keys_plot6 = ['DVol_fix', 'DVol_der']
    # Variation due to definition of the dune foot (fixed vs derivative)
    colors_plot6 = ['brown', 'brown']
    linest_plot6 = ['-', 'dotted']
    
    for i in range(len(keys_plot6)):
        Dimensions_plot.reset_index().plot(kind='line', x = 'years', y = keys_plot6[i], color = colors_plot6[i], linestyle = linest_plot6[i], marker = 'o', ax=axs[0,1])
    
    axs[0,1].set(xlabel='time (yr)', ylabel='volume (m^3/m)',
       title='Dune volume')
    #axs[0,1].set_xlim([1965, 2020])
    #axs[0,1].set_ylim([0, 1000])
    axs[0,1].grid()
    
    plt.savefig(Dir_plots_per_transect + 'Transect_' + trsct + '_dimensions.png')
    pickle.dump(fig, open(Dir_plots_per_transect + 'Transect_' + trsct + '_dimensions.fig.pickle', 'wb'))
    #plt.show()
    plt.close()



#######################################################
#### Plot dune height and dune gradient seperately ####
#######################################################
"""
fig2, axs2 = plt.subplots(nrows=1, ncols=2) #sharex = True
fig2.suptitle('Dune dimensions of Jarkus transect '+ str(transect[0]))

# Plotting dune gradient (both primary and secondary)
keys_plot4 = ['DFront_der_prim_grad', 'DFront_der_sec_grad']
# Variation due to definition of the dune foot (fixed vs derivative)
colors_plot4 = ['black', 'red']
linest_plot4 = ['-','-']
labels4 = ['Primary','Secondary']

for i in range(len(keys_plot4)):
    Dimensions_plot.plot(kind='line', x = 'years', y = keys_plot4[i], color = colors_plot4[i], linestyle = linest_plot4[i], marker = 'o', ax=axs2[1], label = labels4[i])

axs2[1].set(xlabel='time (yr)', ylabel='slope (m/m)')
axs2[1].set_title('Dune front gradient', fontsize=18)
axs2[1].set_xlim([1965, 2020])
#axs2[1].set_ylim([-0.6, -0.1])
axs2[1].grid()

# Plotting dune height (both primary and secondary)
keys_plot5 = ['DT_prim_y', 'DT_sec_y']
colors_plot5 = ['black', 'red']
linest_plot5 = ['-', '-']
labels5 = ['Primary','Secondary']

for i in range(len(keys_plot5)):
    Dimensions_plot.plot(kind='line', x = 'years', y = keys_plot5[i], color = colors_plot5[i], linestyle = linest_plot5[i], marker = 'o', ax=axs2[0], label = labels5[i])

axs2[0].set(xlabel='time (yr)', ylabel='elevation (m)')
axs2[0].set_title('Dune height', fontsize=18)
axs2[0].set_xlim([1965, 2020])
#axs2[0].set_ylim([8, 22])
axs2[0].grid()

plt.show()
"""