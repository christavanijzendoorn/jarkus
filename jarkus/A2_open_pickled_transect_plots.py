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
Created on Thu Oct 17 15:43:48 2019

@author: cijzendoornvan
"""

##################################
####          PACKAGES        ####
##################################
from visualisation import reopen_pickle
from IPython import get_ipython
# Execute %matplotlib auto first, otherwise you get an error
get_ipython().run_line_magic('matplotlib', 'auto')
import json
import pickle
import matplotlib.pyplot as plt

##################################
####    USER-DEFINED REQUEST  ####
##################################
# load basic settings from .txt file
with open("C:/Users/cijzendoornvan/Documents/GitHub/jarkus/jarkus/Settings.txt") as file:
    settings = json.load(file)

# Set the transect and years for retrieval request    
transect_name = "08_Meijendel"
transect_req = [8009325]
years_requested = range(2000, 2020, 1)

##################################
####  LOAD DIMENSIONS FILE    ####
##################################
Dir_pickles = settings['Dir_B']
trsct = str(transect_req[0])

pickle_file = Dir_pickles + 'Transect_' + trsct + '_dataframe.pickle'
Dimensions = pickle.load(open(pickle_file, 'rb'))

##################################
####  REOPEN PICKLED FIGURE   ####
##################################
fig_transect = str(transect_req[0])
Dir_fig = settings['Dir_A']

# Load figure from disk and display
fig = pickle.load(open(Dir_fig + 'Transect_' + fig_transect + '.fig.pickle','rb'))

# plt.plot(Dimensions['DF_pybeach_mix_x'], Dimensions['DF_pybeach_mix_y'], 'g^')
# plt.plot(Dimensions['DF_der_x'], Dimensions['DF_der_y'], 'bs')
# plt.plot(Dimensions['DF_fix_x'], Dimensions['DF_fix_y'], 'ro')
plt.axvline(Dimensions.loc[2018, 'Seaward_x_DoC'])

    
plt.show()



