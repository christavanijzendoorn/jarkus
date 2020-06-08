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
transect_req = [8009575]
years_requested = range(2000, 2020, 1)

##################################
####  LOAD DIMENSIONS FILE    ####
##################################
Dir_pickles = settings['Dir_C3']
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

plt.plot(Dimensions['DF_pybeach_mix_x'], Dimensions['DF_pybeach_mix_y'], 'g^')
plt.plot(Dimensions['DF_der_x'], Dimensions['DF_der_y'], 'bs')
plt.plot(Dimensions['DF_fix_x'], Dimensions['DF_fix_y'], 'ro')

    
plt.show()

