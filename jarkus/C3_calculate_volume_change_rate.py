# -*- coding: utf-8 -*-
"""
Created on Thu Nov 14 16:37:48 2019

@author: cijzendoornvan
"""

##################################
####          PACKAGES        ####
##################################
import json
import pandas as pd
import numpy as np
import os
import pickle
from IPython import get_ipython
get_ipython().run_line_magic('matplotlib', 'auto') ## %matplotlib auto TO GET WINDOW FIGURE

##################################
####      Initialisation      ####
##################################

##################################
####      Initialisation      ####
##################################
# load basic settings from .txt file
with open("C:/Users/cijzendoornvan/Documents/GitHub/jarkus/jarkus/Settings.txt") as file:
    settings = json.load(file)

Dir_pickles = settings['Dir_C2']
Dir_dataframes = settings['Dir_C3']

##################################
####   GET MEASUREMENT DATES  ####
##################################
# Get the dates that correspond to the collection of each yearly measurment per transect    
directory_old = Dir_pickles
directory_new = Dir_dataframes

for filename in os.listdir(Dir_pickles):
    if filename.endswith("Transect_12000775_dataframe.pickle"): 
         pickle_file = os.path.join(directory_old, filename)
         Dimensions = pickle.load(open(pickle_file, 'rb')) #load pickle of transect
         # Calculate volume changes
         Dimensions['DVol_der_rate'] = Dimensions['DVol_der'].shift(-1) - Dimensions['DVol_der']
         Dimensions['DVol_fix_rate'] = Dimensions['DVol_fix'].shift(-1) - Dimensions['DVol_fix'] 
         Dimensions['APVol_fix_rate'] = Dimensions['APVol_fix'].shift(-1) - Dimensions['APVol_fix']
         Dimensions['APVol_var_rate'] = Dimensions['APVol_var'].shift(-1) - Dimensions['APVol_var'] 
         Dimensions['FSVol_fix_rate'] = Dimensions['FSVol_fix'].shift(-1) - Dimensions['FSVol_fix']
         Dimensions['IntVol_fix_rate'] = Dimensions['IntVol_fix'].shift(-1) - Dimensions['IntVol_fix']
         Dimensions['IntVol_var_rate'] = Dimensions['IntVol_var'].shift(-1) - Dimensions['IntVol_var']
         # Calculate changes in dune foot location
         Dimensions['DF_fix_x_change'] = Dimensions['DF_fix_x'].shift(-1) - Dimensions['DF_fix_x']
         Dimensions['DF_fix_y_change'] = Dimensions['DF_fix_y'].shift(-1) - Dimensions['DF_fix_y']
         Dimensions['DF_der_x_change'] = Dimensions['DF_der_x'].shift(-1) - Dimensions['DF_der_x']
         Dimensions['DF_der_y_change'] = Dimensions['DF_der_y'].shift(-1) - Dimensions['DF_der_y']
         # Calculate MSL based on variable MLW and MHW
         Dimensions['MSL_y_var'] = (Dimensions['MLW_y_var'] + Dimensions['MHW_y_var'])/2
         
         pickle_file_new = os.path.join(directory_new, filename)
         Dimensions.to_pickle(pickle_file_new)

