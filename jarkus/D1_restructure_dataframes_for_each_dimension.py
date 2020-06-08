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
Created on Mon Jul 29 11:20:42 2019

@author: cijzendoornvan
"""
##################################
####          PACKAGES        ####
##################################
import json
import pandas as pd
import numpy as np
import os.path
from jarkus.transects import Transects
import pickle
from IPython import get_ipython
get_ipython().run_line_magic('matplotlib', 'auto') ## %matplotlib auto TO GET WINDOW FIGURE

#################################
####        FUNCTIONS        ####
#################################

def extract_variable(variable, dir_per_trsct, dir_per_var, transects, years):
    
    Variable_dataframe = pd.DataFrame({'years': years})
    Variable_dataframe.set_index('years', inplace=True)
    for idx in transects:
        trsct = str(idx)
        pickle_file = dir_per_trsct + 'Transect_' + trsct + '_dataframe.pickle'
        Variable_dataframe[trsct] = np.nan
        
        if os.path.exists(pickle_file):
            Dimensions = pickle.load(open(pickle_file, 'rb')) #load pickle of transect
            
            for i, yr in enumerate(years):  
                if str(yr) in Dimensions.index:
                    Variable_dataframe.loc[yr, trsct] = Dimensions.loc[str(yr), variable] #extract column that corresponds to the requested variable
                    
            print('Extracted transect ' + trsct + ' for variable ' + variable)
        else:
            pass
    Variable_dataframe.to_pickle(dir_per_var + variable + '_dataframe' + '.pickle')
    print(Variable_dataframe)
    print('The dataframe of ' + variable + ' was saved')
    
    return Variable_dataframe

#%%
##################################
####    USER-DEFINED REQUEST  ####
##################################
# load basic settings from .txt file
with open("C:/Users/cijzendoornvan/Documents/GitHub/jarkus/jarkus/Settings.txt") as file:
    settings = json.load(file)

Dir_per_transect = settings['Dir_C3'] #C3 is standard, B if one extra dimension was calculated.
Dir_per_variable = settings['Dir_D1']

years_requested = list(range(1965, 2020))
years_requested_str = [str(yr) for yr in years_requested]
#%%
##################################
####      INITIALISATION      ####
##################################
# Collect the JARKUS data from the server
Jk = Transects(url= settings['url'])
ids = Jk.get_data('id') # ids of all available transects

with open("C:/Users/cijzendoornvan/Documents/GitHub/jarkus/jarkus/Filter.txt") as file:
    filter_transects = json.load(file)

filter_all = []
for key in filter_transects:
    filter_all += list(range(int(filter_transects[key]["begin"]), int(filter_transects[key]["eind"])))

ids_filtered = [x for x in ids if x not in filter_all]
#%%
##################################
####    Transect to Variable  ####
##################################

# Get all variables from the columns in the dataframe with all the dimensions
trsct = str(ids_filtered[0])
pickle_file = Dir_per_transect + 'Transect_' + trsct + '_dataframe.pickle'
Dimensions_test = pickle.load(open(pickle_file, 'rb')) #load pickle of dimensions    
variables = list(Dimensions_test.columns)

variables = ['DF_pybeach_mix_y', 'DF_pybeach_mix_x'] 

for i in range(len(variables)):    
    Variable_DF = extract_variable(variables[i], Dir_per_transect, Dir_per_variable, ids_filtered, years_requested)

#%%
##################################
#### VARIABLE NORMALISATION  ####
##################################

def get_normalisation_value(variable, year, dir_per_dimension): # Get norm values for the cross-shore location for each transect in the norm year
    
    pickle_file = dir_per_dimension + variable + '_dataframe.pickle'
    Dimension_df = pickle.load(open(pickle_file, 'rb')) #load pickle of dimensions    
    Normalisation_df = Dimension_df.loc[[year]]

    return Normalisation_df

norm_variable = 'MSL_x'
norm_year = 1999

# Get all variables that have to be normalized based on the requirement that _x should be in the column name, 
#and that change values do not have to be normalized.
normalized_variables = [var for var in variables if '_x' in var and 'change' not in var]

for i, var in enumerate(normalized_variables):        
    Normalisation_values = get_normalisation_value(norm_variable, norm_year, Dir_per_variable)
        
    pickle_file = Dir_per_variable + var + '_dataframe.pickle'
    Dimension_df = pickle.load(open(pickle_file, 'rb')) #load pickle of dimensions    
    Normalized_df = Dimension_df.copy()
    
    for key in Normalized_df.columns:
        Normalized_df[key] = Normalized_df[key] - Normalisation_values.loc[norm_year, key]
    
    Normalized_df.to_pickle(Dir_per_variable + var + '_normalized_dataframe' + '.pickle')
    print('The dataframe of ' + var + ' was normalized and saved')
