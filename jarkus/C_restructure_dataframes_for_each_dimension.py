# -*- coding: utf-8 -*-
"""
Created on Mon Jul 29 11:20:42 2019

@author: cijzendoornvan
"""
##################################
####          PACKAGES        ####
##################################
import pandas as pd
import numpy as np
import os.path
from jarkus.transects import Transects
import pickle

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
        
Dir_per_transect = "C:/Users/cijzendoornvan/Documents/GitHub/jarkus/jarkus/Figures/Dataframes_per_transect/"
Dir_per_variable = "C:/Users/cijzendoornvan/Documents/GitHub/jarkus/jarkus/Figures/Dataframes_per_dimension/"

years_requested = list(range(1965, 2020))
years_requested_str = [str(yr) for yr in years_requested]
#%%
##################################
####      INITIALISATION      ####
##################################

# Collect the JARKUS data from the server
#Jk = Transects(url='http://opendap.deltares.nl/thredds/catalog/opendap/rijkswaterstaat/jarkus/profiles/catalog.html?dataset=varopendap/rijkswaterstaat/jarkus/profiles/transect_r20190731.nc')
Jk = Transects(url='https://opendap.tudelft.nl/thredds/dodsC/data2/deltares/rijkswaterstaat/jarkus/profiles/transect_r20180914.nc')

ids = Jk.get_data('id') # ids of all available transects

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
#%%
##################################
####    Transect to Variable  ####
##################################

variables = ['DT_prim_x', 'DT_prim_y',
 'DT_sec_x','DT_sec_y',
 'MSL_x',
 'MLW_x_fix', 'MLW_x_var', 'MLW_y_var',
 'MHW_x_fix', 'MHW_x_var', 'MHW_y_var',
 'W_intertidal_fix', 'W_intertidal_var',
 'landward_stable_x', 'landward_6m_x', 
 'Bma_x',
 'seaward_FS_x_all', 'seaward_ActProf_x_all',
 'DF_fix_y', 'DF_fix_x',
 'DF_der_y', 'DF_der_x',
 'BW_fix', 'BW_var', 'BW_der', 
 'B_grad_fix',
 'DFront_fix_prim_W', 'DFront_der_prim_W',
 'DFront_fix_sec_W', 'DFront_der_sec_W',
 'DFront_fix_prim_grad', 'DFront_fix_sec_grad',
 'DVol_fix', 'DVol_der',
 'Int_grad', 'IntVol_fix', 'IntVol_var',
 'FS_grad', 'FSVol_fix',
 'AP_grad', 'APVol_fix', 'APVol_var']


for i in range(len(variables)):    
    Variable_DF = extract_variable(variables[i], Dir_per_transect, Dir_per_variable, ids_filtered, years_requested)

#%%
##################################
#### VARIABLE NORMALLISATION  ####
##################################

def get_normalisation_value(variable, year, dir_per_dimension): # Get norm values for the cross-shore location for each transect in the norm year
    
    pickle_file = dir_per_dimension + variable + '_dataframe.pickle'
    Dimension_df = pickle.load(open(pickle_file, 'rb')) #load pickle of dimensions    
    Normalisation_df = Dimension_df.loc[[year]]

    return Normalisation_df

norm_variable = 'MSL_x'
norm_year = 2000

Dir_per_variable = "C:/Users/cijzendoornvan/Documents/GitHub/jarkus/jarkus/Figures/Dataframes_per_dimension/"

normalized_variables = ['DT_prim_x', 'DT_sec_x',
 'MSL_x', 'MLW_x_fix', 'MLW_x_var',
 'MHW_x_fix', 'MHW_x_var',
 'landward_stable_x', 'landward_6m_x', 
 'Bma_x', 'seaward_FS_x_all', 'seaward_ActProf_x_all',
 'DF_fix_x', 'DF_der_x']

for i, var in enumerate(normalized_variables):        
    Normalisation_values = get_normalisation_value(norm_variable, norm_year, Dir_per_variable)
        
    pickle_file = Dir_per_variable + var + '_dataframe.pickle'
    Dimension_df = pickle.load(open(pickle_file, 'rb')) #load pickle of dimensions    
    Normalized_df = Dimension_df.copy()
    
    for key in Normalized_df.columns:
        Normalized_df[key] = Normalized_df[key] - Normalisation_values.loc[norm_year, key]
    
    Normalized_df.to_pickle(Dir_per_variable + var + '_normalized_dataframe' + '.pickle')
    print('The dataframe of ' + var + ' was normalized and saved')
