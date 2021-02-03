# -*- coding: utf-8 -*-
"""
Created on Mon Dec 14 17:20:52 2020

@author: cijzendoornvan
"""

import yaml
import os
import pickle
import numpy as np
import numpy as np
import pandas as pd
import pickle
import os
import xarray as xr
# get_ipython().run_line_magic('matplotlib', 'auto') ## %matplotlib auto TO GET WINDOW FIGURE

######################
# LOAD SETTINGS
######################
config = yaml.safe_load(open("C:/Users/cijzendoornvan/Documents/DuneForce/JARKUS/Jarkus_Toolbox/jarkus.yml"))
dataset = xr.open_dataset(config['root'] + config['data locations']['DirJK'])

start_yr = 1965
end_yr = 2020
transects_requested = 9010883

time = dataset.variables['time'].values                     
years = pd.to_datetime(time).year                    
years_requested = list(range(start_yr, end_yr))
years_filter =  np.isin(years, years_requested)
years_filtered = np.array(years)[np.nonzero(years_filter)[0]]
years_filtered_idxs = np.where(years_filter)[0]

ids = dataset.variables['id'].values                             
transects_filter = np.isin(ids, transects_requested)
transects_filtered = np.array(ids)[np.nonzero(transects_filter)[0]]
transects_filtered_idxs = np.where(transects_filter)[0]

elevation = dataset.variables['altitude'].values[years_filtered_idxs, transects_filtered_idxs, :]
crossshore = dataset.variables['cross_shore'].values

def dune_foot_2nd_derivative(trscts, years, height_of_peaks=2.4, thresholdFlatDuration=5, thresholdZderiv1=0.001, thresholdZderiv2=0.01):
    
    x_df = np.zeros(len(years))
    y_df = np.zeros(len(years))
    
    z = np.zeros(len(years_filtered_idxs), 1, len(crossshore)) 


return x_df, z_df, x, z, z_mean_high


x_df, y_df, x, z, x_mean_high = dune_foot_2nd_derivative(transects_filtered_idxs, years_filtered_idxs)

# def get_landward_point_derivative(self, trsct_idx):
        
#     ####  Derivative method - Diamantidou ####
#     ###################################
#     # Get landward boundary from peaks in profile
#     from scipy.signal import find_peaks
    
#     height_of_peaks = self.config['user defined']['landward derivative']['min height'] #m
#     height_constraint = self.config['user defined']['landward derivative']['height constraint'] #m
#     peaks_threshold = height_of_peaks + self.dimensions['MHW_y_var'].values[0]
    
#     for i, yr in enumerate(self.data.years_filtered):
#         yr_idx = self.data.years_filtered_idxs[0][i]
        
#         elevation = self.data.variables['altitude'].values[yr_idx, trsct_idx, :]
#         peaks = find_peaks(elevation, prominence = height_of_peaks)[0] # Documentation see get_dune_top
        
#         peaks = elevation[peaks]
#         peaks_filt = peaks[peaks >= peaks_threshold]
        
#         if len(peaks) != 0 and np.nanmax(peaks) > height_constraint:
#             intersections_derivative = find_intersections(elevation, self.crossshore, height_constraint)
#             if len(intersections_derivative) != 0:
#                 self.dimensions.loc[yr, 'Landward_x_der'] = intersections_derivative[-1]
#         elif len(peaks_filt) != 0:
#             self.dimensions.loc[yr, 'Landward_x_der'] = peaks_filt[-1]
#         else:
#             self.dimensions.loc[yr, 'Landward_x_der'] = np.nan


# ####  Derivative method - E.D. ####
# ###################################
# ## Variable dunefoot definition based on first and second derivative of profile
# Dimensions['DF_der_y'] = np.nan
# Dimensions['DF_der_x'] = np.nan
# for i, yr in enumerate(years_requested_str):
#     if yr in years_available:
#         # Get seaward boundary
#         seaward_x = Dimensions.loc[yr, 'MHW_x_fix']
#         # Get landward boundary 
#         landward_x = Dimensions.loc[yr, 'landward_6m_x']   

#         # Give nan values to everything outside of boundaries
#         y_all_domain = []
#         for xc in range(len(cross_shore)): 
#             if cross_shore[xc] > seaward_x or cross_shore[xc] < landward_x:
#                 y_all_domain.append(np.nan)
#             else:
#                 y_all_domain.append(y_all[xc,i])
    
#         # Give nan to years where not enough measurements are available within domain
#         if np.count_nonzero(~np.isnan(y_all_domain)) > 30:
#             # Get first and second derivative
#             y_der1 = np.gradient(y_all_domain, cross_shore)    
#             y_der2 = np.gradient(y_der1, cross_shore)    
        
#             # Set first derivative values between -0.001 and 0.001 to zero
#             # Set second derivative values above 0.01 to zero
#             for n in range(len(y_der1)):
#                 if -0.001 <= y_der1[n] <= 0.001:
#                     y_der1[n] = 0
#                 if y_der2[n] >= 0.01:
#                     y_der2[n] = 0
                    
#             # Set to values to nan, where consecutive zeros occur in the first derivative
#             y_der1_clean = np.zeros(len(y_der1))
#             y_der1_clean[:] = np.nan
#             y_der_remove = False
#             for j in range(len(y_der1)-1):
#                 if y_der1[j] == 0 and y_der1[j] == y_der1[j+1]:
#                     y_der1_clean[j] = np.nan
#                     y_der_remove = True
#                 elif y_der_remove == True:
#                     y_der1_clean[j] = np.nan
#                     y_der_remove = False
#                 else:
#                     y_der1_clean[j] = y_der1[j]
                    
#             # Set locations to True where both first and second derivative are zero
#             dunefoot = np.zeros(len(y_der1))
#             for l in range(len(y_der1)):
#                 if y_der2[l] == 0:
#                     dunefoot[l] = True
                
#             # Get most seaward point where the above condition is True
#             if sum(dunefoot) != 0:
#                 dunefoot_idx = np.where(dunefoot == True)[0][0]
#                 Dimensions.loc[yr, 'DF_der_x'] = cross_shore[dunefoot_idx]
#                 Dimensions.loc[yr, 'DF_der_y'] = y_all[dunefoot_idx, i]