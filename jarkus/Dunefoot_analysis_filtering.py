# -*- coding: utf-8 -*-
"""
Created on Mon Jul 13 18:21:47 2020

@author: cijzendoornvan
"""

import xarray as xr
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import json
import pickle

plt.rcParams.update({'font.size': 26})
plt.rcParams.update({'lines.linewidth': 3})

# load basic settings from .txt file
with open("C:/Users/cijzendoornvan/Documents/GitHub/jarkus/jarkus/Settings.txt") as file:
    settings = json.load(file)                                                  # include USER-DEFINED settings

DirDFAnalysis = r"C:\Users\cijzendoornvan\Documents\DuneForce\JARKUS\ANALYSIS\DF_analysis\\"    
DirDF = r"C:\Users\cijzendoornvan\Documents\DuneForce\JARKUS\Comparison_methods\Derivative_Diamantidou\DF_2nd_deriv.nc"
DirDimensions = settings['Dir_D1']

# create a dataset object, based on locally saved JARKUS dataset
dataset = xr.open_dataset(DirDF)

# Load and plot second derivative method
DF_elev = dataset['dune_foot_2nd_deriv'].values
DF_cross = dataset['dune_foot_2nd_deriv_cross'].values
years = dataset['time'].values.astype('datetime64[Y]').astype(int) + 1970   
trscts = dataset['id'].values    

DF_y_Dia = pd.DataFrame(DF_elev, columns=trscts, index = years)
DF_x_Dia = pd.DataFrame(DF_cross, columns=trscts, index=years) 

DF_x_Dia[DF_x_Dia > 10000000] = np.nan
DF_y_Dia[DF_y_Dia > 10000000] = np.nan

# Load and plot pybeach method version
var = 'Dunefoot_x_pybeach_mix_new'
pickle_file = DirDimensions + var + '_dataframe.pickle'    
DF_x_pybeach_new = pickle.load(open(pickle_file, 'rb')) #load pickle of dimension    
   
var = 'Dunefoot_y_pybeach_mix_new'
pickle_file = DirDimensions + var + '_dataframe.pickle'    
DF_y_pybeach_new = pickle.load(open(pickle_file, 'rb')) #load pickle of dimension    

#%%
import statsmodels.api as sm
def fit_line2(x, y):
    """Return slope, intercept of best fit line."""
    X = sm.add_constant(x)
    model = sm.OLS(y, X, missing='drop') # ignores entires where x or y is NaN
    fit = model.fit()
    rsquared = fit.rsquared
    return fit.params[1], fit.params[0], rsquared # could also return stderr in each via fit.bse

def get_overall_trend(variable_dataframe):
    mean_per_transect = variable_dataframe.mean(axis = 0)
    median_per_transect = variable_dataframe.median(axis = 0)
    mean_per_year = variable_dataframe.mean(axis = 1)
    median_per_year = variable_dataframe.median(axis = 1)
    
    mean_trend, intercept, rsquared = fit_line2(mean_per_year.index, mean_per_year)    
    
    return mean_per_transect, median_per_transect, mean_per_year, median_per_year, mean_trend, intercept, rsquared

def get_trends_per_transect(variable_dataframe):
    trend_per_transect = pd.DataFrame({'transects': variable_dataframe.columns})
    trend_per_transect.set_index('transects', inplace=True)
    
    for i, column in variable_dataframe.iteritems():
        count_notnan = len(column) - column.isnull().sum(axis = 0)
        if count_notnan > 1: 
            trend, intercept, rsquared = fit_line2(column.index, column)
            
            trend_per_transect.loc[i, 'trend'] = trend
            trend_per_transect.loc[i, 'intercept'] = intercept
            trend_per_transect.loc[i, 'r_squared'] = rsquared
        else:
            trend = np.nan
            intercept = np.nan
            
    trend_mean = trend_per_transect['trend'].mean()
    intercept_mean = trend_per_transect['intercept'].mean()
    
    return trend_per_transect, trend_mean, intercept_mean

def get_filtered_trends(variable_dataframe, threshold):
    # Calculate trend per transect
    trend_per_transect, mean_of_trends, mean_of_intercepts = get_trends_per_transect(variable_dataframe)
    trend_per_transect['availability'] = variable_dataframe.count() / len(variable_dataframe) * 100 
    mask = trend_per_transect['availability'] >= threshold

    # Filter dataframe
    variable_dataframe_filt = variable_dataframe.copy()
    for i, col in variable_dataframe_filt.iteritems():
        if mask[i] == False:
            variable_dataframe_filt.loc[:, i] = np.nan
            
    # Calculate trend per transect
    trend_per_transect_filt, mean_of_trends_filt, mean_of_intercepts_filt = get_trends_per_transect(variable_dataframe_filt)
    
    # Calculate averaged dune foot location and trend
    mean_per_transect_filt, median_per_transect_filt, mean_per_year_filt, median_per_year_filt, trend_of_yearly_mean_filt, intercept_of_yearly_mean_filt, rsquared_of_yearly_mean_filt = get_overall_trend(variable_dataframe_filt)
    
    # Calculate trend array
    mean_of_trends_filt_array = mean_of_trends_filt*mean_per_year_filt.index + mean_of_intercepts_filt
    trend_of_yearly_mean_filt_array = trend_of_yearly_mean_filt*mean_per_year_filt.index + intercept_of_yearly_mean_filt
    
    return mean_per_year_filt, median_per_transect_filt, mean_of_trends_filt_array, trend_of_yearly_mean_filt_array, trend_of_yearly_mean_filt, rsquared_of_yearly_mean_filt

#%%
# Get and plot Dune Foot Elevation trends

# Filter columns based on data availability
threshold_availability = [0, 10, 20, 30, 40, 50, 60, 65, 70, 75, 80 , 90, 95] #percent

for threshold in threshold_availability:
    
    # Calculate averaged dune foot elevation and trend - pybeach new
    mean_per_year_filt_Dia, median_per_transect_filt_Dia, mean_of_trends_filt_array_Dia, trend_of_yearly_mean_filt_array_Dia, trend_of_yearly_mean_filt_Dia, rsquared_of_yearly_mean_filt_Dia = get_filtered_trends(DF_y_Dia, threshold)
    
    f, (ax1, ax2) = plt.subplots(2, 1, sharex='all', sharey='all', figsize=(30,20))
    
    ax1.plot(mean_per_year_filt_Dia)
    #ax1.plot(median_per_year_filt_Dia)
    ax1.plot(mean_per_year_filt_Dia.index, trend_of_yearly_mean_filt_array_Dia)
    ax1.plot(mean_per_year_filt_Dia.index, mean_of_trends_filt_array_Dia)
    
    ax1.text(1965, 2.9, str(round(trend_of_yearly_mean_filt_Dia*1000,2)) + ' mm/yr')
    ax1.text(1965, 2.8, str(round(rsquared_of_yearly_mean_filt_Dia,3)) + ' r^2')
    
    # Calculate averaged dune foot elevation and trend - pybeach new
    mean_per_year_filt_pyb, median_per_transect_filt_pyb, mean_of_trends_filt_array_pyb, trend_of_yearly_mean_filt_array_pyb, trend_of_yearly_mean_filt_pyb, rsquared_of_yearly_mean_filt_pyb = get_filtered_trends(DF_y_pybeach_new, threshold)
    
    ax2.plot(mean_per_year_filt_pyb)
    #ax2.plot(median_per_year_filt_pyb)
    ax2.plot(mean_per_year_filt_pyb.index, trend_of_yearly_mean_filt_array_pyb)
    ax2.plot(mean_per_year_filt_pyb.index, mean_of_trends_filt_array_pyb)
    ax2.text(1965, 3.30, str(round(trend_of_yearly_mean_filt_pyb*1000,2)) + ' mm/yr')
    ax2.text(1965, 3.20, str(round(rsquared_of_yearly_mean_filt_pyb,3)) + ' r^2')
    
    ax1.title.set_text('Second derivative method')
    ax2.title.set_text('Pybeach machine learning method')
    plt.xlabel('Time (yr)')
    plt.ylabel('dune foot elevation (m)')
    
    #plt.show()
    
    filename = 'Trends_dunefoot_elevation_' + str(threshold) + 'perc_v2.png'
    plt.savefig(DirDFAnalysis + filename)
