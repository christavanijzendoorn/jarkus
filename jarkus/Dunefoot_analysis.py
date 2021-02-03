# -*- coding: utf-8 -*-
"""
Created on Mon Jul  6 11:29:36 2020

@author: cijzendoornvan
"""

import xarray as xr
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import json
import pickle


def plot_overview(dataframe, trscts, years, var, DirVarPlots):
    plt.figure(figsize=(30,15))
    average = np.nanmean(dataframe[trscts].values)
    stddev = np.nanstd(dataframe[trscts].values, ddof=1)
    range_value = 2*stddev
    fig = plt.pcolor(dataframe[trscts], vmin = average-range_value, vmax = average + range_value)
    plt.title(var)
    ticks_y = range(0, len(years))[0::5]
    ticks_x = range(0, len(trscts))[0::25]
    plt.yticks(ticks_y, labels_y)
    plt.xticks(ticks_x, labels_x, rotation='vertical')
    plt.colorbar()
    plt.savefig(DirVarPlots + var + '_plot.png')
    pickle.dump(fig, open(DirVarPlots + var + '_plot.fig.pickle', 'wb'))
        
    #plt.show()
    plt.close()

# load basic settings from .txt file
with open("C:/Users/cijzendoornvan/Documents/GitHub/jarkus/jarkus/Settings.txt") as file:
    settings = json.load(file)                                                  # include USER-DEFINED settings

DirDFAnalysis = r"C:\Users\cijzendoornvan\Documents\DuneForce\JARKUS\ANALYSIS\DF_analysis\\"
    
DirDF = r"C:\Users\cijzendoornvan\Documents\DuneForce\JARKUS\Comparison_methods\Derivative_Diamantidou\DF_2nd_deriv.nc"

#%%
# create a dataset object, based on locally saved JARKUS dataset
dataset = xr.open_dataset(DirDF)

DF_elev = dataset['dune_foot_2nd_deriv'].values
DF_cross = dataset['dune_foot_2nd_deriv_cross'].values
years = dataset['time'].values.astype('datetime64[Y]').astype(int) + 1970   
trscts = dataset['id'].values    

DF_y_Dia = pd.DataFrame(DF_elev, columns=trscts, index = years)
DF_x_Dia = pd.DataFrame(DF_cross, columns=trscts, index=years) 

DF_x_Dia[DF_x_Dia > 10000000] = np.nan
DF_y_Dia[DF_y_Dia > 10000000] = np.nan

labels_y = [str(yr) for yr in years][0::5]
labels_x = [str(tr) for tr in trscts][0::25]
   
var = 'Dunefoot_x_Diamantidou'
plot_overview(DF_x_Dia, trscts, years, var, DirDFAnalysis)

var = 'Dunefoot_y_Diamantidou'
plot_overview(DF_y_Dia, trscts, years, var, DirDFAnalysis)

#%%

# Load and plot second derivative python version
DirDimensions = settings['Dir_D1']

var = 'Dunefoot_x_der'
pickle_file = DirDimensions + var + '_dataframe.pickle'    
DF_x_python = pickle.load(open(pickle_file, 'rb')) #load pickle of dimension    
plot_overview(DF_x_python, trscts, years, var, DirDFAnalysis)
   
var = 'Dunefoot_y_der'
pickle_file = DirDimensions + var + '_dataframe.pickle'    
DF_y_python = pickle.load(open(pickle_file, 'rb')) #load pickle of dimension    
plot_overview(DF_y_python, trscts, years, var, DirDFAnalysis)

#%%

# Load and plot second derivative python version
DirDimensions = settings['Dir_D1']

var = 'Dunefoot_x_der_new'
pickle_file = DirDimensions + var + '_dataframe.pickle'    
DF_x_python_new = pickle.load(open(pickle_file, 'rb')) #load pickle of dimension    
plot_overview(DF_x_python_new, trscts, years, var, DirDFAnalysis)
   
var = 'Dunefoot_y_der_new'
pickle_file = DirDimensions + var + '_dataframe.pickle'    
DF_y_python_new = pickle.load(open(pickle_file, 'rb')) #load pickle of dimension    
plot_overview(DF_y_python_new, trscts, years, var, DirDFAnalysis)

#%%
# Plot Difference between Diamantidou and python DF calculations

var = 'Dunefoot_diff_y'
DF_diff_y = DF_y_python - DF_y_Dia
plot_overview(DF_diff_y, trscts, years, var, DirDFAnalysis)
   
var = 'Dunefoot_diff_x'
DF_diff_x = DF_x_python - DF_x_Dia
plot_overview(DF_x_python - DF_x_Dia, trscts, years, var, DirDFAnalysis)

#%%
# Plot Difference between Diamantidou and python DF calculations

var = 'Dunefoot_diff_y_new'
DF_diff_y_new = DF_y_python_new - DF_y_Dia
plot_overview(DF_diff_y_new, trscts, years, var, DirDFAnalysis)
   
var = 'Dunefoot_diff_x_new'
DF_diff_x_new = DF_x_python_new - DF_x_Dia
plot_overview(DF_diff_x_new, trscts, years, var, DirDFAnalysis)
#%%

# Load and plot pybeach method version
var = 'Dunefoot_x_pybeach_mix'
pickle_file = DirDimensions + var + '_dataframe.pickle'    
DF_x_pybeach = pickle.load(open(pickle_file, 'rb')) #load pickle of dimension    
plot_overview(DF_x_pybeach, trscts, years, var, DirDFAnalysis)
   
var = 'Dunefoot_y_pybeach_mix'
pickle_file = DirDimensions + var + '_dataframe.pickle'    
DF_y_pybeach = pickle.load(open(pickle_file, 'rb')) #load pickle of dimension    
plot_overview(DF_y_pybeach, trscts, years, var, DirDFAnalysis)

#%%

# Load and plot pybeach method version NEW
var = 'Dunefoot_x_pybeach_mix_new'
pickle_file = DirDimensions + var + '_dataframe.pickle'    
DF_x_pybeach_new = pickle.load(open(pickle_file, 'rb')) #load pickle of dimension    
plot_overview(DF_x_pybeach_new, trscts, years, var, DirDFAnalysis)
   
var = 'Dunefoot_y_pybeach_mix_new'
pickle_file = DirDimensions + var + '_dataframe.pickle'    
DF_y_pybeach_new = pickle.load(open(pickle_file, 'rb')) #load pickle of dimension    
plot_overview(DF_y_pybeach_new, trscts, years, var, DirDFAnalysis)


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

#%%
# Get and plot Dune Foot Elevation trends

# Calculate trend per transect
trend_per_transect_Dia, trend_mean_y_Dia, intercept_mean_y_Dia = get_trends_per_transect(DF_y_Dia)
trend_per_transect_python_new, trend_mean_y_python_new, intercept_mean_y_python_new = get_trends_per_transect(DF_y_python_new)
trend_per_transect_pybeach_new, trend_mean_y_pybeach_new, intercept_mean_y_pybeach_new = get_trends_per_transect(DF_y_pybeach_new)

# Calculate averaged dune foot elevation and trend - Diamantidou
mean_y_per_transect_Dia, median_y_per_transect_Dia, mean_y_per_year_Dia, median_y_per_year_Dia, mean_trend_y_Dia, intercept_y_Dia, rsquared_y_Dia = get_overall_trend(DF_y_Dia)

f, (ax1, ax2) = plt.subplots(2, 1, sharex='all', sharey='all')

ax1.plot(mean_y_per_year_Dia)
#ax1.plot(median_y_per_year_Dia)
ax1.plot(mean_y_per_year_Dia.index, mean_trend_y_Dia*mean_y_per_year_Dia.index + intercept_y_Dia)
ax1.text(1965, 2.9, str(round(mean_trend_y_Dia*1000,2)) + ' mm/yr')
ax1.text(1965, 2.8, str(round(rsquared_y_Dia,3)) + ' r^2')

# # Calculate averaged dune foot elevation and trend - python new
# mean_y_per_transect_python_new, median_y_per_transect_python_new, mean_y_per_year_python_new, median_y_per_year_python_new, mean_trend_y_python_new, intercept_y_python_new, rsquared_y_python_new = get_overall_trend(DF_y_python_new)

# ax2.plot(mean_y_per_year_python_new)
# #ax2.plot(median_y_per_year_python_new)
# ax2.plot(mean_y_per_year_python_new.index, mean_trend_y_python_new*mean_y_per_year_python_new.index + intercept_y_python_new)
# ax2.text(1970, 3.00, str(round(mean_trend_y_python_new*1000,2)) + ' mm/yr')
# ax2.text(1970, 2.80, str(round(rsquared_y_python_new,3)) + ' r^2')

# Calculate averaged dune foot elevation and trend - pybeach new
mean_y_per_transect_pybeach_new, median_y_per_transect_pytbeach_new, mean_y_per_year_pybeach_new, median_y_per_year_pybeach_new, mean_trend_y_pybeach_new, intercept_y_pybeach_new, rsquared_y_pybeach_new = get_overall_trend(DF_y_pybeach_new)

ax2.plot(mean_y_per_year_pybeach_new)
#ax2.plot(median_y_per_year_pybeach_new)
ax2.plot(mean_y_per_year_pybeach_new.index, mean_trend_y_pybeach_new*mean_y_per_year_pybeach_new.index + intercept_y_pybeach_new)
ax2.text(1965, 3.30, str(round(mean_trend_y_pybeach_new*1000,2)) + ' mm/yr')
ax2.text(1965, 3.20, str(round(rsquared_y_pybeach_new,3)) + ' r^2')

ax1.plot(mean_y_per_year_Dia.index, trend_mean_y_Dia*mean_y_per_year_Dia.index + intercept_mean_y_Dia)
# ax2.plot(mean_y_per_year_python_new.index, trend_mean_y_python_new*mean_y_per_year_python_new.index + intercept_mean_y_python_new)
ax2.plot(mean_y_per_year_pybeach_new.index, trend_mean_y_pybeach_new*mean_y_per_year_pybeach_new.index + intercept_mean_y_pybeach_new)

ax1.title.set_text('Second derivative method')
ax2.title.set_text('Pybeach machine learning method')
plt.xlabel('Time (yr)')
plt.ylabel('dune foot elevation (m)')

#%%
# Get and plot Dune Foot Crossshore Location trends

# Calculate averaged dune foot location and trend - Diamantidou
mean_x_per_transect_Dia, median_x_per_transect_Dia, mean_x_per_year_Dia, median_x_per_year_Dia, mean_trend_x_Dia, intercept_x_Dia, rsquared_x_Dia = get_overall_trend(DF_x_Dia)

f, (ax1, ax2) = plt.subplots(2, 1, sharex='all', sharey='all')

ax1.plot(mean_x_per_year_Dia)
#ax1.plot(median_x_per_year_Dia)
ax1.plot(mean_x_per_year_Dia.index, mean_trend_x_Dia*mean_y_per_year_Dia.index + intercept_x_Dia)
ax1.text(1965, 0, str(round(mean_trend_x_Dia,2)) + ' m/yr')
ax1.text(1965, -10, str(round(rsquared_x_Dia,3)) + ' r^2')

# Calculate averaged dune foot elevation and trend - pybeach new
mean_x_per_transect_pybeach_new, median_x_per_transect_pybeach_new, mean_x_per_year_pybeach_new, median_x_per_year_pybeach_new, mean_trend_x_pybeach_new, intercept_x_pybeach_new, rsquared_x_pybeach_new = get_overall_trend(DF_x_pybeach_new)

ax2.plot(mean_x_per_year_pybeach_new)
#ax2.plot(median_x_per_year_pybeach_new)
ax2.plot(mean_x_per_year_pybeach_new.index, mean_trend_x_pybeach_new*mean_x_per_year_pybeach_new.index + intercept_x_pybeach_new)
ax2.text(1965, 0, str(round(mean_trend_x_pybeach_new,2)) + ' m/yr')
ax2.text(1965, -10, str(round(rsquared_x_pybeach_new,3)) + ' r^2')

# Calculate trend per transect
trend_per_transect_Dia_x, trend_mean_x_Dia, intercept_mean_x_Dia = get_trends_per_transect(DF_x_Dia)
trend_per_transect_python_new_x, trend_mean_x_python_new, intercept_mean_x_python_new = get_trends_per_transect(DF_x_python_new)
trend_per_transect_pybeach_new_x, trend_mean_x_pybeach_new, intercept_mean_x_pybeach_new = get_trends_per_transect(DF_x_pybeach_new)

ax1.plot(mean_x_per_year_Dia.index, trend_mean_x_Dia*mean_x_per_year_Dia.index + intercept_mean_x_Dia)
ax2.plot(mean_x_per_year_pybeach_new.index, trend_mean_x_pybeach_new*mean_x_per_year_pybeach_new.index + intercept_mean_x_pybeach_new)

ax1.title.set_text('Second derivative method')
ax2.title.set_text('Pybeach machine learning method')
plt.xlabel('Time (yr)')
plt.ylabel('dune foot elevation (m)')

plt.show()




