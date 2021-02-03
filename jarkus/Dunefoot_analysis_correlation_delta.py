# -*- coding: utf-8 -*-
"""
Created on Tue Aug  4 17:14:16 2020

@author: cijzendoornvan
"""

import xarray as xr
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import json
import pickle
import math

plt.rcParams.update({'font.size': 26})
plt.rcParams.update({'lines.linewidth': 3})

# load basic settings from .txt file
with open("C:/Users/cijzendoornvan/Documents/GitHub/jarkus/jarkus/Settings.txt") as file:
    settings = json.load(file)                                                  # include USER-DEFINED settings
    
# load basic settings from .txt file
with open("C:/Users/cijzendoornvan/Documents/GitHub/jarkus/jarkus/Filter2.txt") as ffile:
    filter_file = json.load(ffile)                                                  # include USER-DEFINED settings

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
area_codes = dataset['areacode'].values

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

def normalisation(DF, Dir_per_variable, variable): # Get norm values for the cross-shore location for each transect in the norm year
    DF_norm = DF.copy()
    for i, col in DF.iteritems():
        DF_norm.loc[:, i] = col - col.mean()
    DF.to_pickle(Dir_per_variable + var + '_normalized_mean_dataframe' + '.pickle')
    print('The dataframe of ' + var + ' was normalized and saved')
       
    return DF_norm

variable = 'Dunefoot_y_secder'
DF_x_Dia_norm = normalisation(DF_x_Dia, DirDimensions, variable)
  
var = 'Dunefoot_x_pybeach_mix_new'
DF_x_pybeach_norm = normalisation(DF_x_pybeach_new, DirDimensions, variable)    


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
            trend_per_transect.loc[i, 'trend'] = np.nan
            trend_per_transect.loc[i, 'intercept'] = np.nan
            trend_per_transect.loc[i, 'r_squared'] = np.nan
            
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

def bad_locations_filter(variable_dataframe, filter_dict):
    # Filter dataframe
    filtered_dataframe = variable_dataframe.copy()
    removed_transects = []
    for i, col in filtered_dataframe.iteritems():
        for key in filter_file.keys():
            if i >= int(filter_file[key]['begin']) and i <= int(filter_file[key]['eind']):
                removed_transects.append(i)
                filtered_dataframe.loc[:, i] = np.nan
    
    percentage = len(removed_transects)/len(filtered_dataframe.columns)*100
    print('Removed percentage of transects is ' + str(percentage))
            
    return filtered_dataframe

def bad_yrs_filter(variable_dataframe, begin_year, end_year):
    # Filter dataframe
    filtered_dataframe = variable_dataframe.copy()
    for i, row in filtered_dataframe.iterrows():
        if i < begin_year or i > end_year:
            filtered_dataframe.loc[i, :] = np.nan
            
    return filtered_dataframe

def nourishment_filter(variable_dataframe):
    Nourishments = pd.read_excel("C:/Users/cijzendoornvan/Documents/Duneforce/JARKUS/Suppletiedatabase.xlsx")
    filtered = []
    for index, row in Nourishments.iterrows():
        if math.isnan(row['BeginRaai']) or math.isnan(row['EindRaai']):# or row['Volume/m'] > 50: # or row['JaarBeginUitvoering'] < 2010: 
            continue
        else:
            code_beginraai = int(row['KustVakNummer'] * 1000000 + row['BeginRaai'] * 100)
            code_eindraai = int(row['KustVakNummer'] * 1000000 + row['EindRaai'] * 100)
            nourished_transects = [i for i in variable_dataframe.columns if i >= code_beginraai and i <= code_eindraai]
            filtered.extend(nourished_transects)
    filtered = set(filtered)
    not_nourished_transects = [i for i in variable_dataframe.columns if i not in filtered]

    # Filter dataframe
    not_nourished_dataframe = variable_dataframe.copy()
    for i, col in not_nourished_dataframe.iteritems():
        if i in filtered:
                not_nourished_dataframe.loc[:, i] = np.nan
    
    nourished_dataframe = variable_dataframe.copy()
    for i, col in nourished_dataframe.iteritems():
        if i not in filtered:
                nourished_dataframe.loc[:, i] = np.nan
            
    return nourished_dataframe, not_nourished_dataframe#, not_nourished_transects

# Create conversion dictionary
def get_conversion_dicts(ids): 
    area_bounds = [2000000, 3000000, 4000000, 5000000, 6000000, 7000000, 8000000, 9000000, 10000000, 11000000, 12000000, 13000000, 14000000, 15000000, 16000000, 17000000, 18000000]
    
    for i, val in enumerate(area_bounds):
        if i == 0: # Flip numbers for first Wadden Island
            ids_filt = trscts[np.where(np.logical_and(trscts>=area_bounds[i], trscts<area_bounds[i+1]))] #- area_bounds[i]
            transition_value = min(ids_filt) - area_bounds[i]
            
            ids_filt = [transition_value + (max(ids_filt) - ids) for ids in ids_filt]
            
            ids_alongshore = ids_filt
        elif i < 6: # For the Wadden Islands, flipping the alongshore numbers and creating space between islands
            ids_filt = trscts[np.where(np.logical_and(trscts>=area_bounds[i], trscts<area_bounds[i+1]))] #- area_bounds[i]
            transition_value = 100
            ids_old = ids_filt
            
            ids_filt = [transition_value + (max(ids_filt) - ids) for ids in ids_filt]
            ids_filt = [max(ids_alongshore) + ids for ids in ids_filt]
            
            ids_alongshore = np.concatenate((ids_alongshore, ids_filt))
        elif i == 6 or i == 7: # Where alongshore numbers are counting throughout consecutive area codes
            ids_filt = trscts[np.where(np.logical_and(trscts>=area_bounds[i], trscts<area_bounds[i+1]))] #- area_bounds[i]
            
            transition_value = (min(ids_filt) - area_bounds[i])  - (max(ids_old) - area_bounds[i-1])
            ids_old = ids_filt
            
            ids_filt = [transition_value + (ids - min(ids_filt)) for ids in ids_filt]
            ids_filt = [max(ids_alongshore) + ids for ids in ids_filt]
            
            ids_alongshore = np.concatenate((ids_alongshore, ids_filt))
        elif i == 16: # Done
            print("Converted all areacodes to alongshore values")
        else: # Create space between area codes and no flipping necessary.
            ids_filt = trscts[np.where(np.logical_and(trscts>=area_bounds[i], trscts<area_bounds[i+1]))] #- area_bounds[i]
            transition_value = 100
            ids_old = ids_filt
            
            ids_filt = [transition_value + (ids - min(ids_filt)) for ids in ids_filt]
            ids_filt = [max(ids_alongshore) + ids for ids in ids_filt]
    
            ids_alongshore = np.concatenate((ids_alongshore, ids_filt))
        
        # Create conversion dictionary  
        conversion_alongshore2ids = dict(zip(ids_alongshore, trscts))
        conversion_ids2alongshore = dict(zip(trscts, ids_alongshore))
        
    return conversion_alongshore2ids, conversion_ids2alongshore

def plot_overview(variable_DF, variable, method, DF_type, DirVarPlots):
    
    variable_DF = variable_DF.copy()
    conversion_alongshore2ids, conversion_ids2alongshore = get_conversion_dicts(variable_DF.columns) 

    variable_DF.rename(columns = conversion_ids2alongshore, inplace = True)
    variable_DF = variable_DF.sort_index(axis=1) # Flip transects based on alongshore order instead of ids order.
    
    variable_DF.rename(columns = conversion_alongshore2ids, inplace = True)
    
    plt.figure(figsize=(50,25))
    average = np.nanmean(variable_DF.values)
    stddev = np.nanstd(variable_DF.values, ddof=1)
    range_value = 2*stddev
    fig = plt.pcolor(variable_DF, vmin = average-range_value, vmax = average + range_value)
    plt.title(variable + ' ' + method + ' ' + DF_type)
    ticks_y = range(0, len(years))[0::5]
    ticks_x = range(0, len(variable_DF.columns))[0::25]
    labels_y = [str(yr) for yr in years][0::5]
    labels_x = [str(tr) for tr in variable_DF.columns][0::25]
    plt.yticks(ticks_y, labels_y)
    plt.xticks(ticks_x, labels_x, rotation='vertical')
    plt.colorbar()
    plt.savefig(DirVarPlots + 'Overview_' + variable.replace(' ', '') + '_' + DF_type.replace(' ','') + '_' + method + '.png')
            
    plt.show()
    # plt.close()

def plot_overview_types(variable_dataframe, filter_file, begin_yr, end_yr, variable, method, DirVarPlots): 

    DF_filtered = bad_locations_filter(variable_dataframe, filter_file)
    DF_filtered = bad_yrs_filter(DF_filtered, begin_yr, end_yr)
    DF_nourished, DF_not_nourished = nourishment_filter(DF_filtered)
    
    # Set new column ids based on alongshore values
    DF_type = 'filtered'
    plot_overview(DF_filtered, variable, method, DF_type, DirVarPlots)
    DF_type = 'nourished'
    plot_overview(DF_nourished, variable, method, DF_type, DirVarPlots)
    DF_type = 'not_nourished'
    plot_overview(DF_not_nourished, variable, method, DF_type, DirVarPlots)

    return DF_filtered, DF_nourished, DF_not_nourished #, not_nourished_transects     

def plot_trend(DF, variable, method, unit, DF_type, begin_yr, end_yr, ylimit):
    
    threshold = 0
    area_bounds = [2000000, 7000000, 10000000]        
    
    conversion_alongshore2ids, conversion_ids2alongshore = get_conversion_dicts(DF.columns) 

    for i, a in enumerate(area_bounds):
        if i == len(area_bounds) - 1:
            DF_area = DF.loc[:, area_bounds[i] <= DF.columns]
        else:
            DF_area = DF.loc[:, (area_bounds[i] <= DF.columns) & (DF.columns < area_bounds[i+1])]
        
        DF_area.rename(columns = conversion_ids2alongshore, inplace = True)
        DF_area = DF_area.sort_index(axis=1) # Flip transects based on alongshore order instead of ids order.    
        
        ids_block = [conversion_alongshore2ids[col] for col in DF_area.columns]    
        
        missing_trscts = 0
        DF_noNans = DF_area.isna().sum()
        for i in DF_noNans:
            if i == len(DF_area.index):
                missing_trscts += 1
        print('There are ' + str(len(ids_block) - missing_trscts) + 'transects in ' + variable + ' (' + method + ', ' + DF_type + ') ' + 'between transect ' + str(min(ids_block)) + ' and ' + str(max(ids_block)))
        
        if DF_area.empty == False and DF_area.dropna(how='all').empty == False:
    
            mean_per_year_filt, median_per_transect_filt, mean_of_trends_filt_array, trend_of_yearly_mean_filt_array, trend_of_yearly_mean_filt, rsquared_of_yearly_mean_filt = get_filtered_trends(DF_area, threshold)
            
            plt.figure(figsize=(15,10))
            
            plt.plot(mean_per_year_filt)
            plt.plot(mean_per_year_filt.index, trend_of_yearly_mean_filt_array)
            plt.plot(mean_per_year_filt.index, mean_of_trends_filt_array)
            plt.ylim(ylimit)
            plt.xlim([begin_yr, end_yr])
            
            plotloc2 = (ylimit[1] - ylimit[0])*0.2 + ylimit[0]
            plotloc1 = (ylimit[1] - ylimit[0])*0.25 + ylimit[0]
            
            if variable == 'dune toe elevation':
                plt.text(begin_yr, plotloc1, str(round(trend_of_yearly_mean_filt*1000,2)) + ' mm/yr')
            elif variable == 'dune toe location':
                plt.text(begin_yr, plotloc1, str(round(trend_of_yearly_mean_filt,2)) + ' m/yr')
            
            plt.text(begin_yr, plotloc2, str(round(rsquared_of_yearly_mean_filt,3)) + ' r^2')
                
            plt.title('Trend in ' + variable + ' (' + method + ', ' + DF_type + ') ' + 'between transect ' + str(min(ids_block)) + ' and ' + str(max(ids_block)))
            
            plt.xlabel('Time (yr)')
            plt.ylabel(variable + '(' + unit + ')')
        
            variable_str = variable.replace(' ','')
            DF_type_str = DF_type.replace(' ','')
            filename = 'Trend_' + variable_str + '_' + DF_type_str + '_' + method + '_' + str(min(ids_block)) + '.png'
            plt.savefig(DirDFAnalysis + filename)
            print('saved figure')
            
            plt.close()
            


# Set new column ids based on alongshore values

# Plot overviews and trends for Derivative method DF elevation
begin_yr = 1980
end_yr = 2017
DF = DF_y_Dia
variable = 'dune toe elevation'
method = 'secder'
DF_y_Dia_filtered, DF_y_Dia_nourished, DF_y_Dia_not_nourished = plot_overview_types(DF, filter_file, begin_yr, end_yr, variable, method, DirDFAnalysis)

# Plot overviews and trends for pybeach method DF elevation
DF = DF_y_pybeach_new
variable = 'dune toe elevation'
method = 'pybeach'
DF_y_pybeach_filtered, DF_y_pybeach_nourished, DF_y_pybeach_not_nourished = plot_overview_types(DF, filter_file, begin_yr, end_yr, variable, method, DirDFAnalysis)


# Plot overviews and trends for Derivative method DF location
DF = DF_x_Dia_norm
variable = 'dune toe location'
method = 'secder'
DF_x_Dia_filtered, DF_x_Dia_nourished, DF_x_Dia_not_nourished = plot_overview_types(DF, filter_file, begin_yr, end_yr, variable, method, DirDFAnalysis)


# Plot overviews and trends for pybeach method DF location
DF = DF_x_pybeach_norm
variable = 'dune toe location'
method = 'pybeach'
DF_x_pybeach_filtered, DF_x_pybeach_nourished, DF_x_pybeach_not_nourished = plot_overview_types(DF, filter_file, begin_yr, end_yr, variable, method, DirDFAnalysis)

#%%
def plot_correlation(DF_x, DF_y, begin_yr, end_yr, xlimit, ylimit, method):
    
    area_bounds = [2000000, 7000000, 10000000]        
    area_bounds = [0, 18000000]
    
    conversion_alongshore2ids, conversion_ids2alongshore = get_conversion_dicts(DF.columns) 

    for i, a in enumerate(area_bounds):
        if i == len(area_bounds) - 1:
            DF_area_x = DF_x.loc[:, area_bounds[i] <= DF_x.columns]
            DF_area_y = DF_y.loc[:, area_bounds[i] <= DF_y.columns]
        else:
            DF_area_x = DF_x.loc[:, (area_bounds[i] <= DF_x.columns) & (DF_x.columns < area_bounds[i+1])]
            DF_area_y = DF_y.loc[:, (area_bounds[i] <= DF_y.columns) & (DF_y.columns < area_bounds[i+1])]
        
        DF_area_x.rename(columns = conversion_ids2alongshore, inplace = True)
        DF_area_x = DF_area_x.sort_index(axis=1) # Flip transects based on alongshore order instead of ids order.    
        
        DF_area_y.rename(columns = conversion_ids2alongshore, inplace = True)
        DF_area_y = DF_area_y.sort_index(axis=1) # Flip transects based on alongshore order instead of ids order.    
        
        ids_block = [conversion_alongshore2ids[col] for col in DF_area_x.columns]    
        
        if DF_area_x.empty == False and DF_area_x.dropna(how='all').empty == False:
    
            x = list(DF_area_x.mean(axis = 1))
            y = list(DF_area_y.mean(axis = 1))
            
            delta_x = [x[n]-x[n-1] for n in range(1,len(x))]
            delta_y = [y[n]-y[n-1] for n in range(1,len(y))]
            
            plt.figure(figsize=(25,20))
            
            plt.scatter(delta_x, delta_y)
            plt.ylim(ylimit)
            plt.xlim(xlimit)
        
            plt.title('Correlation between change in dune foot elevation and cross shore location between ' + str(min(ids_block)) + ' and ' + str(max(ids_block)))
            
            plt.ylabel('Dune foot elevation change (m)')
            plt.xlabel('Dune foot location change (m)')
        
            filename = 'Trend_DF_deltax_and_deltay' + '_' + method + '_' + str(min(ids_block)) + 'all.png'
            plt.savefig(DirDFAnalysis + filename)
            print('saved figure')
            
            plt.close()
        
begin_yr = 1985
end_yr = 2017
ylimit = [-1,1]
xlimit = [-10,10]
method = 'secder'
plot_correlation(DF_x_Dia_filtered, DF_y_Dia_filtered, begin_yr, end_yr, xlimit, ylimit, method)           


begin_yr = 1985
end_yr = 2017
ylimit = [-1,1]
xlimit = [-10,10]
method = 'pybeach'
plot_correlation(DF_x_pybeach_filtered, DF_y_pybeach_filtered, begin_yr, end_yr, xlimit, ylimit, method)           
        
        #%%
def get_delta_x_and_y(DF_x, DF_y):
    
    area_bounds = [2000000, 7000000, 10000000]        
    
    conversion_alongshore2ids, conversion_ids2alongshore = get_conversion_dicts(DF.columns) 
    
    delta_x_areas = pd.DataFrame()
    delta_y_areas = pd.DataFrame()
    
    for i, a in enumerate(area_bounds):
        if i == len(area_bounds) - 1:
            DF_area_x = DF_x.loc[:, area_bounds[i] <= DF_x.columns]
            DF_area_y = DF_y.loc[:, area_bounds[i] <= DF_y.columns]
        else:
            DF_area_x = DF_x.loc[:, (area_bounds[i] <= DF_x.columns) & (DF_x.columns < area_bounds[i+1])]
            DF_area_y = DF_y.loc[:, (area_bounds[i] <= DF_y.columns) & (DF_y.columns < area_bounds[i+1])]
        
        DF_area_x.rename(columns = conversion_ids2alongshore, inplace = True)
        DF_area_x = DF_area_x.sort_index(axis=1) # Flip transects based on alongshore order instead of ids order.    
        
        DF_area_y.rename(columns = conversion_ids2alongshore, inplace = True)
        DF_area_y = DF_area_y.sort_index(axis=1) # Flip transects based on alongshore order instead of ids order.    
        
        ids_block = [conversion_alongshore2ids[col] for col in DF_area_x.columns]    
        
        if DF_area_x.empty == False and DF_area_x.dropna(how='all').empty == False:
    
            x = list(DF_area_x.mean(axis = 1))
            y = list(DF_area_y.mean(axis = 1))
            
            delta_x = [x[n]-x[n-1] for n in range(1,len(x))]
            delta_y = [y[n]-y[n-1] for n in range(1,len(y))]
        
        delta_x_areas[area_bounds[i]] = delta_x
        delta_y_areas[area_bounds[i]] = delta_y
            
    return delta_x_areas, delta_y_areas
        
delta_x_Dia, delta_y_Dia = get_delta_x_and_y(DF_x_Dia_filtered, DF_y_Dia_filtered)           
delta_x_pybeach, delta_y_pybeach = get_delta_x_and_y(DF_x_pybeach_filtered, DF_y_pybeach_filtered)      

begin_yr = 1985
end_yr = 2017
ylimit = [-0.8,0.8]
xlimit = [-12,12]
method1 = 'second derivative method'
method2 = 'pybeach method'

plt.figure(figsize=(12,12))

aa = plt.scatter(delta_x_pybeach[7000000], delta_y_pybeach[7000000], marker = 'x', color = 'r', s=100)
ab = plt.scatter(delta_x_pybeach[10000000], delta_y_pybeach[10000000], marker = 'o', color = 'r', s=100)
ba = plt.scatter(delta_x_Dia[7000000], delta_y_Dia[7000000], marker = 'x', color = 'b', s=80)
bb = plt.scatter(delta_x_Dia[10000000], delta_y_Dia[10000000], marker = 'o', color = 'b', s=80)
plt.ylim(ylimit)
plt.xlim(xlimit)

proxy1 = plt.plot([], [], 'o', markerfacecolor='k', markeredgecolor='k', markersize=12)[0]
proxy2 = plt.plot([], [], 'x', markerfacecolor='k', markeredgecolor='k', markersize=12)[0]
patch1 = mpatches.Patch(color='r')
patch2 = mpatches.Patch(color='b')

list_of_proxy_artists = [proxy1, proxy2, patch1, patch2]
list_of_labels = ['Holland Coast', 'Delta Coast', 'second derivative method', 'pybeach method']

plt.legend(list_of_proxy_artists, list_of_labels, 
            scatterpoints=1,
            loc='lower left',
            ncol=2,
            fontsize=22)

plt.title('Correlation between change in elevation and \n change in cross shore location of the dune toe', pad=20)

plt.ylabel('Dune toe elevation change (m)')
plt.xlabel('Dune toe location change (m)')

filename = 'Trend_DF_deltax_and_deltay.png'
plt.savefig(DirDFAnalysis + filename)
print('saved figure')

# plt.close()     

#%%
def plot_correlation_trends(DF_x, DF_y, begin_yr, end_yr, xlimit, ylimit, method):
    
    threshold = 0
    area_bounds = [7000000, 10000000]        
    
    conversion_alongshore2ids, conversion_ids2alongshore = get_conversion_dicts(DF_x.columns) 
    
    fig = plt.figure(figsize=(12,12))
    
    trends_per_area = pd.DataFrame()
    
    trend_per_transect_x, trend_mean_x, intercept_mean_x = get_trends_per_transect(DF_x)
    trend_per_transect_y, trend_mean_y, intercept_mean_y = get_trends_per_transect(DF_y)
    
    DF_x = trend_per_transect_x.transpose()
    DF_y = trend_per_transect_y.transpose()
    
    for i, a in enumerate(area_bounds):
        if i == len(area_bounds) - 1:
            DF_area_x = DF_x.loc[:, area_bounds[i] <= DF_x.columns]
            DF_area_y = DF_y.loc[:, area_bounds[i] <= DF_y.columns]
        else:
            DF_area_x = DF_x.loc[:, (area_bounds[i] <= DF_x.columns) & (DF_x.columns < area_bounds[i+1])]
            DF_area_y = DF_y.loc[:, (area_bounds[i] <= DF_y.columns) & (DF_y.columns < area_bounds[i+1])]
        
        DF_area_x.rename(columns = conversion_ids2alongshore, inplace = True)
        DF_area_x = DF_area_x.sort_index(axis=1) # Flip transects based on alongshore order instead of ids order.    
        
        DF_area_y.rename(columns = conversion_ids2alongshore, inplace = True)
        DF_area_y = DF_area_y.sort_index(axis=1) # Flip transects based on alongshore order instead of ids order.    
        
        ids_block = [conversion_alongshore2ids[col] for col in DF_area_x.columns]    
        
        if min(ids_block) == area_bounds[0]:
            region = 'Holland Coast'   
            color = 'r'
        elif min(ids_block) == area_bounds[1]:
            region = 'Delta Coast'
            color = 'b'
        plt.scatter(DF_area_x.loc['trend'], DF_area_y.loc['trend'], color = color, marker = 'x', s = 8, label = region)
        
    plt.ylim(ylimit)
    plt.xlim(xlimit)
    
    plt.xlabel('Trend in cross shore dune toe location')
    plt.ylabel('Trend in dune toe elevation')
        
    plt.tick_params(axis='x', pad=15)
        
    plt.legend()
    
    # fig.tight_layout(rect=[0, 0.03, 1, 0.95])
        
    filename = 'Trends_correlation_' + method + '.png'
    plt.savefig(DirDFAnalysis + filename)
    print('saved figure')
    
    # plt.close()
        
begin_yr = 1980
end_yr = 2017
ylimit = [-0.25,0.25]
xlimit = [-10, 10]
method = 'secder'
plot_correlation_trends(DF_x_Dia_filtered, DF_y_Dia_filtered, begin_yr, end_yr, xlimit, ylimit, method)           


begin_yr = 1980
end_yr = 2017
ylimit = [-0.25,0.25]
xlimit = [-10,10]
method = 'pybeach'
plot_correlation_trends(DF_x_pybeach_filtered, DF_y_pybeach_filtered, begin_yr, end_yr, xlimit, ylimit, method)           
        