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

var = 'Dunefoot_x_der_new'
pickle_file = DirDimensions + var + '_dataframe.pickle'    
DF_x_python_new = pickle.load(open(pickle_file, 'rb')) #load pickle of dimension    
   
var = 'Dunefoot_y_der_new'
pickle_file = DirDimensions + var + '_dataframe.pickle'    
DF_y_python_new = pickle.load(open(pickle_file, 'rb')) #load pickle of dimension    


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

def get_filter_DF(variable_dataframe, filter_dict):
    # Filter dataframe
    filtered_dataframe = variable_dataframe.copy()
    for i, col in filtered_dataframe.iteritems():
        for key in filter_file.keys():
            if i >= int(filter_file[key]['begin']) and i <= int(filter_file[key]['eind']):
                filtered_dataframe.loc[:, i] = np.nan
            
    return filtered_dataframe

def get_filter_yrs_DF(variable_dataframe, begin_year, end_year):
    # Filter dataframe
    filtered_dataframe = variable_dataframe.copy()
    for i, row in filtered_dataframe.iterrows():
        if i < begin_year or i > end_year:
            filtered_dataframe.loc[i, :] = np.nan
            
    return filtered_dataframe

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

def plot_overview(variable_DF):
    
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
    plt.title('DF elevation pybeach with flipped alongshore values')
    ticks_y = range(0, len(years))[0::5]
    ticks_x = range(0, len(variable_DF.columns))[0::25]
    labels_y = [str(yr) for yr in years][0::5]
    labels_x = [str(tr) for tr in variable_DF.columns][0::25]
    plt.yticks(ticks_y, labels_y)
    plt.xticks(ticks_x, labels_x, rotation='vertical')
    plt.colorbar()
    # plt.savefig(DirVarPlots + var + '_plot.png')
    # pickle.dump(fig, open(DirVarPlots + var + '_plot.fig.pickle', 'wb'))
            
    plt.show()
    # plt.close()

#%% CREATE GRAPHS FOR PRESET SIZED WINDOWS

DF_y_Dia = get_filter_DF(DF_y_Dia, filter_file)
DF_y_pybeach_new = get_filter_DF(DF_y_pybeach_new, filter_file)

begin_yr = 1985
end_yr = 2017

DF_y_Dia = get_filter_yrs_DF(DF_y_Dia, begin_yr, end_yr)
DF_y_pybeach_new = get_filter_yrs_DF(DF_y_pybeach_new, begin_yr, end_yr)

conversion_alongshore2ids, conversion_ids2alongshore = get_conversion_dicts(DF_y_Dia.columns)

plot_overview(DF_y_Dia)
plot_overview(DF_y_pybeach_new)

window_size = 1000 # decameter
threshold = 0
area_bounds = [2000000, 3000000, 4000000, 5000000, 6000000, 7000000, 8000000, 9000000, 10000000, 11000000, 12000000, 13000000, 14000000, 15000000, 16000000, 17000000]

for i, a in enumerate(area_bounds):
    if i == len(area_bounds) - 1:
        DF_y_Dia_area = DF_y_Dia.loc[:, area_bounds[i] <= DF_y_Dia.columns]
        DF_y_pybeach_area = DF_y_pybeach_new.loc[:, area_bounds[i] <= DF_y_pybeach_new.columns]
    else:
        DF_y_Dia_area = DF_y_Dia.loc[:, (area_bounds[i] <= DF_y_Dia.columns) & (DF_y_Dia.columns < area_bounds[i+1])]
        DF_y_pybeach_area = DF_y_pybeach_new.loc[:, (area_bounds[i] <= DF_y_pybeach_new.columns) & (DF_y_pybeach_new.columns < area_bounds[i+1])]
    
    DF_y_Dia_area.rename(columns = conversion_ids2alongshore, inplace = True)
    DF_y_Dia_area = DF_y_Dia_area.sort_index(axis=1) # Flip transects based on alongshore order instead of ids order.    
    
    DF_y_pybeach_area.rename(columns = conversion_ids2alongshore, inplace = True)
    DF_y_pybeach_area = DF_y_pybeach_area.sort_index(axis=1) # Flip transects based on alongshore order instead of ids order.    

    begin_end_points = list(range(min(DF_y_Dia_area.columns), max(DF_y_Dia_area.columns), window_size))
    print(begin_end_points)

    for j, point in enumerate(begin_end_points):
        if j == len(begin_end_points)-1:
            DF_y_Dia_block = DF_y_Dia_area.loc[:, (begin_end_points[j] <= DF_y_Dia_area.columns)]
            DF_y_pybeach_block = DF_y_pybeach_area.loc[:, (begin_end_points[j] <= DF_y_pybeach_area.columns)]
        else:
            DF_y_Dia_block = DF_y_Dia_area.loc[:, (begin_end_points[j] <= DF_y_Dia_area.columns) & (DF_y_Dia_area.columns < begin_end_points[j+1])]    
            DF_y_pybeach_block = DF_y_pybeach_area.loc[:, (begin_end_points[j] <= DF_y_pybeach_area.columns) & (DF_y_pybeach_area.columns < begin_end_points[j+1])]    
        
        ids_block = [conversion_alongshore2ids[col] for col in DF_y_Dia_block.columns]
        
        # Calculate averaged dune foot elevation and trend
        if DF_y_Dia_block.empty == False and DF_y_Dia_block.dropna(how='all').empty == False:

            mean_per_year_filt, median_per_transect_filt, mean_of_trends_filt_array, trend_of_yearly_mean_filt_array, trend_of_yearly_mean_filt, rsquared_of_yearly_mean_filt = get_filtered_trends(DF_y_Dia_block, threshold)
            
            plt.figure(figsize=(20,10))
            
            plt.plot(mean_per_year_filt)
            #ax1.plot(median_per_year_filt_Dia)
            plt.plot(mean_per_year_filt.index, trend_of_yearly_mean_filt_array)
            plt.plot(mean_per_year_filt.index, mean_of_trends_filt_array)
            plt.ylim([0, 5])
            
            plt.text(1985, 3.9, str(round(trend_of_yearly_mean_filt*1000,2)) + ' mm/yr')
            plt.text(1985, 3.7, str(round(rsquared_of_yearly_mean_filt,3)) + ' r^2')
            
            plt.title('Trend in dune foot elevation (2nd deriv.) between transect ' + str(min(ids_block)) + ' and ' + str(max(ids_block)))
            
            plt.xlabel('Time (yr)')
            plt.ylabel('Dune foot elevation (m)')
        
            filename = 'Trends_dunefoot_elevation_block_' + str(min(ids_block)) + '_secondder_1985.png'
            plt.savefig(DirDFAnalysis + filename)
            print('saved figure')
            
            plt.close()
        
        if DF_y_pybeach_block.empty == False and DF_y_pybeach_block.dropna(how='all').empty == False:

            mean_per_year_filt, median_per_transect_filt, mean_of_trends_filt_array, trend_of_yearly_mean_filt_array, trend_of_yearly_mean_filt, rsquared_of_yearly_mean_filt = get_filtered_trends(DF_y_pybeach_block, threshold)
            
            plt.figure(figsize=(20,10))
            
            plt.plot(mean_per_year_filt)
            #ax1.plot(median_per_year_filt_Dia)
            plt.plot(mean_per_year_filt.index, trend_of_yearly_mean_filt_array)
            plt.plot(mean_per_year_filt.index, mean_of_trends_filt_array)
            plt.ylim([0, 5])
            
            plt.text(1985, 3.9, str(round(trend_of_yearly_mean_filt*1000,2)) + ' mm/yr')
            plt.text(1985, 3.7, str(round(rsquared_of_yearly_mean_filt,3)) + ' r^2')
            
            plt.title('Trend in dune foot elevation (pybeach) between transect ' + str(min(ids_block)) + ' and ' + str(max(ids_block)))
            
            plt.xlabel('Time (yr)')
            plt.ylabel('Dune foot elevation (m)')
        
            filename = 'Trends_dunefoot_elevation_block_' + str(min(ids_block)) + '_pybeach_1985.png'
            plt.savefig(DirDFAnalysis + filename)
            print('saved figure')
            
            plt.close()
            
#%% CREATE GRAPHS FOR WADDEN, HOLLAND and DELTA COAST.
    
# Set new column ids based on alongshore values
conversion_alongshore2ids, conversion_ids2alongshore = get_conversion_dicts(DF_y_Dia.columns)

threshold = 0
area_bounds = [2000000, 7000000, 10000000]

DF_y_Dia = get_filter_DF(DF_y_Dia, filter_file)
DF_y_pybeach_new = get_filter_DF(DF_y_pybeach_new, filter_file)


begin_yr = 1985
end_yr = 2017

DF_y_Dia = get_filter_yrs_DF(DF_y_Dia, begin_yr, end_yr)
DF_y_pybeach_new = get_filter_yrs_DF(DF_y_pybeach_new, begin_yr, end_yr)

for i, a in enumerate(area_bounds):
    if i == len(area_bounds) - 1:
        DF_y_Dia_area = DF_y_Dia.loc[:, area_bounds[i] <= DF_y_Dia.columns]
        DF_y_pybeach_area = DF_y_pybeach_new.loc[:, area_bounds[i] <= DF_y_pybeach_new.columns]
    else:
        DF_y_Dia_area = DF_y_Dia.loc[:, (area_bounds[i] <= DF_y_Dia.columns) & (DF_y_Dia.columns < area_bounds[i+1])]
        DF_y_pybeach_area = DF_y_pybeach_new.loc[:, (area_bounds[i] <= DF_y_pybeach_new.columns) & (DF_y_pybeach_new.columns < area_bounds[i+1])]
    
    DF_y_Dia_area.rename(columns = conversion_ids2alongshore, inplace = True)
    DF_y_Dia_area = DF_y_Dia_area.sort_index(axis=1) # Flip transects based on alongshore order instead of ids order.    
    
    DF_y_pybeach_area.rename(columns = conversion_ids2alongshore, inplace = True)
    DF_y_pybeach_area = DF_y_pybeach_area.sort_index(axis=1) # Flip transects based on alongshore order instead of ids order.    

    ids_block = [conversion_alongshore2ids[col] for col in DF_y_Dia_area.columns]
    
    # Calculate averaged dune foot elevation and trend
    if DF_y_Dia_area.empty == False and DF_y_Dia_area.dropna(how='all').empty == False:

        mean_per_year_filt, median_per_transect_filt, mean_of_trends_filt_array, trend_of_yearly_mean_filt_array, trend_of_yearly_mean_filt, rsquared_of_yearly_mean_filt = get_filtered_trends(DF_y_Dia_area, threshold)
        
        plt.figure(figsize=(15,10))
        
        plt.plot(mean_per_year_filt)
        #ax1.plot(median_per_year_filt_Dia)
        plt.plot(mean_per_year_filt.index, trend_of_yearly_mean_filt_array)
        plt.plot(mean_per_year_filt.index, mean_of_trends_filt_array)
        plt.ylim([2, 4])
        
        plt.text(1985, 3.9, str(round(trend_of_yearly_mean_filt*1000,2)) + ' mm/yr')
        plt.text(1985, 3.7, str(round(rsquared_of_yearly_mean_filt,3)) + ' r^2')
        
        plt.title('Trend in dune foot elevation (2nd deriv.) between transect ' + str(min(ids_block)) + ' and ' + str(max(ids_block)))
        
        plt.xlabel('Time (yr)')
        plt.ylabel('Dune foot elevation (m)')
    
        filename = 'Trends_dunefoot_elevation_area_' + str(min(ids_block)) + '_secondder_1985.png'
        plt.savefig(DirDFAnalysis + filename)
        print('saved figure')
        
        plt.close()
    
    if DF_y_pybeach_area.empty == False and DF_y_pybeach_area.dropna(how='all').empty == False:

        mean_per_year_filt, median_per_transect_filt, mean_of_trends_filt_array, trend_of_yearly_mean_filt_array, trend_of_yearly_mean_filt, rsquared_of_yearly_mean_filt = get_filtered_trends(DF_y_pybeach_area, threshold)
        
        plt.figure(figsize=(15,10))
        
        plt.plot(mean_per_year_filt)
        #ax1.plot(median_per_year_filt_Dia)
        plt.plot(mean_per_year_filt.index, trend_of_yearly_mean_filt_array)
        plt.plot(mean_per_year_filt.index, mean_of_trends_filt_array)
        plt.ylim([2, 4])
        
        plt.text(1985, 3.9, str(round(trend_of_yearly_mean_filt*1000,2)) + ' mm/yr')
        plt.text(1985, 3.7, str(round(rsquared_of_yearly_mean_filt,3)) + ' r^2')
        
        plt.title('Trend in dune foot elevation (pybeach) between transect ' + str(min(ids_block)) + ' and ' + str(max(ids_block)))
        
        plt.xlabel('Time (yr)')
        plt.ylabel('Dune foot elevation (m)')
    
        filename = 'Trends_dunefoot_elevation_area_' + str(min(ids_block)) + '_pybeach_1985.png'
        plt.savefig(DirDFAnalysis + filename)
        print('saved figure')
        
        plt.close()

#%% CREATE GRAPHS FOR WADDEN, HOLLAND and DELTA COAST. - FILTERED
    
# Set new column ids based on alongshore values
conversion_alongshore2ids, conversion_ids2alongshore = get_conversion_dicts(DF_y_Dia.columns)

DF_y_Dia = get_filter_DF(DF_y_Dia, filter_file)
DF_y_pybeach_new = get_filter_DF(DF_y_pybeach_new, filter_file)

begin_yr = 1965
end_yr = 2017

DF_y_Dia = get_filter_yrs_DF(DF_y_Dia, begin_yr, end_yr)
DF_y_pybeach_new = get_filter_yrs_DF(DF_y_pybeach_new, begin_yr, end_yr)

# Set new column ids based on alongshore values
plot_overview(DF_y_Dia)
plot_overview(DF_y_pybeach_new)

threshold = 0
area_bounds = [2000000, 7000000, 10000000]

for i, a in enumerate(area_bounds):
    if i == len(area_bounds) - 1:
        DF_y_Dia_area = DF_y_Dia.loc[:, area_bounds[i] <= DF_y_Dia.columns]
        DF_y_pybeach_area = DF_y_pybeach_new.loc[:, area_bounds[i] <= DF_y_pybeach_new.columns]
    else:
        DF_y_Dia_area = DF_y_Dia.loc[:, (area_bounds[i] <= DF_y_Dia.columns) & (DF_y_Dia.columns < area_bounds[i+1])]
        DF_y_pybeach_area = DF_y_pybeach_new.loc[:, (area_bounds[i] <= DF_y_pybeach_new.columns) & (DF_y_pybeach_new.columns < area_bounds[i+1])]
    
    DF_y_Dia_area.rename(columns = conversion_ids2alongshore, inplace = True)
    DF_y_Dia_area = DF_y_Dia_area.sort_index(axis=1) # Flip transects based on alongshore order instead of ids order.    
    
    DF_y_pybeach_area.rename(columns = conversion_ids2alongshore, inplace = True)
    DF_y_pybeach_area = DF_y_pybeach_area.sort_index(axis=1) # Flip transects based on alongshore order instead of ids order.    

    ids_block = [conversion_alongshore2ids[col] for col in DF_y_Dia_area.columns]
    
    # Calculate averaged dune foot elevation and trend
    if DF_y_Dia_area.empty == False and DF_y_Dia_area.dropna(how='all').empty == False:

        mean_per_year_filt, median_per_transect_filt, mean_of_trends_filt_array, trend_of_yearly_mean_filt_array, trend_of_yearly_mean_filt, rsquared_of_yearly_mean_filt = get_filtered_trends(DF_y_Dia_area, threshold)
        
        plt.figure(figsize=(15,10))
        
        plt.plot(mean_per_year_filt)
        #ax1.plot(median_per_year_filt_Dia)
        plt.plot(mean_per_year_filt.index, trend_of_yearly_mean_filt_array)
        plt.plot(mean_per_year_filt.index, mean_of_trends_filt_array)
        plt.ylim([2, 4])
        
        plt.text(1965, 3.9, str(round(trend_of_yearly_mean_filt*1000,2)) + ' mm/yr')
        plt.text(1965, 3.7, str(round(rsquared_of_yearly_mean_filt,3)) + ' r^2')
        
        plt.title('Trend in dune foot elevation (2nd deriv.) between transect ' + str(min(ids_block)) + ' and ' + str(max(ids_block)))
        
        plt.xlabel('Time (yr)')
        plt.ylabel('Dune foot elevation (m)')
    
        filename = 'Trends_dunefoot_elevation_area_filtered_allyrs' + str(min(ids_block)) + '_secondder.png'
        plt.savefig(DirDFAnalysis + filename)
        print('saved figure')
        
        plt.close()
    
    if DF_y_pybeach_area.empty == False and DF_y_pybeach_area.dropna(how='all').empty == False:

        mean_per_year_filt, median_per_transect_filt, mean_of_trends_filt_array, trend_of_yearly_mean_filt_array, trend_of_yearly_mean_filt, rsquared_of_yearly_mean_filt = get_filtered_trends(DF_y_pybeach_area, threshold)
        
        plt.figure(figsize=(15,10))
        
        plt.plot(mean_per_year_filt)
        #ax1.plot(median_per_year_filt_Dia)
        plt.plot(mean_per_year_filt.index, trend_of_yearly_mean_filt_array)
        plt.plot(mean_per_year_filt.index, mean_of_trends_filt_array)
        plt.ylim([2, 4])
        
        plt.text(1965, 3.9, str(round(trend_of_yearly_mean_filt*1000,2)) + ' mm/yr')
        plt.text(1965, 3.7, str(round(rsquared_of_yearly_mean_filt,3)) + ' r^2')
        
        plt.title('Trend in dune foot elevation (pybeach) between transect ' + str(min(ids_block)) + ' and ' + str(max(ids_block)))
        
        plt.xlabel('Time (yr)')
        plt.ylabel('Dune foot elevation (m)')
    
        filename = 'Trends_dunefoot_elevation_area_filtered_allyrs' + str(min(ids_block)) + '_pybeach.png'
        plt.savefig(DirDFAnalysis + filename)
        print('saved figure')
        
        plt.close()
        
#%% CREATE GRAPHS FOR WADDEN, HOLLAND and DELTA COAST. - FILTERED - Crossshore location
    
# Set new column ids based on alongshore values
conversion_alongshore2ids, conversion_ids2alongshore = get_conversion_dicts(DF_x_Dia.columns)

DF_x_Dia = get_filter_DF(DF_x_Dia, filter_file)
DF_x_pybeach_new = get_filter_DF(DF_x_pybeach_new, filter_file)

begin_yr = 1985
end_yr = 2017

DF_x_Dia = get_filter_yrs_DF(DF_x_Dia, begin_yr, end_yr)
DF_x_pybeach_new = get_filter_yrs_DF(DF_x_pybeach_new, begin_yr, end_yr)

# Set new column ids based on alongshore values
plot_overview(DF_x_Dia)
plot_overview(DF_x_pybeach_new)

threshold = 0
area_bounds = [2000000, 7000000, 10000000]

for i, a in enumerate(area_bounds):
    if i == len(area_bounds) - 1:
        DF_x_Dia_area = DF_x_Dia.loc[:, area_bounds[i] <= DF_x_Dia.columns]
        DF_x_pybeach_area = DF_x_pybeach_new.loc[:, area_bounds[i] <= DF_x_pybeach_new.columns]
    else:
        DF_x_Dia_area = DF_x_Dia.loc[:, (area_bounds[i] <= DF_x_Dia.columns) & (DF_x_Dia.columns < area_bounds[i+1])]
        DF_x_pybeach_area = DF_x_pybeach_new.loc[:, (area_bounds[i] <= DF_x_pybeach_new.columns) & (DF_x_pybeach_new.columns < area_bounds[i+1])]
    
    DF_x_Dia_area.rename(columns = conversion_ids2alongshore, inplace = True)
    DF_x_Dia_area = DF_x_Dia_area.sort_index(axis=1) # Flip transects based on alongshore order instead of ids order.    
    
    DF_x_pybeach_area.rename(columns = conversion_ids2alongshore, inplace = True)
    DF_x_pybeach_area = DF_x_pybeach_area.sort_index(axis=1) # Flip transects based on alongshore order instead of ids order.    

    ids_block = [conversion_alongshore2ids[col] for col in DF_x_Dia_area.columns]
    
    # Calculate averaged dune foot elevation and trend
    if DF_x_Dia_area.empty == False and DF_x_Dia_area.dropna(how='all').empty == False:

        mean_per_year_filt, median_per_transect_filt, mean_of_trends_filt_array, trend_of_yearly_mean_filt_array, trend_of_yearly_mean_filt, rsquared_of_yearly_mean_filt = get_filtered_trends(DF_x_Dia_area, threshold)
        
        plt.figure(figsize=(15,10))
        
        plt.plot(mean_per_year_filt)
        #ax1.plot(median_per_year_filt_Dia)
        plt.plot(mean_per_year_filt.index, trend_of_yearly_mean_filt_array)
        plt.plot(mean_per_year_filt.index, mean_of_trends_filt_array)
        plt.ylim([-100, 100])
        
        plt.text(1965, 0, str(round(trend_of_yearly_mean_filt,2)) + ' m/yr')
        plt.text(1965, 10, str(round(rsquared_of_yearly_mean_filt,3)) + ' r^2')
        
        plt.title('Trend in dune foot location (2nd deriv.) between transect ' + str(min(ids_block)) + ' and ' + str(max(ids_block)))
        
        plt.xlabel('Time (yr)')
        plt.ylabel('Dune foot location (m)')
    
        filename = 'Trends_dunefoot_location_area_filtered_' + str(min(ids_block)) + '_secondder.png'
        plt.savefig(DirDFAnalysis + filename)
        print('saved figure')
        
        plt.close()
    
    if DF_x_pybeach_area.empty == False and DF_x_pybeach_area.dropna(how='all').empty == False:

        mean_per_year_filt, median_per_transect_filt, mean_of_trends_filt_array, trend_of_yearly_mean_filt_array, trend_of_yearly_mean_filt, rsquared_of_yearly_mean_filt = get_filtered_trends(DF_x_pybeach_area, threshold)
        
        plt.figure(figsize=(15,10))
        
        plt.plot(mean_per_year_filt)
        #ax1.plot(median_per_year_filt_Dia)
        plt.plot(mean_per_year_filt.index, trend_of_yearly_mean_filt_array)
        plt.plot(mean_per_year_filt.index, mean_of_trends_filt_array)
        plt.ylim([-100, 100])
        
        plt.text(1965, 0, str(round(trend_of_yearly_mean_filt,2)) + ' m/yr')
        plt.text(1965, 10, str(round(rsquared_of_yearly_mean_filt,3)) + ' r^2')
        
        plt.title('Trend in dune foot location (pybeach) between transect ' + str(min(ids_block)) + ' and ' + str(max(ids_block)))
        
        plt.xlabel('Time (yr)')
        plt.ylabel('Dune foot location (m)')
    
        filename = 'Trends_dunefoot_location_area_filtered_' + str(min(ids_block)) + '_pybeach.png'
        plt.savefig(DirDFAnalysis + filename)
        print('saved figure')
        
        plt.close()