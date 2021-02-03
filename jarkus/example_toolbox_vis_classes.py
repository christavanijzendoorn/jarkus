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
Created on Mon Nov  2 11:56:17 2020

@author: cijzendoornvan
"""

######################
# PACKAGES
######################
import yaml
import pickle
import math
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt 
from IPython import get_ipython
from Jarkus_Analysis_Toolbox import Transects
import Filtering_functions as Ff
get_ipython().run_line_magic('matplotlib', 'auto') ## %matplotlib auto TO GET WINDOW FIGURE

######################
# LOAD SETTINGS
######################
config = yaml.safe_load(open("C:/Users/cijzendoornvan/Documents/DuneForce/JARKUS/Jarkus_Toolbox/jarkus.yml"))
location_filter = yaml.safe_load(open(config['root'] + config['data locations']['LocFilter']))
plot_titles = yaml.safe_load(open(config['root'] + config['data locations']['Titles'])) 

start_yr = 1980 
end_yr = 2020

variable = 'Dunefoot_y_der'

#%%
##################################
####       PREPARATIONS       ####
##################################
# Load jarkus dataset
data = Transects(config)
conversion_alongshore2ids, conversion_ids2alongshore = data.get_conversion_dicts()

# dimension = pickle.load(open(config['root'] + config['save locations']['DirD'] + variable + '_dataframe' + '.pickle','rb'))   
dimension = pickle.load(open(r"C:\Users\cijzendoornvan\Documents\DuneForce\JARKUS\ANALYSIS\D1_dataframes_per_dimension\Dunefoot_y_der_new_dataframe.pickle",'rb')) 
dimension = Ff.bad_locations_filter(dimension, location_filter)
dimension.rename(columns = conversion_ids2alongshore, inplace = True)
dimension = dimension.sort_index(axis=1) # Flip transects based on alongshore order instead of ids order.

# Calculate spatial and temporal average
average_through_space = dimension.loc[list(range(start_yr, end_yr))].mean(axis=0)
average_through_time = dimension.loc[list(range(start_yr, end_yr))].mean(axis=1)

# Calculate overall average and stddev, used for range of colorbar
average         = np.nanmean(dimension.values)
stddev          = np.nanstd(dimension.values, ddof=1)
range_value     = 2*stddev
range_value_avg = stddev
vmin            = average - range_value
vmax            = average + range_value
vmin_avg        = average - range_value_avg
vmax_avg        = average + range_value_avg

# Create an array with locations and an array with labels of the ticks
ticks_x = [350, 1100, 1900]
labels_x = ['Wadden Coast', 'Holland Coast', 'Delta Coast']

years_requested = list(range(start_yr, end_yr))
ticks_y = range(0, len(years_requested))[0::5]
labels_y = [str(yr) for yr in years_requested][0::5]

#%% Prepare plotting of Tidal range

# Load and plot pybeach method version
var = 'MHW_y_var'
# DF_MHW = pickle.load(open(config['root'] + config['save locations']['DirD'] + var + '_dataframe.pickle','rb'))
DF_MHW = pickle.load(open(r"C:\Users\cijzendoornvan\Documents\DuneForce\JARKUS\ANALYSIS\D1_dataframes_per_dimension\MHW_y_var_dataframe.pickle",'rb'))
DF_MHW = Ff.bad_locations_filter(DF_MHW, location_filter)
DF_MHW.rename(columns = conversion_ids2alongshore, inplace = True)
DF_MHW = DF_MHW.sort_index(axis=1) # Flip transects based on alongshore order instead of ids order.
DF_MHW.rename(columns = conversion_alongshore2ids, inplace = True)
   
MHW = DF_MHW.loc[1965]
plt.rcParams.update({'lines.linewidth': 3})

#%%###############################
####       PLOTTING       ####
##################################
# Plot overviews and trends for Derivative method DF elevation
figure_title = 'Alongshore and temporal variation of dune toe elevation (m)'
colorbar_label = 'Dune toe elevation (m)'
colormap_var = "Greens"
file_name = 'dune_foot_elevation'

# Set-up of figure
fig = plt.figure(figsize=(25,7)) 

# PLOT SPATIAL AVERAGES OF VARIABLE
cmap = plt.cm.get_cmap(colormap_var) # Define color use for colorbar
colorplot = plt.scatter(range(0, len(average_through_space)), average_through_space, c=average_through_space, cmap=cmap, vmin=0, vmax=6)
# Set labels and ticks of x and y axis
plt.xlim([0, len(average_through_space)])
plt.xticks(ticks_x, labels_x) 
plt.ylabel('Elevation (m)', fontsize = 20)
plt.tick_params(axis='x', which='both',length=0, labelsize = 20)
plt.ylim([0, 6])
plt.tick_params(axis='y', which='both',length=5, labelsize = 16)
# plt.plot(range(0, len(average_through_space)),tidal_range, color = '#4169E1', label = 'Tidal range (m)', linewidth = 6) 
plt.plot(range(0, len(average_through_space)),MHW, color = '#4169E1', label = 'Mean High Water (m)', linewidth = 6) 

plt.axvline(x=686, color='r') # Boundary kustvak 6 en 7, Den Helder, trsct 7000000, decameter 23417
plt.axvline(x=1507, color='r')# Boundary kustvak 9 en 10, Hoek van Holland, trsct 10000000, decameter 35908

plt.legend(fontsize = 16)

# Plot colorbar
cbar = fig.colorbar(colorplot)
cbar.set_label(colorbar_label,size=18, labelpad = 20)
cbar.ax.tick_params(labelsize=16) 

plt.tight_layout
plt.show()

filename2 = 'Overview_DF_part1' + file_name + '.pdf'
# plt.savefig(DirDFAnalysis + filename2)
print('saved figure')
#plt.close()










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
            
            if variable == 'dune foot elevation':
                plt.text(begin_yr, plotloc1, str(round(trend_of_yearly_mean_filt*1000,2)) + ' mm/yr')
            elif variable == 'dune foot location':
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



# DF = DF_x_Dia_norm
# figure_title = 'Alongshore and temporal variation of the cross shore dune toe location (m)'
# colorbar_label = 'Crossshore dune toe location (m)'
# colormap_var = "Greens"
# file_name = 'dune_foot_location'





