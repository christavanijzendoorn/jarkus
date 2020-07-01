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

# This script was created to try and build a simpler version of the already existing Jarkus Transects python toolbox. Work in progress....

# -*- coding: utf-8 -*-
"""
Created on Tue Nov 19 11:31:25 2019

@author: cijzendoornvan
"""

import numpy as np
import pandas as pd

#################################
####     DATA-EXTRACTION     ####
#################################

def get_jarkus_data(DirJk):
    import xarray as xr
    
    # create a dataset object, based on locally saved JARKUS dataset
    dataset = xr.open_dataset(DirJk)
    variables = dataset.variables
    
    return dataset, variables

def get_years_filtered(start_year, end_year, variables):
    time = variables['time'].values                                                 # retrieve years from jarkus dataset
    years = pd.to_datetime(time).year                                               # convert to purely integers indicating the measurement year
    years_requested = list(range(start_year, end_year))
    years_filter =  np.isin(years, years_requested)
    years_filtered = np.array(years)[np.nonzero(years_filter)[0]]
    years_filtered_idxs = np.where(years_filter)
    
    return years_filtered, years_filtered_idxs

def get_transects_filtered(transects_requested, variables):
    ids = variables['id'].values                                                    # retrieve transect ids from jarkus dataset
    transects_filter = np.isin(ids, transects_requested)
    transects_filtered = np.array(ids)[np.nonzero(transects_filter)[0]]
    transects_filtered_idxs = np.where(transects_filter)[0]
    
    return transects_filtered, transects_filtered_idxs

def get_dataframe_transect(variables, start_yr, end_yr, trsct, trsct_id):
    
    yrs_filt, yrs_filt_idxs = get_years_filtered(start_yr, end_yr, variables)             # returns filter based on years requested for data retrieval
    
    crossshore = variables['cross_shore'].values
    
    j = trsct_id
    trsct = variables['id'].values[j]
    areacode = variables['areacode'].values[j]                             # code per region
    alongshore = variables['alongshore'].values[j]                         # alongshore position within region
    
    angle = variables['angle'].values[j]                                   # angle of transect, positive clockwise to 0 North
    rsp_x = variables['rsp_x'].values[j]                                   # x position of rijksstrandpaal (beach pole) in RijksDriehoek projected coordinates
    rsp_y = variables['rsp_y'].values[j]                                   # y position of rijksstrandpaal (beach pole) in RijksDriehoek projected coordinates
    rsp_lat = variables['rsp_lat'].values[j]                               # x position of rijksstrandpaal (beach pole) in WGS84 projected coordinates
    rsp_lon = variables['rsp_lon'].values[j]                               # y position of rijksstrandpaal (beach pole) in WGS84 projected coordinates
    x = variables['x'].values[j,:]                                         # x position of profile point in RijksDriehoek projected coordinates
    y = variables['y'].values[j,:]                                         # y position of profile point in RijksDriehoek projected coordinates
    lat = variables['lat'].values[j,:]                                     # x position of profile point in WGS84 projected coordinates
    lon = variables['lon'].values[j,:]                                     # y position of profile point in WGS84 projected coordinates
    
    mhw = variables['mean_high_water'].values[j]                            # mean high water location per transect relative to NAP
    mlw = variables['mean_low_water'].values[j]                             # mean low water location per transect relative to NAP
    
    df_jrk_yrs = pd.DataFrame()
    for i in yrs_filt_idxs[0]:
        df_jrk = pd.DataFrame()
        yr = variables['time'].values[i].astype('datetime64[Y]').astype(int) + 1970
        date_start = pd.to_datetime(variables['time_bounds'].values[i,0])  # start date of measurement year
        date_end = pd.to_datetime(variables['time_bounds'].values[i,1])    # end date of measurement year
        
        max_cross_shore_measurement = variables['max_cross_shore_measurement'].values[i, j]    # seaward cross-shore boundary of profile measurement
        min_cross_shore_measurement = variables['min_cross_shore_measurement'].values[i, j]    # landward cross-shore boundary of profile measurement
        max_altitude_measurement = variables['max_altitude_measurement'].values[i, j]          # maximum elevation in profile            
        min_altitude_measurement = variables['min_altitude_measurement'].values[i, j]          # minimum elevation in profile
        
        nsources = variables['nsources'].values[i, j]                      # number of sources used for proile
        origin = variables['origin'].values[i, j, :]                       # datasource per point in profile, 1) beach only, 2) beach overlap, 3) interpolation, 4) sea overlap, 5) sea only
        time_topo = pd.to_datetime(variables['time_topo'].values[i, j])     # time of measurement of topography
        time_bathy = pd.to_datetime(variables['time_bathy'].values[i, j])   # time of measurement of bathymetry
        
        altitude = variables['altitude'].values[i, j, :]                    # elevation of profile point

        df_jrk = pd.DataFrame({'transect': trsct, 'year': yr, 'areacode': areacode, 'alongshore': alongshore, 'crossshore': crossshore, 
                               'angle': angle, 'rsp_x': rsp_x, 'rsp_y': rsp_y, 'rsp_lat': rsp_lat, 'rsp_lon': rsp_lon,
                               'x': x, 'y': y, 'lat': lat, 'lon': lon, 'mhw': mhw, 'mlw': mlw, 'date_start': date_start, 'date_end': date_end,
                               'max_cross_shore_measurement': max_cross_shore_measurement, 'min_cross_shore_measurement': min_cross_shore_measurement,
                               'nsources': nsources, 'origin': origin, 
                               'max_altitude_measurement': max_altitude_measurement, 'min_altitude_measurement': min_altitude_measurement, 
                               'time_topo': time_topo, 'time_bathy': time_bathy, 'altitude': altitude})
    
        df_jrk_yrs = df_jrk_yrs.append(df_jrk)
        
    #df_jrk_yrs.set_index(['transect', 'year'], inplace=True)
        
    return df_jrk_yrs

#################################
####     VISUALISATION       ####
#################################
    
def get_transect_plot(df, trsct, dirplots):
    
    import matplotlib.pyplot as plt
    import matplotlib.colors as colors
    import matplotlib.cm as cm
    import pickle
    
    years = df.index

	# Set figure layout
    fig = plt.figure(figsize=(30,15))
    ax = fig.add_subplot(111)
    ax.tick_params(axis='x', labelsize=20)
    ax.tick_params(axis='y', labelsize=20)
    
    jet = plt.get_cmap('jet') 
    cNorm  = colors.Normalize(vmin=min(years), vmax=max(years))
    scalarMap = cm.ScalarMappable(norm=cNorm, cmap=jet)    
    
    # Load and plot data per year
    for i, yr in enumerate(years):
        colorVal = scalarMap.to_rgba(yr)
        df.loc[yr].plot(color=colorVal, label = str(yr), linewidth = 2.5)
    
    # Added this to get the legend to work
    handles,labels = ax.get_legend_handles_labels()
    ax.legend(handles, labels, loc='lower left',ncol=2, fontsize = 20)
    
    # Label the axes and provide a title
    ax.set_title("Transect {}".format(str(trsct)), fontsize = 28)
    ax.set_xlabel("Cross shore distance [m]", fontsize = 24)
    ax.set_ylabel("Elevation [m to datum]", fontsize = 24)
    
    # Set x and y limit of plots - Leave lists empty (i.e. []) for automatic axis limits
    xlim = [-200,500] # EXAMPLE: [-400,1000]
    ylim = [-10, 25] # EXAMPLE: [-10,22]
    if len(xlim) != 0:
        ax.set_xlim(xlim)
    if len(ylim) != 0:
        ax.set_ylim(ylim)
    #ax.grid()
    #ax.invert_xaxis()
        
    # Save figure as png in predefined directory
    plt.savefig(dirplots + 'Transect_' + str(trsct) + '.png')
    pickle.dump(fig, open(dirplots + 'Transect_' + str(trsct) + '.fig.pickle', 'wb'))
    
    plt.close()
    
def reopen_pickle(plots_dir="", transect=""):
    #To reopen pickle:
    import pickle
    figx = pickle.load(open(plots_dir + 'Transect_' + transect + '.fig.pickle','rb'))    
    figx.show()
    
#################################
####     DATA PREPARATION    ####
#################################
    
def elevation_filter(elev_dataframe, min_elev, max_elev):
    for idx, row in elev_dataframe.iterrows():
        if min(row) > min_elev or max(row) < max_elev:
            elev_dataframe.drop(idx, axis=0)
    
    return elev_dataframe

def get_elevations_dataframe(df, min_elev, max_elev):
    elev_dataframe = df.pivot(index='year', columns='crossshore', values='altitude')
    elev_dataframe_filtered = elevation_filter(elev_dataframe, min_elev, max_elev)
    
    return elev_dataframe_filtered

#################################
####    ANALYSIS FUNCTIONS   ####
#################################
def find_intersections(profile, y_value):
    value_vec = np.array([y_value] * len(profile.index))
    profile.interpolate(inplace=True)
    
    diff = np.nan_to_num(np.diff(np.sign(profile.values - value_vec)))
    intersection_idxs = np.nonzero(diff)
    intersection_x = np.array([profile.index[idx] for idx in intersection_idxs[0]])
    
    return intersection_x

def get_gradient(elev_dataframe, dimensions_df, seaward_bound, landward_bound):
    gradient = []
    for yr, row in elev_dataframe.iterrows():
        
        # Get seaward boundary
        seaward_x = dimensions_df.loc[yr, seaward_bound]
        # Get landward boundary 
        landward_x = dimensions_df.loc[yr, landward_bound] 
        
        # Remove everything outside of boundaries
        row = row.drop(row.index[row.index > seaward_x]) # drop everything seaward of seaward boundary
        row = row.drop(row.index[row.index < landward_x]).interpolate() # drop everything landward of landward boundary and interpolate remaining data
        
        # remove nan values otherqise polyfit does not work
        row = row.dropna(axis=0)
        
        # Calculate gradient for domain
        if sum(row.index) == 0:
            gradient.append(np.nan)
        else:
            gradient.append(np.polyfit(row.index, row.values, 1)[0])    
            
    return gradient

def get_volume(elev_dataframe, dimensions_df, seaward_bound, landward_bound):
    volume = []
    for yr, row in elev_dataframe.iterrows(): 
        
        # Get seaward boundary
        seaward_x = np.ceil(dimensions_df.loc[yr, seaward_bound])
        # Get landward boundary 
        landward_x = np.floor(dimensions_df.loc[yr, landward_bound] )
        
        if np.isnan(seaward_x) or np.isnan(landward_x):
            volume.append(np.nan)
        else:
            # Remove everything outside of boundaries
            row = row.drop(row.index[row.index > seaward_x]) # drop everything seaward of seaward boundary
            row = row.drop(row.index[row.index < landward_x]).interpolate() # drop everything landward of landward boundary and interpolate remaining data
            volume_y = row - row.min()
            
            volume_trapz = np.trapz(volume_y, x = volume_y.index)
            volume.append(volume_trapz)
    
    return volume

def zero_runs(y_der, threshold_zeroes):                    
    # Create an array that is 1 where y_der is 0, and pad each end with an extra 0.
    iszero = np.concatenate(([0], np.equal(y_der, 0).view(np.int8), [0]))
    absdiff = np.abs(np.diff(iszero))
    # Runs start and end where absdiff is 1.
    zero_sections = np.where(absdiff == 1)[0].reshape(-1, 2)                     
    zero_section_len = zero_sections[:,1] - zero_sections[:,0]
    
    zero_sections = zero_sections[zero_section_len > threshold_zeroes]
            
    return zero_sections

#################################
####   DIMENSION EXTRACTION  ####
#################################

def get_dune_height_and_location(dune_height_and_location, elev_dataframe, dimensions_df):
    
    if dune_height_and_location == True:
        from scipy.signal import find_peaks
        
        for yr, row in elev_dataframe.iterrows():
            dune_top_prim = find_peaks(row, height = 5, prominence = 2.0) # , distance = 5
            dune_top_sec = find_peaks(row, height = 3, prominence = 0.5) # , distance = 5
            # Documentation:
            # https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.find_peaks.html#scipy.signal.find_peaks
            # https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.peak_prominences.html#scipy.signal.peak_prominences    
            # The prominence of a peak measures how much a peak stands out from the surrounding baseline of the signal and is defined as the vertical distance between the peak and its lowest contour line.
            if len(dune_top_prim[0]) != 0: # If a peak is found in the profile
                # Select the most seaward peak found of the primary and secondary peaks
                dune_top_prim_idx = dune_top_prim[0][-1]
                dune_top_sec_idx = dune_top_sec[0][-1]
                #print(yr, cross_shore[dune_top_prim_idx], cross_shore[dune_top_sec_idx])
                
                if  dune_top_sec_idx <= dune_top_prim_idx: 
                    # If most seaward secondary peak is located at the same place or landward of the most seaward primary peak
                    dimensions_df.loc[yr, 'DuneTop_prim_x'] = row.index[dune_top_prim_idx]  # Save the primary peak location
                    dimensions_df.loc[yr, 'DuneTop_prim_y'] = row.iloc[dune_top_prim_idx]
                    #Assume that there is no seaward secondary peak, so no value filled in (i.e. it stays nan).
                else:            
                    # Otherwise save both the primary and secondary peak location
                    dimensions_df.loc[yr, 'DuneTop_prim_x'] = row.index[dune_top_prim_idx] 
                    dimensions_df.loc[yr, 'DuneTop_prim_y'] = row.iloc[dune_top_prim_idx]
                    dimensions_df.loc[yr, 'DuneTop_sec_x'] = row.index[dune_top_sec_idx]
                    dimensions_df.loc[yr, 'DuneTop_sec_y'] = row.iloc[dune_top_sec_idx]
            else:
                dimensions_df.loc[yr, 'DuneTop_prim_x'] = np.nan
                dimensions_df.loc[yr, 'DuneTop_prim_y'] = np.nan
                dimensions_df.loc[yr, 'DuneTop_sec_x'] = np.nan
                dimensions_df.loc[yr, 'DuneTop_sec_y'] = np.nan
    
    return dimensions_df

def get_mean_sea_level(mean_sea_level, elev_dataframe, dimensions_df):
    if mean_sea_level == True:
    
        MSL_y       = 0 # in m above reference datum    ASSUMPTION
        
        for yr, row in elev_dataframe.iterrows(): 
            intersections = find_intersections(row, MSL_y)
            if len(intersections) != 0 and np.isnan(dimensions_df.loc[yr, 'DuneTop_prim_x']):
                dimensions_df.loc[yr, 'MSL_x'] = intersections[-1] # get most seaward intersect
            
            # The following filtering is implemented to make sure offshore shallow parts are not identified as MSL. This is mostyl applicable for the Wadden Islands and Zeeland.
            elif len(intersections) != 0: 
                # get all intersections seaward of dunetop
                intersection_sw = intersections[intersections > dimensions_df.loc[yr, 'DuneTop_prim_x']] 
                # if distance between intersections seaward of dune peak is larger than 100m:
                if len(intersection_sw) != 0:
                    if max(intersection_sw) - min(intersection_sw) > 100: 
                        # get intersections at least 100m landwards of most offshore intersection 
                        intersection_lw = intersection_sw[intersection_sw < (min(intersection_sw) + 100)] 
                        # Of these, select the most seaward intersection
                        dimensions_df.loc[yr, 'MSL_x'] = intersection_lw[-1] 
                    else: 
                        # If the intersection seaward of the dunetop are within 100m of each other take the most seaward one.
                        dimensions_df.loc[yr, 'MSL_x'] = intersection_sw[-1]
                else:
                    dimensions_df.loc[yr, 'MSL_x'] = np.nan
            else:
                dimensions_df.loc[yr, 'MSL_x'] = np.nan
     
    return dimensions_df

def get_mean_low_water_fixed(mean_low_water_fixed, elev_dataframe, dimensions_df):
    if mean_low_water_fixed == True:
        
        MLW_y_fixed   = -1 # in m above reference datum
        
        for yr, row in elev_dataframe.iterrows(): 
            intersections_fix = find_intersections(row, MLW_y_fixed)
            if len(intersections_fix) != 0:
                # filter intersections based on the assumption that mean low water should be a maximum of 250 m offshore
                intersections_fix_filt = intersections_fix[(intersections_fix < dimensions_df.loc[yr, 'MSL_x'] + 250)]
                if len(intersections_fix_filt) == 0:
                    dimensions_df.loc[yr, 'MLW_x_fix'] = intersections_fix[-1]
                else: 
                    dimensions_df.loc[yr, 'MLW_x_fix'] = intersections_fix_filt[-1]
            else:
                dimensions_df.loc[yr, 'MLW_x_fix'] = np.nan
                    
    return dimensions_df

def get_mean_low_water_variable(mean_low_water_variable, elev_dataframe, dataframe, dimensions_df):
    if mean_low_water_variable == True:
        
        for yr, row in elev_dataframe.iterrows():
            
            MLW_y_variable   = dataframe.loc[yr, 'mlw'].iloc[0]     # gets the mean low water level as included in the Jarkus dataset, so it varies per location
            dimensions_df.loc[yr, 'MLW_y_var'] = MLW_y_variable     # save assumed mean low water level as extracted from the jarkus dataset
            
            intersections_var = find_intersections(row, MLW_y_variable)
            if len(intersections_var) != 0:
                # filter intersections based on the assumption that mean low water should be a maximum of 250 m offshore
                intersections_var_filt = intersections_var[(intersections_var < dimensions_df.loc[yr, 'MSL_x'] + 250)]
                if len(intersections_var_filt) == 0:
                    dimensions_df.loc[yr, 'MLW_x_var'] = intersections_var[-1]
                else:
                    dimensions_df.loc[yr, 'MLW_x_var'] = intersections_var_filt[-1]
            else:
                dimensions_df.loc[yr, 'MLW_x_var'] = np.nan
                    
    return dimensions_df

def get_mean_high_water_fixed(mean_high_water_fixed, elev_dataframe, dimensions_df):
    if mean_high_water_fixed == True:
        
        MLW_y_fixed   = 1 # in m above reference datum
        
        for yr, row in elev_dataframe.iterrows(): 
            intersections_fix = find_intersections(row, MLW_y_fixed)
            if len(intersections_fix) != 0:
                # filter intersections based on the assumption that mean high water should be a maximum of 250 m landward
                intersections_fix_filt = intersections_fix[(intersections_fix < dimensions_df.loc[yr, 'MSL_x'] - 250)]
                if len(intersections_fix_filt) == 0:
                    dimensions_df.loc[yr, 'MHW_x_fix'] = intersections_fix[-1]
                else: 
                    dimensions_df.loc[yr, 'MHW_x_fix'] = intersections_fix_filt[-1]
            else:
                dimensions_df.loc[yr, 'MHW_x_fix'] = np.nan
                    
    return dimensions_df

def get_mean_high_water_variable(mean_high_water_variable, elev_dataframe, dataframe, dimensions_df):
    if mean_high_water_variable == True:
        
        for yr, row in elev_dataframe.iterrows():
            
            MLW_y_variable   = dataframe.loc[yr, 'mhw'].iloc[0]     # gets the mean low water level as included in the Jarkus dataset, so it varies per location
            dimensions_df.loc[yr, 'MHW_y_var'] = MLW_y_variable     # save assumed mean low water level as extracted from the jarkus dataset
            
            intersections_var = find_intersections(row, MLW_y_variable)
            if len(intersections_var) != 0:
                # filter intersections based on the assumption that mean high water should be a maximum of 250 m landward
                intersections_var_filt = intersections_var[(intersections_var < dimensions_df.loc[yr, 'MSL_x'] - 250)]
                if len(intersections_var_filt) == 0:
                    dimensions_df.loc[yr, 'MHW_x_var'] = intersections_var[-1]
                else:
                    dimensions_df.loc[yr, 'MHW_x_var'] = intersections_var_filt[-1]
            else:
                dimensions_df.loc[yr, 'MHW_x_var'] = np.nan
                    
    return dimensions_df

def get_mean_sea_level_variable(mean_sea_level_variable, dimensions_df):
    if mean_sea_level_variable == True:                              
        dimensions_df['MSL_x_var'] = (dimensions_df['MLW_x_var']+dimensions_df['MHW_x_var'])/2 # Base MSL on the varying location of the low and high water line
    
    return dimensions_df

def get_intertidal_width_variable(intertidal_width_variable, dimensions_df):
    if intertidal_width_variable == True:
        # Collect info on seaward boundary in dataframe
        dimensions_df['W_intertidal_var'] = dimensions_df['MLW_x_var'] - dimensions_df['MHW_x_var']
        
    return dimensions_df

def get_intertidal_width_fixed(intertidal_width_fixed, dimensions_df):
    if intertidal_width_fixed == True:
        # Collect info on seaward boundary in dataframe
        dimensions_df['W_intertidal_fix'] = dimensions_df['MLW_x_fix'] - dimensions_df['MHW_x_fix']
        
    return dimensions_df

def get_landwardpoint_variance(landward_point_variance, elev_dataframe, dimensions_df):
    if landward_point_variance == True:
        
        ####  Variance method - Sierd de Vries ####
        var_threshold = 0.1 # very dependent on area and range of years!
        var_y = elev_dataframe.var()
        #mean_y = elev_dataframe.mean()
        
        # Gives locations where variance is below threshold
        stable_points = var_y[(var_y < var_threshold)].index
        # Gives locations landward of primary dune
        dunes = elev_dataframe.columns[elev_dataframe.columns < dimensions_df['DuneTop_prim_x'].max()]
            
        try: 
            # Get most seaward stable point that is landward of dunes and with a variance below the threshold
            stable_point = np.intersect1d(stable_points, dunes)[-1]
        except:
            print("No stable point found")
            stable_point = np.nan
        
        # add info on landward boundary to dataframe
        dimensions_df['Landward_x_variance'] = stable_point
        
    return dimensions_df
        
def get_landwardpoint_derivative(landward_point_derivative, elev_dataframe, dimensions_df):
    if landward_point_derivative == True:
        
        ####  Derivative method - Diamantidou ####
        ###################################
        # Get landward boundary from peaks in profile
        from scipy.signal import find_peaks
        
        height_of_peaks = 2.4 #m
        height_constraint = 6.0 #m
        MHW = dimensions_df['MHW_y_var'].values[0]
        peaks_threshold = height_of_peaks + MHW
        
        for yr, row in elev_dataframe.iterrows():
            peaks = find_peaks(row, prominence = height_of_peaks)[0] # Documentation see get_dune_top
            
            peaks_df = row.iloc[peaks]
            peaks_df_filt = peaks_df[peaks_df >= peaks_threshold]
            
            if len(peaks_df) != 0 and peaks_df.values.max() > height_constraint:
                intersections_derivative = find_intersections(row, height_constraint)
                if len(intersections_derivative) != 0:
                    dimensions_df.loc[yr, 'Landward_der_x'] = intersections_derivative[-1]
            elif len(peaks_df_filt) != 0:
                dimensions_df.loc[yr, 'Landward_x_der'] = peaks_df_filt.index[-1]
            else:
                dimensions_df.loc[yr, 'Landward_x_der'] = np.nan
                        
    return dimensions_df
                        
def get_landwardpoint_bma(landward_point_bma, elev_dataframe, dimensions_df):
    if landward_point_bma == True:
        
        ####       Bma calculation     ####
        ###################################
        # Calculating the approximate boundary between the marine and aeolian zone.
        # Based on De Vries et al, 2010, published in Coastal Engeineering.
    
        bma_y = 2.0
    
        for yr, row in elev_dataframe.iterrows(): 
            intersections_bma = find_intersections(row, bma_y)
            if len(intersections_bma) != 0:
                dimensions_df.loc[yr, 'Landward_x_bma'] = intersections_bma[-1]
            else:
                dimensions_df.loc[yr, 'Landward_x_bma'] = np.nan
                    
    return dimensions_df

def get_seawardpoint_foreshore(seaward_point_foreshore, elev_dataframe, dimensions_df):
    if seaward_point_foreshore == True:
    
        seaward_FS_y = -4.0
    
        for yr, row in elev_dataframe.iterrows(): 
            intersections_FS = find_intersections(row, seaward_FS_y)
            if len(intersections_FS) != 0:
                dimensions_df.loc[yr, 'Seaward_x_FS'] = intersections_FS[-1]
            else:
                dimensions_df.loc[yr, 'Seaward_x_FS'] = np.nan
                
    return dimensions_df
        
def get_seawardpoint_activeprofile(seaward_point_activeprofile, elev_dataframe, dimensions_df):
    if seaward_point_activeprofile == True:    
        
        seaward_ActProf_y = -8.0
    
        for yr, row in elev_dataframe.iterrows(): 
            intersections_AP = find_intersections(row, seaward_ActProf_y)
            if len(intersections_AP) != 0:
                dimensions_df.loc[yr, 'Seaward_x_AP'] = intersections_AP[-1]
            else:
                dimensions_df.loc[yr, 'Seaward_x_AP'] = np.nan
    
    return dimensions_df

def get_dune_foot_fixed(dune_foot_fixed, elev_dataframe, dimensions_df):
    if dune_foot_fixed == True:
        
        #### Fixed dunefoot definition ####
        DF_fixed_y = 3 # in m above reference datum
    
        for yr, row in elev_dataframe.iterrows(): 
            intersections_DF = find_intersections(row, DF_fixed_y)
            if len(intersections_DF) != 0:
                dimensions_df.loc[yr, 'Dunefoot_x_fix'] = intersections_DF[-1]
            else:
                dimensions_df.loc[yr, 'Dunefoot_x_fix'] = np.nan
    
    return dimensions_df
            
def get_dune_foot_derivative(dune_foot_derivative, elev_dataframe, dimensions_df):
    if dune_foot_derivative == True:        
        
        ####  Derivative method - E.D. ####
        ###################################
        ## Variable dunefoot definition based on first and second derivative of profile

        threshold_zeroes = 5
        threshold_firstder = 0.001
        threshold_secondder = 0.01
        
        for yr, row in elev_dataframe.iterrows(): 
            # Get seaward boundary
            seaward_x = dimensions_df['MHW_x_fix'].values[0]
            # Get landward boundary 
            landward_x = dimensions_df.loc[yr, 'Landward_x_der']   
        
            # Only keep elevation values within boundaries
            row = row.drop(row.index[row.index > seaward_x]) # drop everything seaward of seaward boundary
            row = row.drop(row.index[row.index < landward_x]).interpolate() # drop everything landward of landward boundary and interpolate remaining data
            
            if len(row) != 0:
                # Get first derivative
                y_der1_index = row.index
                y_der1 = np.gradient(row.values, y_der1_index)  
                # Set first derivative values between -0.001 and 0.001 to zero
                for n in range(len(y_der1)):
                    if abs(y_der1[n]) <= threshold_firstder:
                        y_der1[n] = 0
                
                # Get locations where long sequences of zeroes occur
                zero_sec1 = zero_runs(y_der1, threshold_zeroes)
                
                if len(zero_sec1) != 0:
                    # The profile seaward of and including the most landward sequence of zeroes is deleted
                    y_der1 = np.array_split(y_der1, zero_sec1[0,0])[0]
                    y_der1_index = np.array_split(row.index, zero_sec1[0,0])[0]
                
                # Get second derivative
                y_der2 = np.gradient(y_der1, y_der1_index)    
            
                # Set second derivative values above 0.01 to zero
                y_der2[y_der2 < threshold_secondder] = 0
                
                # Get locations where long sequences of zeroes occur
                zero_sec2 = zero_runs(y_der2, threshold_zeroes)
                
                if len(zero_sec2) != 0:
                    # The profile seaward of and including the most landward sequence of zeroes is set to zero
                    y_der2[zero_sec2[0,0]:] = 0
                        
                # Locations where 2nd derivative is above the threshold:
                dunefoot_locs = y_der1_index[y_der2 > threshold_secondder]
                    
                # Get most seaward point where the above condition is true
                if len(dunefoot_locs) != 0:
                    dimensions_df.loc[yr, 'Dunefoot_x_der'] = dunefoot_locs[-1]
                    dimensions_df.loc[yr, 'Dunefoot_y_der'] = row.loc[dunefoot_locs[-1]]
                else:
                    dimensions_df.loc[yr, 'Dunefoot_x_der'] = np.nan
                    dimensions_df.loc[yr, 'Dunefoot_y_der'] = np.nan
            else:
                dimensions_df.loc[yr, 'Dunefoot_x_der'] = np.nan
                dimensions_df.loc[yr, 'Dunefoot_y_der'] = np.nan
                
    return dimensions_df
                    
def get_dune_foot_pybeach(dune_foot_pybeach, elev_dataframe, dimensions_df):
    if dune_foot_pybeach == True:
        
        from pybeach.beach import Profile
        
        ####  Pybeach methods ####
        ###################################
        
        for yr, row in elev_dataframe.iterrows(): 
                
            # Get seaward boundary
            seaward_x = dimensions_df['MHW_x_fix'].values[0]
            # Get landward boundary 
            landward_x = dimensions_df.loc[yr, 'DuneTop_prim_x']     

            # Remove everything outside of boundaries
            row = row.drop(row.index[row.index > seaward_x]) # drop everything seaward of seaward boundary
            row = row.drop(row.index[row.index < landward_x]).interpolate() # drop everything landward of landward boundary and interpolate remaining data
            
            if np.isnan(sum(row)) == False and len(row) > 5:
                x_ml = np.array(row.index) # pybeach asks ndarray, so convert with np.array(). Note it should be land-left, sea-right otherwise use np.flip()
                y_ml = np.array(row.values) 
                
                p = Profile(x_ml, y_ml)
                toe_ml, prob_ml = p.predict_dunetoe_ml('mixed_clf')  # predict toe using machine learning model
                
                dimensions_df.loc[yr, 'Dunefoot_pybeach_mix_y'] = y_ml[toe_ml[0]]
                dimensions_df.loc[yr, 'Dunefoot_pybeach_mix_x'] = x_ml[toe_ml[0]]
            else:
                dimensions_df.loc[yr, 'Dunefoot_pybeach_mix_y'] = np.nan
                dimensions_df.loc[yr, 'Dunefoot_pybeach_mix_x'] = np.nan
                
    return dimensions_df

def get_beach_width_fix(beach_width_fix, elev_dataframe, dimensions_df):
    if beach_width_fix == True:
        dimensions_df['Beach_width_fix'] = dimensions_df['MSL_x'] - dimensions_df['Dunefoot_x_fix']
    
    return dimensions_df

def get_beach_width_var(beach_width_var, elev_dataframe, dimensions_df):
    if beach_width_var == True:
        dimensions_df['Beach_width_var'] = dimensions_df['MSL_x_var'] - dimensions_df['Dunefoot_x_fix']
    
    return dimensions_df
        
def get_beach_width_der(beach_width_der, elev_dataframe, dimensions_df):
    if beach_width_der == True:
        dimensions_df['Beach_width_der'] = dimensions_df['MSL_x'] - dimensions_df['Dunefoot_x_der'] 

    return dimensions_df

def get_beach_width_der_var(beach_width_der_var, elev_dataframe, dimensions_df):
    if beach_width_der_var == True:
        dimensions_df['Beach_width_der_var'] = dimensions_df['MSL_x_var'] - dimensions_df['Dunefoot_x_der'] 

    return dimensions_df
    
def get_beach_gradient_fix(beach_gradient_fix, elev_dataframe, dimensions_df):
    if beach_gradient_fix == True:   
        
        # dimensions used as landward and seaward boundary
        seaward_bound = 'MSL_x'
        landward_bound = 'Dunefoot_x_fix'
            
        dimensions_df['Beach_gradient_fix'] = get_gradient(elev_dataframe, dimensions_df, seaward_bound, landward_bound)
        
    return dimensions_df

def get_beach_gradient_var(beach_gradient_var, elev_dataframe, dimensions_df):
    if beach_gradient_var == True:   
        
        # dimensions used as landward and seaward boundary
        seaward_bound = 'MSL_x_var'
        landward_bound = 'Dunefoot_x_fix'
        
        dimensions_df['Beach_gradient_var'] = get_gradient(elev_dataframe, dimensions_df, seaward_bound, landward_bound)
        
    return dimensions_df

def get_beach_gradient_der(beach_gradient_der, elev_dataframe, dimensions_df):
    if beach_gradient_der == True:   
        
        # dimensions used as landward and seaward boundary
        seaward_bound = 'MSL_x'
        landward_bound = 'Dunefoot_x_der'
        
        dimensions_df['Beach_gradient_der'] = get_gradient(elev_dataframe, dimensions_df, seaward_bound, landward_bound)
    
    return dimensions_df

def get_dune_front_width_prim_fix(dune_front_width_prim_fix, elev_dataframe, dimensions_df):
    if dune_front_width_prim_fix == True: 
        dimensions_df['Dunefront_width_prim_fix'] = dimensions_df['Dunefoot_x_fix'] - dimensions_df['DuneTop_prim_x']
    
    return dimensions_df

def get_dune_front_width_prim_der(dune_front_width_prim_der, elev_dataframe, dimensions_df):
    if dune_front_width_prim_der == True:
        dimensions_df['Dunefront_width_prim_der'] = dimensions_df['Dunefoot_x_der'] - dimensions_df['DuneTop_prim_x']
    
    return dimensions_df
        
def get_dune_front_width_sec_fix(dune_front_width_sec_fix, elev_dataframe, dimensions_df):
    if dune_front_width_sec_fix == True:
        dimensions_df['Dunefront_width_sec_fix'] = dimensions_df['Dunefoot_x_fix'] - dimensions_df['DuneTop_sec_x'] 

    return dimensions_df

def get_dune_front_width_sec_der(dune_front_width_sec_der, elev_dataframe, dimensions_df):
    if dune_front_width_sec_der == True:
        dimensions_df['Dunefront_width_sec_der'] = dimensions_df['Dunefoot_x_der'] - dimensions_df['DuneTop_sec_x']

    return dimensions_df

def get_dune_front_gradient_prim_fix(dune_front_gradient_prim_fix, elev_dataframe, dimensions_df):
    if dune_front_gradient_prim_fix == True:   
        
        # dimensions used as landward and seaward boundary
        seaward_bound = 'DuneTop_prim_x'
        landward_bound = 'Dunefoot_x_fix'
            
        dimensions_df['Dunefront_gradient_prim_fix'] = get_gradient(elev_dataframe, dimensions_df, seaward_bound, landward_bound)
        
    return dimensions_df

def get_dune_front_gradient_prim_der(dune_front_gradient_prim_der, elev_dataframe, dimensions_df):
    if dune_front_gradient_prim_der == True:   
        
        # dimensions used as landward and seaward boundary
        seaward_bound = 'DuneTop_prim_x'
        landward_bound = 'Dunefoot_x_der'
            
        dimensions_df['Dunefront_gradient_prim_der'] = get_gradient(elev_dataframe, dimensions_df, seaward_bound, landward_bound)
        
    return dimensions_df

def get_dune_front_gradient_sec_fix(dune_front_gradient_sec_fix, elev_dataframe, dimensions_df):
    if dune_front_gradient_sec_fix == True:   
        
        # dimensions used as landward and seaward boundary
        seaward_bound = 'DuneTop_sec_x'
        landward_bound = 'Dunefoot_x_fix'
            
        dimensions_df['Dunefront_gradient_sec_fix'] = get_gradient(elev_dataframe, dimensions_df, seaward_bound, landward_bound)
        
    return dimensions_df

def get_dune_front_gradient_sec_der(dune_front_gradient_sec_der, elev_dataframe, dimensions_df):
    if dune_front_gradient_sec_der == True:   
        
        # dimensions used as landward and seaward boundary
        seaward_bound = 'DuneTop_sec_x'
        landward_bound = 'Dunefoot_x_der'
            
        dimensions_df['Dunefront_gradient_sec_der'] = get_gradient(elev_dataframe, dimensions_df, seaward_bound, landward_bound)
        
    return dimensions_df

def get_dune_volume_fix(dune_volume_fix, elev_dataframe, dimensions_df):
    if dune_volume_fix == True:
        
        # dimensions used as landward and seaward boundary
        seaward_bound = 'Dunefoot_x_fix'
        landward_bound = 'Landward_x_variance'
        
        dimensions_df['DuneVol_fix'] = get_volume(elev_dataframe, dimensions_df, seaward_bound, landward_bound)
    
    return dimensions_df
    
def get_dune_volume_der(dune_volume_der, elev_dataframe, dimensions_df):
    if dune_volume_der == True:
        
        # dimensions used as landward and seaward boundary
        seaward_bound = 'Dunefoot_x_der'
        landward_bound = 'Landward_x_variance'
        
        dimensions_df['DuneVol_der'] = get_volume(elev_dataframe, dimensions_df, seaward_bound, landward_bound)
    
    return dimensions_df
    
def get_intertidal_gradient(intertidal_gradient, elev_dataframe, dimensions_df):
    if intertidal_gradient == True:   
        
        # dimensions used as landward and seaward boundary
        seaward_bound = 'MLW_x_fix'
        landward_bound = 'MHW_x_fix'
            
        dimensions_df['Intertidal_gradient'] = get_gradient(elev_dataframe, dimensions_df, seaward_bound, landward_bound)
        
    return dimensions_df
    
def get_intertidal_volume_fix(intertidal_volume_fix, elev_dataframe, dimensions_df):
    if intertidal_volume_fix == True:
        
        # dimensions used as landward and seaward boundary
        seaward_bound = 'MLW_x_fix'
        landward_bound = 'MHW_x_fix'
        
        dimensions_df['Intertidal_volume_fix'] = get_volume(elev_dataframe, dimensions_df, seaward_bound, landward_bound)
    
    return dimensions_df

def get_intertidal_volume_var(intertidal_volume_var, elev_dataframe, dimensions_df):
    if intertidal_volume_var == True:
        
        # dimensions used as landward and seaward boundary
        seaward_bound = 'MLW_x_var'
        landward_bound = 'MHW_x_var'
        
        dimensions_df['Intertidal_volume_var'] = get_volume(elev_dataframe, dimensions_df, seaward_bound, landward_bound)
    
    return dimensions_df

def get_foreshore_gradient(foreshore_gradient, elev_dataframe, dimensions_df):
    if foreshore_gradient == True:   
        
        # dimensions used as landward and seaward boundary
        seaward_bound = 'Seaward_x_FS'
        landward_bound = 'Landward_x_bma'
            
        dimensions_df['Foreshore_gradient'] = get_gradient(elev_dataframe, dimensions_df, seaward_bound, landward_bound)
        
    return dimensions_df

def get_foreshore_volume(foreshore_volume, elev_dataframe, dimensions_df):
    if foreshore_volume == True:
        
        # dimensions used as landward and seaward boundary
        seaward_bound = 'Seaward_x_FS'
        landward_bound = 'Landward_x_bma'
        
        dimensions_df['Foreshore_volume'] = get_volume(elev_dataframe, dimensions_df, seaward_bound, landward_bound)
    
    return dimensions_df

def get_active_profile_gradient(active_profile_gradient, elev_dataframe, dimensions_df):
    if active_profile_gradient == True:   
        
        # dimensions used as landward and seaward boundary
        seaward_bound = 'Seaward_x_AP'
        landward_bound = 'Landward_x_bma'
            
        dimensions_df['Active_profile_gradient'] = get_gradient(elev_dataframe, dimensions_df, seaward_bound, landward_bound)
        
    return dimensions_df

def get_active_profile_volume(active_profile_volume, elev_dataframe, dimensions_df):
    if active_profile_volume == True:
        
        # dimensions used as landward and seaward boundary
        seaward_bound = 'Seaward_x_AP'
        landward_bound = 'Landward_x_bma'
        
        dimensions_df['Active_profile_volume'] = get_volume(elev_dataframe, dimensions_df, seaward_bound, landward_bound)
    
    return dimensions_df

def save_dimensions_dataframe(dimensions_df, DirDF):
    # Save dataframe for each transect.
    # Later these can all be loaded to calculate averages for specific sites/sections along the coast
    trsct = str(dimensions_df['transect'].iloc[0])
    dimensions_df.to_pickle(DirDF + 'Transect_' + trsct + '_dataframe.pickle')
