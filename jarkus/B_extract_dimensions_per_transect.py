# -*- coding: utf-8 -*-
"""
Created on Mon Jul 29 11:20:42 2019

@author: cijzendoornvan
"""
##################################
####          PACKAGES        ####
##################################
import numpy as np
import pandas as pd
from scipy.interpolate import griddata
from scipy.signal import find_peaks
import time

from analysis_functions import get_volume, get_gradient, find_intersections
from jarkus.transects import Transects

#%%
##################################
####       RETRIEVE DATA      ####
##################################
# Collect the JARKUS data from the server
#Jk = Transects(url='http://opendap.deltares.nl/thredds/catalog/opendap/rijkswaterstaat/jarkus/profiles/catalog.html?dataset=varopendap/rijkswaterstaat/jarkus/profiles/transect_r20190731.nc')
Jk = Transects(url='https://opendap.tudelft.nl/thredds/dodsC/data2/deltares/rijkswaterstaat/jarkus/profiles/transect_r20180914.nc')
ids = Jk.get_data('id') # ids of all available transects

#%%
##################################
####    USER-DEFINED REQUEST  ####
##################################
# Select which dimensions should be calculated from the transects
dune_height_and_location = True
mean_sea_level      = True
mean_low_water      = True
mean_high_water     = True
intertidal_width    = True
landward_point      = True
seaward_point       = True
dune_foot_location  = True
beach_width         = True
beach_gradient      = True
dune_front_width    = True
dune_gradient       = True
dune_volume         = True
intertidal_gradient = True
intertidal_volume   = True
foreshore_gradient  = True
foreshore_volume    = True
active_profile_gradient  = True
active_profile_volume    = True

# Set which years should be analysed
years_requested = list(range(1965, 2020))

# Set whether all transect should be analysed or define a retrieval request
execute_all_transects = True

if execute_all_transects == False:
    # Set the transect and years for retrieval request
    transect_name   = "08_Meijendel"
    #transect_req = np.arange(17000011, 17001467, 1)
    transect_req        = [8009325]
    idxs = np.isin(transect_req, ids) # check which transect are available of those that were requested
    ids_filtered = np.array(transect_req)[np.nonzero(idxs)[0]]
    Dir_pickles = "C:/Users/cijzendoornvan/Documents/GitHub/jarkus/jarkus/Figures/" + transect_name + '/'  
else:
    # Filter out location that are not suitable for analysis. Based on Kustlijnkaarten 2019, RWS.
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
    remove15 = list(ids[0:1707]) #cn be removed
    
    remove_all = remove1 + remove2 + remove3 + remove4 + remove5 + remove6 + remove7 + remove8 + remove9 + remove10 + remove11 + remove12 + remove13 + remove14 + remove15
    
    ids_filtered = [x for x in ids if x not in remove_all]
    Dir_pickles = "C:/Users/cijzendoornvan/Documents/GitHub/jarkus/jarkus/Figures/Dataframes_per_transect_try2/"

#%%
##################################
####      PREPROCESS DATA     ####
##################################
years_requested_str = [str(yr) for yr in years_requested]

for idx in ids_filtered: # For each available transect go though the entire analysis
    print(idx)
    start = time.time()
    Dimensions = []
    trsct = str(idx)
    
    # Here the JARKUS filter is set and the data for the requested transect and years is retrieved
    Jk = Transects(url='https://opendap.tudelft.nl/thredds/dodsC/data2/deltares/rijkswaterstaat/jarkus/profiles/transect_r20180914.nc')
    df, years_available = Jk.get_dataframe(idx, years_requested)
    
    # Interpolate x and y along standardized cross shore axis
    cross_shore = list(range(-3000, 9320, 1))
    
    # Convert elevation data for each year of each transect into aray that can be easily analysed
    y_all = [] 
    for i, yr in enumerate(years_requested_str):
        if yr in years_available:
            y = np.array(df.loc[trsct, yr]['y'])
            x = np.array(df.loc[trsct, yr]['x'])
            y_grid = griddata(x,y,cross_shore)
            if i == 0:
                y_all = y_grid
            else:
                y_all = np.column_stack((y_all, y_grid))
                y_grid = griddata(x,y,cross_shore)
        else:
            y_grid = np.empty((len(cross_shore),))
            y_grid[:] = np.nan
            #print(y_grid)
            if i == 0:
                y_all = y_grid
            else:
                y_all = np.column_stack((y_all, y_grid))
        
    # Filter the years based on the min and max elevation. Kick out those that have data missing between -1 and +5 m.
    years_filtered = []
    for i, yr in enumerate(years_available):
        max_y = np.nanmax(y_all[:, i])
        min_y = np.nanmin(y_all[:, i])
        if max_y < 5 or min_y > -1:
            years_filtered.append(np.nan)
        else: 
            years_filtered.append(yr)
    
    Dimensions = pd.DataFrame({'transect': trsct, 'years': years_requested_str})
    Dimensions.set_index('years', inplace=True)
    
    Time1 = time.time()
    print('initialisation ' + str(Time1 - start) + ' seconds')
    
#%%
###################################  
### DEFINE DUNE CREST LOCATION  ###
###################################
    if dune_height_and_location == True:
        
        Dimensions['DT_prim_x'] = np.nan
        Dimensions['DT_prim_y'] = np.nan
        Dimensions['DT_sec_x'] = np.nan
        Dimensions['DT_sec_y'] = np.nan
        
        for i, yr in enumerate(years_requested_str):
            dune_top_prim = find_peaks(y_all[:,i], height = 5, prominence = 2.0) # , distance = 5
            dune_top_sec = find_peaks(y_all[:,i], height = 3, prominence = 0.5) # , distance = 5
            # Documentation:
            # https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.find_peaks.html#scipy.signal.find_peaks
            # https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.peak_prominences.html#scipy.signal.peak_prominences    
            # The prominence of a peak measures how much a peak stands out from the surrounding baseline of the signal and is defined as the vertical distance between the peak and its lowest contour line.
            if yr in years_available and len(dune_top_prim[0]) != 0: # If the year is available and a peak is found in the profile
                # Select the most seaward peak found of the primary and secondary peaks
                dune_top_prim_idx = dune_top_prim[0][-1]
                dune_top_sec_idx = dune_top_sec[0][-1]
                #print(yr, cross_shore[dune_top_prim_idx], cross_shore[dune_top_sec_idx])
                
                if  dune_top_sec_idx <= dune_top_prim_idx: 
                    # If most seaward secondary peak is located at the same place or landward of the most seaward primary peak
                    Dimensions.loc[yr, 'DT_prim_x'] = cross_shore[dune_top_prim_idx]  # Save the primary peak location
                    Dimensions.loc[yr, 'DT_prim_y'] = y_all[dune_top_prim_idx, i]
                    #Assume that there is no seaward secondary peak, so no value filled in (i.e. it stays nan).
                else:            
                    # Otherwise save both the primary and secondary peak location
                    Dimensions.loc[yr, 'DT_prim_x'] = cross_shore[dune_top_prim_idx] 
                    Dimensions.loc[yr, 'DT_prim_y'] = y_all[dune_top_prim_idx, i]
                    Dimensions.loc[yr, 'DT_sec_x'] = cross_shore[dune_top_sec_idx]
                    Dimensions.loc[yr, 'DT_sec_y'] = y_all[dune_top_sec_idx, i]
    else:
        pass
    
    Time2 = time.time()
    print('dune top calculation ' + str(Time2 - Time1) + ' seconds')

#%%    
##################################
####   DEFINE MEAN SEA LEVEL  ####
##################################
    if mean_sea_level == True:
    
        MSL_y       = 0 # in m above reference datum
        Dimensions['MSL_x'] = np.nan
        
        for i, yr in enumerate(years_requested_str): 
            if yr in years_filtered:
                intersect_idx = find_intersections(cross_shore, y_all[:,i], MSL_y)
                if len(intersect_idx[0]) != 0 and np.isnan(Dimensions['DT_prim_x'][i]):
                    intersect_x = np.array([cross_shore[idx] for idx in intersect_idx[0]])
                    Dimensions.loc[yr, 'MSL_x'] = intersect_x[-1]
                elif len(intersect_idx[0]) != 0:
                    intersect_x = np.array([cross_shore[idx] for idx in intersect_idx[0]])
                    intersect_x_sw = intersect_x[intersect_x > Dimensions['DT_prim_x'][i]]
                    if max(intersect_x_sw) - min(intersect_x_sw) > 100:
                        intersect_x_lw = intersect_x_sw[intersect_x_sw < (min(intersect_x_sw) + 100)]
                        Dimensions.loc[yr, 'MSL_x'] = intersect_x_lw[-1]
                    else: 
                        Dimensions.loc[yr, 'MSL_x'] = intersect_x_sw[-1]
    else:
        pass
    
    Time3 = time.time()
    print('MSL calculation ' + str(Time3 - Time2) + ' seconds')
#%%
##################################
####   DEFINE MEAN LOW WATER  ####
##################################
    if mean_low_water == True:
        
        MLW_y_fix   = -1 # in m above reference datum
        Dimensions['MLW_x_fix'] = np.nan
        Dimensions['MLW_x_var'] = np.nan
        Dimensions['MLW_y_var'] = np.nan 
        
        for i, yr in enumerate(years_requested_str): 
            if yr in years_filtered:
                MLW_y_variable   = df.loc[trsct, yr]['mlw'][0]
                intersect_idx_fix = find_intersections(cross_shore, y_all[:,i], MLW_y_fix)
                intersect_idx_var = find_intersections(cross_shore, y_all[:,i], MLW_y_variable)
                if len(intersect_idx_fix[0]) != 0:
                    intersect_x_fix = np.array([cross_shore[idx] for idx in intersect_idx_fix[0]])
                    intersect_x_filt_fix = intersect_x_fix[(intersect_x_fix < Dimensions['MSL_x'][i] + 250)]
                    #print(intersect_x_fix, Dimensions['MSL_x'][i]) #################
                    if len(intersect_x_filt_fix) == 0:
                        Dimensions.loc[yr, 'MLW_x_fix'] = intersect_x_fix[-1]
                    else: 
                        Dimensions.loc[yr, 'MLW_x_fix'] = intersect_x_filt_fix[-1]
                    
                if len(intersect_idx_var[0]) != 0:
                    intersect_x_var = np.array([cross_shore[idx] for idx in intersect_idx_var[0]])
                    intersect_x_filt_var = intersect_x_var[(intersect_x_var < Dimensions['MSL_x'][i] + 250)]
                    if len(intersect_x_filt_var) == 0:
                        Dimensions.loc[yr, 'MLW_x_var'] = intersect_x_var[-1]
                        Dimensions.loc[yr, 'MLW_y_var'] = MLW_y_variable
                    else:
                        Dimensions.loc[yr, 'MLW_x_var'] = intersect_x_filt_var[-1]
                        Dimensions.loc[yr, 'MLW_y_var'] = MLW_y_variable    
    else:
        pass
    
    Time4 = time.time()
    print('MLW calculation ' + str(Time4 - Time3) + ' seconds')  
#%%
##################################
####  DEFINE MEAN HIGH WATER  ####
##################################
    if mean_high_water == True:
    
        MHW_y_fix   = 1 # in m above reference datum
        Dimensions['MHW_x_fix'] = np.nan
        Dimensions['MHW_x_var'] = np.nan
        Dimensions['MHW_y_var'] = np.nan
        
        for i, yr in enumerate(years_requested_str): 
            if yr in years_filtered:
                MHW_y_variable   = df.loc[trsct, yr]['mhw'][0]
                intersect_idx_fix = find_intersections(cross_shore, y_all[:,i], MHW_y_fix)
                intersect_idx_var = find_intersections(cross_shore, y_all[:,i], MHW_y_variable)
                if len(intersect_idx_fix[0]) != 0:
                    intersect_x_fix = np.array([cross_shore[idx] for idx in intersect_idx_fix[0]])
                    intersect_x_filt_fix = intersect_x_fix[(intersect_x_fix > Dimensions['MSL_x'][i] - 250)]
                    if len(intersect_x_filt_fix) == 0:
                        Dimensions.loc[yr, 'MHW_x_fix'] = intersect_x_fix[-1]
                    else:
                        Dimensions.loc[yr, 'MHW_x_fix'] = intersect_x_filt_fix[-1]
                    
                if len(intersect_idx_var[0]) != 0:
                    intersect_x_var = np.array([cross_shore[idx] for idx in intersect_idx_var[0]])
                    intersect_x_filt_var = intersect_x_var[(intersect_x_var > Dimensions['MSL_x'][i] - 250)]
                    if len(intersect_x_filt_var) == 0:                
                        Dimensions.loc[yr, 'MHW_x_var'] = intersect_x_var[-1]
                        Dimensions.loc[yr, 'MHW_y_var'] = MHW_y_variable
                    else:
                        Dimensions.loc[yr, 'MHW_x_var'] = intersect_x_filt_var[-1]
                        Dimensions.loc[yr, 'MHW_y_var'] = MHW_y_variable    
    else:
        pass
    
    Time5 = time.time()
    print('MHW calculation ' + str(Time5 - Time4) + ' seconds')    
#%%
###################################
####CALCULATE INTERTIDAL WIDTH ####
###################################
    if intertidal_width == True:
        # Collect info on seaward boundary in dataframe
        Dimensions['W_intertidal_fix'] = Dimensions['MLW_x_fix'] - Dimensions['MHW_x_fix']
        Dimensions['W_intertidal_var'] = Dimensions['MLW_x_var'] - Dimensions['MHW_x_var']
    else:
        pass

#%%        
###################################
####  DEFINE LANDWARD BOUNDARY ####
###################################
    if landward_point == True:
        
        ####  Variance method - Sierd  ####
        var_threshold = 0.1 # very dependent on area and range of years!
        
        var_y = np.nanvar(y_all, axis=1)
        mean_y = np.nanmean(y_all, axis=1)
        
        # find the first (seaward) location where elevation y_all > 5 and var_y < 0.2
        stable_points_index = [i for i,var in enumerate(var_y) if var < var_threshold]
        try:
            stable_point_idx = [idx for i,idx in enumerate(stable_points_index) if cross_shore[idx] < np.nanmax(Dimensions['DT_prim_x'])][-1]
            stable_x = cross_shore[stable_point_idx]
        except:
            print("No stable point found")
            stable_x = np.nan
        
        # add info on landward boundary to dataframe
        Dimensions['landward_stable_x'] = stable_x
        
        Time6 = time.time()
        print('landward point calculation ' + str(Time6 - Time5) + ' seconds')  
        
        ####  Derivative method - E.D. ####
        ###################################
        # Get landward boundary from peaks in profile
        """ Diamantidou uses more complicated manner to retrieve the landward boundary, 
        however, here only the 6 m threshold is used because all peaks in the studied area are above 6 m.
        threshold_peak = 2.4
        threshold_constr = 6
        y_thr_peak = [y_grid >= threshold_peak][0]
        y_thr_der = [y_der1 < 0.05][0] """
    
        landward_y = 6.0
        Dimensions['landward_6m_x'] = np.nan
    
        for i, yr in enumerate(years_requested_str):
            if yr in years_available:
                intersect = find_intersections(cross_shore, y_all[:,i], landward_y)    
                if len(intersect[0]) != 0:
                    idx = intersect[0][-1]
                    landward_x = cross_shore[idx]    
                    Dimensions.loc[yr, 'landward_6m_x'] = landward_x
        
        ####       Bma calculation     ####
        ###################################
        # Calculating the approximate boundary between the marine and aeolian zone.
        # Based on De Vries et al, 2010, Published in Coastal Engeineering.
        Bma_y = 2.0
        Dimensions['Bma_x'] = np.nan
    
        for i, yr in enumerate(years_requested_str):
            if yr in years_available:
                intersect = find_intersections(cross_shore, y_all[:,i], Bma_y)    
                if len(intersect[0]) != 0:
                    idx = intersect[0][-1]
                    Bma_x = cross_shore[idx]    
                    Dimensions.loc[yr, 'Bma_x'] = Bma_x    
    else:
        pass
#%%    
###################################
####  DEFINE SEAWARD BOUNDARY  ####
###################################
    if seaward_point == True:
    
        seaward_FS_y = -4.0
        Dimensions['seaward_FS_x_all'] = np.nan
    
        for i, yr in enumerate(years_requested_str):
            if yr in years_available:
                intersect = find_intersections(cross_shore, y_all[:,i], seaward_FS_y)    
                if len(intersect[0]) != 0:
                    idx = intersect[0][-1]
                    seaward_FS_x = cross_shore[idx]    
                    Dimensions.loc[yr, 'seaward_FS_x_all'] = seaward_FS_x
        
        seaward_ActProf_y = -8.0
        Dimensions['seaward_ActProf_x_all'] = np.nan
    
        for i, yr in enumerate(years_requested_str):
            if yr in years_available:
                intersect = find_intersections(cross_shore, y_all[:,i], seaward_ActProf_y)    
                if len(intersect[0]) != 0:
                    idx = intersect[0][-1]
                    seaward_ActProf_x = cross_shore[idx]    
                    Dimensions.loc[yr, 'seaward_ActProf_x_all'] = seaward_ActProf_x    
    else:
        pass
#%%
###################################  
#### DEFINE DUNEFOOT LOCATION  ####
###################################
    if dune_foot_location == True:
            
        #### Fixed dunefoot definition ####
        ###################################
        DF_fixed_y = 3 # in m above reference datum
        Dimensions['DF_fix_y'] = np.nan
        Dimensions['DF_fix_x'] = np.nan
        
        for i, yr in enumerate(years_requested_str):
            if yr in years_available:
                intersect = find_intersections(cross_shore, y_all[:,i], DF_fixed_y)
                if len(intersect[0]) != 0:
                    idx = intersect[0][-1]
                    Dimensions.loc[yr, 'DF_fix_x'] = cross_shore[idx]
                    Dimensions.loc[yr, 'DF_fix_y'] = DF_fixed_y
        
        ####  Derivative method - E.D. ####
        ###################################
        ## Variable dunefoot definition based on first and second derivative of profile
        Dimensions['DF_der_y'] = np.nan
        Dimensions['DF_der_x'] = np.nan
        for i, yr in enumerate(years_requested_str):
            if yr in years_available:
                # Get seaward boundary
                seaward_x = Dimensions.loc[yr, 'MHW_x_fix']
                # Get landward boundary 
                landward_x = Dimensions.loc[yr, 'landward_6m_x']   
        
                # Give nan values to everything outside of boundaries
                y_all_domain = []
                for xc in range(len(cross_shore)): 
                    if cross_shore[xc] > seaward_x or cross_shore[xc] < landward_x:
                        y_all_domain.append(np.nan)
                    else:
                        y_all_domain.append(y_all[xc,i])
            
                # Give nan to years where not enough measurements are available within domain
                if np.count_nonzero(~np.isnan(y_all_domain)) > 30:
                    # Get first and second derivative
                    y_der1 = np.gradient(y_all_domain, cross_shore)    
                    y_der2 = np.gradient(y_der1, cross_shore)    
                
                    # Set first derivative values between -0.001 and 0.001 to zero
                    # Set second derivative values above 0.01 to zero
                    for n in range(len(y_der1)):
                        if -0.001 <= y_der1[n] <= 0.001:
                            y_der1[n] = 0
                        if y_der2[n] >= 0.01:
                            y_der2[n] = 0
                            
                    # Set to values to nan, where consecutive zeros occur in the first derivative
                    y_der1_clean = np.zeros(len(y_der1))
                    y_der1_clean[:] = np.nan
                    y_der_remove = False
                    for j in range(len(y_der1)-1):
                        if y_der1[j] == 0 and y_der1[j] == y_der1[j+1]:
                            y_der1_clean[j] = np.nan
                            y_der_remove = True
                        elif y_der_remove == True:
                            y_der1_clean[j] = np.nan
                            y_der_remove = False
                        else:
                            y_der1_clean[j] = y_der1[j]
                            
                    # Set locations to True where both first and second derivative are zero
                    dunefoot = np.zeros(len(y_der1))
                    for l in range(len(y_der1)):
                        if y_der2[l] == 0:
                            dunefoot[l] = True
                        
                    # Get most seaward point where the above condition is True
                    if sum(dunefoot) != 0:
                        dunefoot_idx = np.where(dunefoot == True)[0][0]
                        Dimensions.loc[yr, 'DF_der_y'] = cross_shore[dunefoot_idx]
                        Dimensions.loc[yr, 'DF_der_x'] = y_all[dunefoot_idx, i]
    else:
        pass
    
    Time7 = time.time()
    print('dune foot ' + str(Time7 - Time6) + ' seconds')   
#%%    
###################################  
###      CALC BEACH WIDTH       ###
###################################
    if beach_width == True:
        # Get landward boundary for varying water line
        B_seaward_var = (Dimensions['MLW_x_var']+Dimensions['MHW_x_var'])/2 # Base beach width on the varying location of the low and high water line
        
        Dimensions['BW_fix'] = Dimensions['MSL_x'] - Dimensions['DF_fix_x']
        Dimensions['BW_var'] = B_seaward_var - Dimensions['DF_fix_x']
        Dimensions['BW_der'] = Dimensions['MSL_x'] - Dimensions['DF_der_x']       
    else:
        pass
    
    Time8 = time.time()
    print('beach width calculation ' + str(Time8 - Time7) + ' seconds')   
#%%    
###################################  
###     CALC BEACH GRADIENT     ###
###################################
    if beach_gradient == True:        
        Dimensions['B_grad_fix'] = get_gradient(cross_shore, y_all, years_requested, Dimensions['MSL_x'], Dimensions['DF_fix_x'])
        #Dimensions['B_grad_var'] = get_gradient(cross_shore, y_all, years_requested, B_seaward_var, Dimensions['DF_fix_x'])
        #Dimensions['B_grad_der'] = get_gradient(cross_shore, y_all, years_requested, Dimensions['MSL_x'], Dimensions['DF_der_x'])
    else:
        pass
    
    Time9 = time.time()
    print('beach gradient calculation ' + str(Time9 - Time8) + ' seconds')   

#%%    
###################################  
####  CALC DUNE FRONT WIDTH    ####
###################################
    if dune_front_width == True:        
        # Calculate the width of the dune front that corresponds to the dune front gradient
        Dimensions['DFront_fix_prim_W'] = Dimensions['DF_fix_x'] - Dimensions['DT_prim_x']
        Dimensions['DFront_der_prim_W'] = Dimensions['DF_der_x'] - Dimensions['DT_prim_x']
        Dimensions['DFront_fix_sec_W'] = Dimensions['DF_fix_x'] - Dimensions['DT_sec_x']
        Dimensions['DFront_der_sec_W'] = Dimensions['DF_der_x'] - Dimensions['DT_sec_x']    
    else:
        pass
    Time10 = time.time()
    print('dune width calculation ' + str(Time10 - Time9) + ' seconds')   
    
#%%    
###################################  
###      CALC DUNE GRADIENT     ###
###################################
    if dune_gradient == True:
        Dimensions['DFront_fix_prim_grad'] = get_gradient(cross_shore, y_all, years_requested, Dimensions['DF_fix_x'], Dimensions['DT_prim_x'])
        #Dimensions['DFront_der_prim_grad'] = get_gradient(cross_shore, y_all, years_requested, Dimensions['DF_der_x'], Dimensions['DT_prim_x'])
        Dimensions['DFront_fix_sec_grad'] = get_gradient(cross_shore, y_all, years_requested, Dimensions['DF_fix_x'], Dimensions['DT_sec_x'])
        #Dimensions['DFront_der_sec_grad'] = get_gradient(cross_shore, y_all, years_requested, Dimensions['DF_der_x'], Dimensions['DT_sec_x'])
    else:
        pass
    
    Time11 = time.time()
    print('dune gradient calculation ' + str(Time11 - Time10) + ' seconds')  

#%%    
###################################
###    CALC DUNE VOLUME    ### 
###################################
    if dune_volume == True:
        Dimensions['DVol_fix'] = get_volume(cross_shore, y_all, years_requested, Dimensions['DF_fix_x'], Dimensions['landward_stable_x'])
        Dimensions['DVol_der'] = get_volume(cross_shore, y_all, years_requested, Dimensions['DF_der_x'], Dimensions['landward_stable_x']) 
    else:
        pass
    
    Time12 = time.time()
    print('dune volume calculation ' + str(Time12 - Time11) + ' seconds')   
    
#%%    
###################################
###   CALC INTERTIDAL GRADIENT  ### 
###################################
    if intertidal_gradient == True:        
        Dimensions['Int_grad'] = get_gradient(cross_shore, y_all, years_requested, Dimensions['MLW_x_fix'], Dimensions['MHW_x_fix'])
    else:
        pass
    
    Time13 = time.time()
    print('intertidal gradient calculation ' + str(Time13 - Time12) + ' seconds')  

#%%    
###################################
####   CALC INTERTIDAL VOLUME  #### 
###################################
    if intertidal_volume == True:        
        Dimensions['IntVol_fix'] = get_volume(cross_shore, y_all, years_requested, Dimensions['MLW_x_fix'], Dimensions['MHW_x_fix'])
        Dimensions['IntVol_var'] = get_volume(cross_shore, y_all, years_requested, Dimensions['MLW_x_var'], Dimensions['MHW_x_var'])        
    else:
        pass
    
    Time14 = time.time()
    print('intertidal volume calculation ' + str(Time14 - Time13) + ' seconds')   

#%%    
###################################
###   CALC FORESHORE GRADIENT   ### 
###################################
    if foreshore_gradient == True:        
        Dimensions['FS_grad'] = get_gradient(cross_shore, y_all, years_requested, Dimensions['seaward_FS_x_all'], Dimensions['Bma_x'])
    else:
        pass
    
    Time15 = time.time()
    print('foreshore gradient calculation ' + str(Time15 - Time14) + ' seconds')  

#%%    
###################################
###    CALC FORESHORE VOLUME    ### 
###################################
    if foreshore_volume == True:    
        Dimensions['FSVol_fix'] = get_volume(cross_shore, y_all, years_requested, Dimensions['seaward_FS_x_all'], Dimensions['Bma_x'])    
    else:
        pass
    
    Time16 = time.time()
    print('foreshore volume calculation ' + str(Time16 - Time15) + ' seconds')  

#%%
###################################
### CALC ACTIVE PROFILE GRADIENT### 
###################################
    if active_profile_gradient == True:        
        Dimensions['AP_grad'] = get_gradient(cross_shore, y_all, years_requested, Dimensions['seaward_ActProf_x_all'], Dimensions['Bma_x'])
    else:
        pass
    
    Time17 = time.time()
    print('active profile gradient calculation ' + str(Time17 - Time16) + ' seconds')  
    
#%%    
###################################
###  CALC ACTIVE PROFILE VOLUME ### 
###################################
    if active_profile_volume == True:
        Dimensions['APVol_fix'] = get_volume(cross_shore, y_all, years_requested, Dimensions['seaward_ActProf_x_all'], Dimensions['Bma_x'])
        Dimensions['APVol_var'] = get_volume(cross_shore, y_all, years_requested, Dimensions['seaward_ActProf_x_all'], Dimensions['Bma_x'])
    else:
        pass
    
    Time18 = time.time()
    print('active profile volume calculation ' + str(Time18 - Time17) + ' seconds')  
    
    
    end = time.time()
    time_elapsed = str(end-start)
    print('Elapsed time for transect ' + trsct + ' is ' + time_elapsed + ' seconds')

#%%    
###################################
###        SAVE DATAFRAME       ### 
###################################
    # Save dataframe for each transect.
    # Later these can all be loaded to calculate averages for specific sites/sections along the coast
    Dimensions.to_pickle(Dir_pickles + 'Transect_' + trsct + '_dataframe.pickle')
    
