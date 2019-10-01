# -*- coding: utf-8 -*-
"""
Created on Mon Jul 29 11:20:42 2019

@author: cijzendoornvan
"""

#################################
####        FUNCTIONS        ####
#################################

def get_x_from_y(x, y, y_req):
    f = interp1d(y,x, kind='linear')
    x_req = f(y_req)
    return x_req

def expand_2_axis(complete_axis, x_values, y_values):
    y_expanded = [np.nan] * len(complete_axis)
    for j in range(len(complete_axis)):
        if complete_axis[j] in x_values:
            index = np.where(x_values == complete_axis[j])
            y_expanded[j] = y_values[index]
    return y_expanded

def get_volume(x, y, years_included, dune_foot_x, landward_bound_x):
    volume = []
    dune_idx = []
    for i, yr in enumerate(years_included):
        DF_x = np.ceil(dune_foot_x[i])
        LWB_x = np.floor(landward_bound_x[i])
        if np.isnan(landward_bound_x[i]) or np.isnan(dune_foot_x[i]):
            volume.append(np.nan)
        else:
            dune_idx_DF = np.where(x <= DF_x)
            dune_idx_LWB = np.where(x >= LWB_x)
            dune_idx = [value for value in dune_idx_DF[0] if value in dune_idx_LWB[0]]
            dune_x = [x[value] for value in dune_idx_DF[0] if value in dune_idx_LWB[0]]
            if LWB_x in dune_x == False or DF_x in dune_x == False:
                volume.append(np.nan)
            else:
                x_dune = [x[j] for j in dune_idx]
                y_dune = [y[k,i] for k in dune_idx]
                volume_trapz = np.trapz(y_dune, x = x_dune)
                volume.append(volume_trapz)
    
    return volume

def get_gradient(x, y, years_included, seaward_bound, landward_bound):
    gradient = []
    for i, yr in enumerate(years_included):
        # Extract elevation profile with seaward and landward boundaries
        elevation_y = []
        cross_shore_x = []
        
        for xc in range(len(x)): 
            if x[xc] < seaward_bound[i] and x[xc] > landward_bound[i] and np.isnan(y[xc,i]) == False:
                elevation_y.append(y[xc,i])
                cross_shore_x.append(cross_shore[xc])
        # Calculate gradient for domain
        if cross_shore_x == []:
            gradient.append(np.nan)
        else:
            gradient.append(np.polyfit(cross_shore_x, elevation_y, 1)[0])
        
    return gradient

def find_intersections(x, y, years_included, y_value):
    value_vec = []
    for x_val in range(len(x)):
        value_vec.append(y_value)
    
    diff = np.nan_to_num(np.diff(np.sign(y - value_vec)))
    intersections = np.nonzero(diff)
    
    return intersections

##################################
####          PACKAGES        ####
##################################
#%matplotlib auto
from jarkus.transects import Transects
import math
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.interpolate import interp1d, griddata
from scipy.signal import find_peaks

##################################
####    USER-DEFINED REQUEST  ####
##################################

# Set the transect and years for retrieval request
transect_name   = "04_Terschelling"
transect        = [4000160]
years_requested = list(range(1965, 2020))

##################################
####  REOPEN TRANSECT FIGURE  ####
##################################
# Execute %matplotlib auto first, otherwise you get an error
from visualisation import reopen_pickle
fig_transect = str(transect[0])
Dir_fig = "C:\\Users\\cijzendoornvan\\Documents\\GitHub\\jarkus\\jarkus\\Figures\\" + transect_name + "\\"

reopen_pickle(Dir_fig, fig_transect)

##################################
#### RETRIEVE/PREPROCESS DATA ####
##################################

# Collect the JARKUS data from the server
#Jk = Transects(url='http://opendap.deltares.nl/thredds/dodsC/opendap/rijkswaterstaat/jarkus/profiles/transect_r20180914.nc')
Jk = Transects(url='https://opendap.tudelft.nl/thredds/dodsC/data2/deltares/rijkswaterstaat/jarkus/profiles/transect_r20180914.nc')
# Here the JARKUS filter is set and the data for the requested transects and years is retrieved
df, years_included = Jk.get_dataframe(transect, years_requested)    

# Here a loop could be build in to do the procedure for multiple transects
trsct = str(transect[0])

# Interpolate x and y along standardized cross shore axis
cross_shore = list(range(-3000, 9320, 1))

y_all = []
for i, yr in enumerate(years_included):
    y = np.array(df.loc[trsct, yr]['y'])
    x = np.array(df.loc[trsct, yr]['x'])
    y_grid = griddata(x,y,cross_shore)
    if i == 0 :
        y_all = y_grid
    else:
        y_all = np.column_stack((y_all, y_grid))

# Filter the years based on the min and max elevation.
years       = []
for i, yr in enumerate(years_included):
    max_y = np.nanmax(y_all[:, i])
    min_y = np.nanmin(y_all[:, i])
    if max_y < 5 or min_y > -1:
        years.append(np.nan)
    else: 
        years.append(yr)
        
##################################
####   DEFINE MEAN SEA LEVEL  ####
##################################

MSL_y       = 0 # in m above reference datum
MSL_x = []

for i, yr in enumerate(years):
    if math.isnan(float(yr)):
        MSL_x.append(np.nan)
    else: 
        intersect = find_intersections(cross_shore, y_all[:,i], years_included, MSL_y)        
        print(intersect)
        if len(intersect[0]) == 0:
            MSL_x.append(np.nan)    
        else:
            idx = intersect[0][-1]
            MSL_x.append(cross_shore[idx])
    
Dimensions = pd.DataFrame({'transect': trsct, 'years': years_included, 'MSL_x': MSL_x})

##################################
####  DEFINE SEAWARD BOUNDARY ####
##################################
        

MHW_y_fix   = 1
MLW_y_fix   = -1
offshore_y  = -6

MSL_x       = []
MHW_x_fix   = []
MLW_x_fix   = []
offshore_x  = []
MHW_x_var   = []
MHW_y_var   = []
MLW_x_var   = []
MLW_y_var   = []

for i, yr in enumerate(years):
    if math.isnan(float(yr)):                
        MSL_x.append(np.nan)
        MHW_x_fix.append(np.nan)
        MLW_x_fix.append(np.nan)
        MHW_x_var.append(np.nan)
        MHW_y_var.append(np.nan)
        MLW_x_var.append(np.nan)
        MLW_y_var.append(np.nan)
    else:       
        MHW_y_variable   = df.loc[trsct, yr]['mhw'][0]
        MLW_y_variable   = df.loc[trsct, yr]['mlw'][0]
        
        MSL_x.append(get_x_from_y(cross_shore, y_all[:,i], MSL_y))
        MHW_x_fix.append(get_x_from_y(cross_shore, y_all[:,i], MHW_y_fix))
        MLW_x_fix.append(get_x_from_y(cross_shore, y_all[:,i], MLW_y_fix))
        MHW_x_var.append(get_x_from_y(cross_shore, y_all[:,i], MHW_y_variable))
        MHW_y_var.append(MHW_y_variable)
        MLW_x_var.append(get_x_from_y(cross_shore, y_all[:,i], MLW_y_variable))
        MLW_y_var.append(MHW_y_variable)
        
# Collect info on seaward boundary in dataframe
W_intertidal_fix = np.subtract(MLW_x_fix, MHW_x_fix)
W_intertidal_var = np.subtract(MLW_x_var, MHW_x_var)
Dimensions = pd.DataFrame({'transect': trsct, 'years': years_included, 'MSL_x': MSL_x, 'MHW_x_fix': MHW_x_fix, 'MLW_x_fix': MLW_x_fix, 'MHW_x_var': MHW_x_var, 'MHW_y_var': MHW_y_var, 'MLW_x_var': MLW_x_var, 'MLW_y_var': MLW_y_var, 'W_intertidal_fix': W_intertidal_fix, 'W_intertidal_var': W_intertidal_var})

###################################  
### DEFINE DUNE CREST LOCATION  ###
###################################

dune_top_prim_x = []
dune_top_prim_y = []
dune_top_sec_x = []
dune_top_sec_y = []
for i, yr in enumerate(years_included):
    dune_top_prim = find_peaks(y_all[:,i], height = 5, prominence = 2.0) # , distance = 5
    dune_top_sec = find_peaks(y_all[:,i], height = 3, prominence = 0.5) # , distance = 5
    # Documentation:
    # https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.find_peaks.html#scipy.signal.find_peaks
    # https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.peak_prominences.html#scipy.signal.peak_prominences    
    # The prominence of a peak measures how much a peak stands out from the surrounding baseline of the signal and is defined as the vertical distance between the peak and its lowest contour line.
    if  len(dune_top_prim[0]) == 0:
        dune_top_prim_x.append(np.nan)
        dune_top_prim_y.append(np.nan)
        dune_top_sec_x.append(np.nan)
        dune_top_sec_y.append(np.nan)
    else:
        dune_top_prim_idx = dune_top_prim[0][-1]
        dune_top_sec_idx = dune_top_sec[0][-1]
        #print(yr, cross_shore[dune_top_prim_idx], cross_shore[dune_top_sec_idx])
        
        if  dune_top_sec_idx <= dune_top_prim_idx:
            dune_top_prim_x.append(cross_shore[dune_top_prim_idx])
            dune_top_prim_y.append(y_all[dune_top_prim_idx, i])
            dune_top_sec_x.append(np.nan)
            dune_top_sec_y.append(np.nan)
        else:            
            dune_top_prim_x.append(cross_shore[dune_top_prim_idx])
            dune_top_prim_y.append(y_all[dune_top_prim_idx, i])
            dune_top_sec_x.append(cross_shore[dune_top_sec_idx])
            dune_top_sec_y.append(y_all[dune_top_sec_idx, i])

Dimensions['DT_prim_x'] = dune_top_prim_x
Dimensions['DT_prim_y'] = dune_top_prim_y
Dimensions['DT_sec_x'] = dune_top_sec_x
Dimensions['DT_sec_y'] = dune_top_sec_y

###################################
####  DEFINE LANDWARD BOUNDARY ####
###################################

####  Variance method - Sierd  ####
###################################
var_threshold = 0.1 # very dependent on area and range of years!

var_y = np.nanvar(y_all, axis=1)
mean_y = np.nanmean(y_all, axis=1)

# find the first (seaward) location where elevation y_all > 5 and var_y < 0.2
stable_points_index = [i for i,var in enumerate(var_y) if var < var_threshold]
stable_point_idx = [idx for i,idx in enumerate(stable_points_index) if cross_shore[idx] < np.nanmax(Dimensions['DT_prim_x'])][-1]
stable_x = cross_shore[stable_point_idx]

# add info on landward boundary to dataframe
Dimensions['landward_stable_x'] = stable_x

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
landward_x_all = []

for i, yr in enumerate(years_included):
    intersect = find_intersections(cross_shore, y_all[:,i], years_included, landward_y)    
    if len(intersect[0]) == 0:
        landward_x_all.append(np.nan)    
    else:
        idx = intersect[0][-1]
        landward_x = cross_shore[idx]    
        landward_x_all.append(landward_x)    
    
Dimensions['landward_6m_x'] = landward_x_all

###################################  
#### DEFINE DUNEFOOT LOCATION  ####
###################################

#### Fixed dunefoot definition ####
###################################
DF_fixed_y = 3 # in m above reference datum
DF_fix_x = []
DF_fix_y = []

for i, yr in enumerate(years_included):
    intersect = find_intersections(cross_shore, y_all[:,i], years_included, DF_fixed_y)
    if len(intersect[0]) == 0:
       DF_fix_x.append(np.nan)
       DF_fix_y.append(np.nan)    
    else:
        idx = intersect[0][-1]
        DF_fix_x.append(cross_shore[idx])  
        DF_fix_y.append(DF_fixed_y)

Dimensions['DF_fix_y'] = DF_fix_y
Dimensions['DF_fix_x'] = DF_fix_x

####  Derivative method - E.D. ####
###################################
## Variable dunefoot definition based on first and second derivative of profile
dunefoot_x = []
dunefoot_y = []
for i, yr in enumerate(years):
    # Get seaward boundary
    seaward_x = Dimensions['MHW_x_fix'][i] # USED NOW!!! -> REVIEW
#    seaward_x = Dimensions['MHW_x_var'][i] 
#    seaward_x = -100
    
    # Get landward boundary 
    landward_x = landward_x_all[i]    

    # Give nan values to everything outside of boundaries
    y_all_domain = []
    for xc in range(len(cross_shore)): 
        if cross_shore[xc] > seaward_x or cross_shore[xc] < landward_x:
            y_all_domain.append(np.nan)
        else:
            y_all_domain.append(y_all[xc,i])
    
    # Give nan to years where not enough measurements are available within domain or where years not 
    if np.count_nonzero(~np.isnan(y_all_domain)) < 30 or math.isnan(float(yr)):
        dunefoot_x.append(np.nan)
        dunefoot_y.append(np.nan)
    
    else: 
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
        if sum(dunefoot) == 0:
            dunefoot_x.append(np.nan)
            dunefoot_y.append(np.nan)
        else:
            dunefoot_idx = np.where(dunefoot == True)[0][0]
            dunefoot_x.append(cross_shore[dunefoot_idx])
            dunefoot_y.append(y_all[dunefoot_idx, i])
        
Dimensions['DF_der_y'] = dunefoot_y
Dimensions['DF_der_x'] = dunefoot_x

###################################  
###      CALC BEACH WIDTH       ###
###################################

# Get seaward boundary
B_landward_fix = Dimensions['DF_fix_x']
B_landward_var = Dimensions['DF_fix_x']
B_landward_der = Dimensions['DF_der_x']

# Get landward boundary 
B_seaward_fix = Dimensions['MSL_x']
B_seaward_var = (Dimensions['MLW_x_var']+Dimensions['MHW_x_var'])/2
B_seaward_der = Dimensions['MSL_x']

Dimensions['BW_fix'] = B_seaward_fix - B_landward_fix
Dimensions['BW_var'] = B_seaward_var - B_landward_var
Dimensions['BW_der'] = B_seaward_der - B_landward_der

###################################  
###     CALC BEACH GRADIENT     ###
###################################

Dimensions['B_grad_fix'] = get_gradient(cross_shore, y_all, years_included, B_seaward_fix, B_landward_fix)
Dimensions['B_grad_var'] = get_gradient(cross_shore, y_all, years_included, B_seaward_var, B_landward_var)
Dimensions['B_grad_der'] = get_gradient(cross_shore, y_all, years_included, B_seaward_der, B_landward_der)

###################################  
###       CALC DUNE WIDTH       ###
###################################
# Calcualte the width of the dune front that corresponds to the dune front gradient
Dimensions['DFront_fix_prim_W'] = Dimensions['DF_fix_x'] - Dimensions['DT_prim_x']
Dimensions['DFront_der_prim_W'] = Dimensions['DF_der_x'] - Dimensions['DT_prim_x']
Dimensions['DFront_fix_sec_W'] = Dimensions['DF_fix_x'] - Dimensions['DT_sec_x']
Dimensions['DFront_der_sec_W'] = Dimensions['DF_der_x'] - Dimensions['DT_sec_x']

###################################  
###      CALC DUNE GRADIENT     ###
###################################

Dimensions['DFront_fix_prim_grad'] = get_gradient(cross_shore, y_all, years_included, Dimensions['DF_fix_x'], Dimensions['DT_prim_x'])
Dimensions['DFront_der_prim_grad'] = get_gradient(cross_shore, y_all, years_included, Dimensions['DF_der_x'], Dimensions['DT_prim_x'])
Dimensions['DFront_fix_sec_grad'] = get_gradient(cross_shore, y_all, years_included, Dimensions['DF_fix_x'], Dimensions['DT_sec_x'])
Dimensions['DFront_der_sec_grad'] = get_gradient(cross_shore, y_all, years_included, Dimensions['DF_der_x'], Dimensions['DT_sec_x'])

###################################
###    CALC DUNE VOLUME    ### 
###################################

Dimensions['DVol_fix'] = get_volume(cross_shore, y_all, years_included, Dimensions['DF_fix_x'], Dimensions['landward_stable_x'])
Dimensions['DVol_der'] = get_volume(cross_shore, y_all, years_included, Dimensions['DF_der_x'], Dimensions['landward_stable_x'])

###################################
###        SAVE DATAFRAME       ### 
###################################

# Save dataframe for each transect.
# Later these can all be loaded to calculate averages for specific sites/sections along the coast

###################################
####         VISUALIZE         ####
###################################
Dimensions_plot = Dimensions.astype(float) # To make sure all values are converted to float, otherwise error during plotting


fig, axs = plt.subplots(nrows=2, ncols=3) #sharex = True
fig.suptitle('Dimensions of Jarkus transect ' + str(transect[0]))

# Plotting the seaward boundary
keys_plot1 = ['MLW_x_fix', 'MLW_x_var', 'MSL_x', 'MHW_x_fix', 'MHW_x_var', 'DF_fix_x', 'DF_der_x']
colors_plot1 = ['black', 'black', 'blue', 'red', 'red', 'orange', 'orange']
linest_plot1 = ['-','--','-','-','--', '-', '--']

for i in range(len(keys_plot1)):
    Dimensions_plot.plot(kind='line', x = 'years', y = keys_plot1[i], color = colors_plot1[i], linestyle = linest_plot1[i], marker = 'o', ax=axs[0,0])

axs[0,0].set(xlabel='time (yr)', ylabel='cross shore distance (m)',
   title='Cross shore movement of MLW, MSL, MHW and DF')
axs[0,0].set_xlim([1965, 2020])
#axs[0,0].set_ylim([0, 300])
axs[0,0].grid()

# Plotting beach width and intertidal width
keys_plot2 = ['W_intertidal_fix', 'W_intertidal_var', 'BW_fix', 'BW_var', 'BW_der']
# Variation due to definition of the MHW and MSL (fixed vs jarkus data)
# Variation due to definition of dune foot (fixed vs derivative) and the water line (MSL vs MLW-MHW-combo based on the jarkus data)
colors_plot2 = ['green', 'green', 'cyan', 'cyan', 'cyan']
linest_plot2 = ['-', '--', '-', '--', 'dotted']
    
for i in range(len(keys_plot2)):
    Dimensions_plot.plot(kind='line', x = 'years', y = keys_plot2[i], color = colors_plot2[i], linestyle = linest_plot2[i], marker = 'o', ax=axs[1,0])

axs[1,0].set(xlabel='time (yr)', ylabel='width (m)',
   title='Width of the intertidal area and beach')
axs[1,0].set_xlim([1965, 2020])
#axs[1,0].set_ylim([0, 200])
axs[1,0].grid()

# Plotting beach gradient
keys_plot3 = ['B_grad_fix', 'B_grad_var', 'B_grad_der']
# Variation due to definition of the dune foot (fixed vs derivative) and the water line (MSL vs MLW-MHW-combo)
colors_plot3 = ['darkblue', 'darkblue', 'darkblue']
linest_plot3 = ['-', '--', 'dotted']

for i in range(len(keys_plot3)):
    Dimensions_plot.plot(kind='line', x = 'years', y = keys_plot3[i], color = colors_plot3[i], linestyle = linest_plot3[i], marker = 'o', ax=axs[1,2])

axs[1,2].set(xlabel='time (yr)', ylabel='slope (m/m)',
   title='Beach gradient')
axs[1,2].set_xlim([1965, 2020])
#axs[1,2].set_ylim([-0.05, -0.01])
axs[1,2].grid()

# Plotting dune gradient (both primary and secondary)
keys_plot4 = ['DFront_fix_prim_grad', 'DFront_der_prim_grad', 'DFront_fix_sec_grad', 'DFront_der_sec_grad']
# Variation due to definition of the dune foot (fixed vs derivative)
colors_plot4 = ['purple', 'purple', 'pink', 'pink']
linest_plot4 = ['-', 'dotted', '-', 'dotted']

for i in range(len(keys_plot4)):
    Dimensions_plot.plot(kind='line', x = 'years', y = keys_plot4[i], color = colors_plot4[i], linestyle = linest_plot4[i], marker = 'o', ax=axs[1,1])

axs[1,1].set(xlabel='time (yr)', ylabel='slope (m/m)',
   title='Dune front gradient')
axs[1,1].set_xlim([1965, 2020])
#axs[1,1].set_ylim([-0.6, -0.1])
axs[1,1].grid()

# Plotting dune height (both primary and secondary)
keys_plot5 = ['DT_prim_y', 'DT_sec_y']
colors_plot5 = ['grey', 'red']
linest_plot5 = ['-', '-']

for i in range(len(keys_plot5)):
    Dimensions_plot.plot(kind='line', x = 'years', y = keys_plot5[i], color = colors_plot5[i], linestyle = linest_plot5[i], marker = 'o', ax=axs[0,2])

axs[0,2].set(xlabel='time (yr)', ylabel='elevation (m)',
   title='Dune height')
axs[0,2].set_xlim([1965, 2020])
#axs[0,2].set_ylim([5, 25])
axs[0,2].grid()

# Plotting dune volume
keys_plot6 = ['DVol_fix', 'DVol_der']
# Variation due to definition of the dune foot (fixed vs derivative)
colors_plot6 = ['brown', 'brown']
linest_plot6 = ['-', 'dotted']

for i in range(len(keys_plot6)):
    Dimensions_plot.plot(kind='line', x = 'years', y = keys_plot6[i], color = colors_plot6[i], linestyle = linest_plot6[i], marker = 'o', ax=axs[0,1])

axs[0,1].set(xlabel='time (yr)', ylabel='volume (m^3/m)',
   title='Dune volume')
axs[0,1].set_xlim([1965, 2020])
#axs[0,1].set_ylim([0, 1000])
axs[0,1].grid()


#fig.savefig("test.png")
plt.show()



fig2, axs2 = plt.subplots(nrows=1, ncols=2) #sharex = True
fig.suptitle('Dune dimensions of Jarkus transect '+ str(transect[0]))

# Plotting dune gradient (both primary and secondary)
keys_plot4 = ['DFront_der_prim_grad', 'DFront_der_sec_grad']
# Variation due to definition of the dune foot (fixed vs derivative)
colors_plot4 = ['black', 'red']
linest_plot4 = ['-','-']
labels4 = ['Primary','Secondary']

for i in range(len(keys_plot4)):
    Dimensions_plot.plot(kind='line', x = 'years', y = keys_plot4[i], color = colors_plot4[i], linestyle = linest_plot4[i], marker = 'o', ax=axs2[1], label = labels4[i])

axs2[1].set(xlabel='time (yr)', ylabel='slope (m/m)')
axs2[1].set_title('Dune front gradient', fontsize=18)
axs2[1].set_xlim([1965, 2020])
#axs2[1].set_ylim([-0.6, -0.1])
axs2[1].grid()

# Plotting dune height (both primary and secondary)
keys_plot5 = ['DT_prim_y', 'DT_sec_y']
colors_plot5 = ['black', 'red']
linest_plot5 = ['-', '-']
labels5 = ['Primary','Secondary']

for i in range(len(keys_plot5)):
    Dimensions_plot.plot(kind='line', x = 'years', y = keys_plot5[i], color = colors_plot5[i], linestyle = linest_plot5[i], marker = 'o', ax=axs2[0], label = labels5[i])

axs2[0].set(xlabel='time (yr)', ylabel='elevation (m)')
axs2[0].set_title('Dune height', fontsize=18)
axs2[0].set_xlim([1965, 2020])
#axs2[0].set_ylim([8, 22])
axs2[0].grid()

plt.show()
