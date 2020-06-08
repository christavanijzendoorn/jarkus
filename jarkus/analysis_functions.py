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
Created on Fri Oct 11 17:21:09 2019

@author: cijzendoornvan
"""

# This script was developed to extract volumes, gradient and intersections from the Jarkus dataset

##################################
####          PACKAGES        ####
##################################
import numpy as np

#################################
####        FUNCTIONS        ####
#################################

def get_volume(x, y, years, lower_boundary_x, upper_boundary_x):
    volume = []
    volume_idx = []
    for i, yr in enumerate(years):
        LB_x = np.ceil(lower_boundary_x[i])
        UB_x = np.floor(upper_boundary_x[i])
        if np.isnan(upper_boundary_x[i]) or np.isnan(lower_boundary_x[i]):
            volume.append(np.nan)
        else:
            volume_idx_LB = np.where(x <= LB_x)
            volume_idx_UB = np.where(x >= UB_x)
            volume_idx = [value for value in volume_idx_LB[0] if value in volume_idx_UB[0]]
            volume_x = [x[value] for value in volume_idx_LB[0] if value in volume_idx_UB[0]]
            #print(volume_x)
            if UB_x in volume_x == False or LB_x in volume_x == False:
                volume.append(np.nan)
            else:
                y_values = [y[k,i] for k in volume_idx]
                volume_y = [value - min(y_values) for value in y_values]
                
                #print(y_dune)
                volume_trapz = np.trapz(volume_y, x = volume_x)
                volume.append(volume_trapz)
    
    return volume

def get_gradient(x, y, years, seaward_bound, landward_bound):
    gradient = []
    for i, yr in enumerate(years):
        # Extract elevation profile with seaward and landward boundaries
        elevation_y = []
        cross_shore_x = []
        
        for xc in range(len(x)): 
            if x[xc] < seaward_bound[i] and x[xc] > landward_bound[i] and np.isnan(y[xc,i]) == False:
                elevation_y.append(y[xc,i])
                cross_shore_x.append(x[xc])
        # Calculate gradient for domain
        if cross_shore_x == []:
            gradient.append(np.nan)
        else:
            gradient.append(np.polyfit(cross_shore_x, elevation_y, 1)[0])
        
    return gradient

def find_intersections(x, y, y_value):
    value_vec = []
    for x_val in range(len(x)):
        value_vec.append(y_value)
    
    diff = np.nan_to_num(np.diff(np.sign(y - value_vec)))
    intersections = np.nonzero(diff)
    
    return intersections
