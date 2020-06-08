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

import netCDF4

##################################

# load basic settings from .txt file
with open("C:/Users/cijzendoornvan/Documents/GitHub/jarkus/jarkus/Settings.txt") as file:
    settings = json.load(file)

# specify an url, the JARKUS dataset in this case
url = settings['url']
# for local windows files, note that '\t' defaults to the tab character in python, so use prefix r to indicate that it is a raw string.
#url = r'f:\opendap\rijkswaterstaat\jarkus\profiles\transect.nc'
# create a dataset object
dataset = netCDF4.Dataset(url)
 
# lookup all variables available
variables = dataset.variables
# print the structure of the ncd file
print(variables)

# lookup all variables available
variable = dataset.variables['time_topo']

variable = dataset.variables['alongshore']
print(variable[175])


# print the structure of the ncd file
time_1965 = variable[0]
print(np.nanmin(time_1965), np.nanmax(time_1965))
time_1970 = variable[5]
print(np.nanmin(time_1970), np.nanmax(time_1970))
print()
