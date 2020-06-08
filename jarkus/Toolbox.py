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
