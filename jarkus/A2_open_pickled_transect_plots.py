# -*- coding: utf-8 -*-
"""
Created on Thu Oct 17 15:43:48 2019

@author: cijzendoornvan
"""

##################################
####    USER-DEFINED REQUEST  ####
##################################
# Set the transect and years for retrieval request
transect_name = "06_Texel"
transect_req = 6002521
years_requested = range(1970, 2020, 1)

##################################
####  REOPEN PICKLED FIGURE   ####
##################################
# Execute %matplotlib auto first, otherwise you get an error
from visualisation import reopen_pickle
fig_transect = str(transect_req[0])
Dir_fig = "C:\\Users\\cijzendoornvan\\Documents\\GitHub\\jarkus\\jarkus\\Figures\\" + transect_name + "\\"

reopen_pickle(Dir_fig, fig_transect)