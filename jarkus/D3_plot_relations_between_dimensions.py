# -*- coding: utf-8 -*-
"""
Created on Tue Nov 19 14:10:50 2019

@author: cijzendoornvan
"""
##################################
####          PACKAGES        ####
##################################
import json
import numpy as np
import pandas as pd
import os.path
import pickle
import matplotlib.pyplot as plt
from IPython import get_ipython
get_ipython().run_line_magic('matplotlib', 'auto') ## %matplotlib auto TO GET WINDOW FIGURE
#################################
####        FUNCTIONS        ####
#################################
def plot_relation(Dir_variables, variables, title, x_name, y_name, Dir, file_name, area, xlimits, ylimits):
    pickle_file1 = Dir_variables + variables[0] + '_dataframe.pickle'    
    pickle_file2 = Dir_variables + variables[1] + '_dataframe.pickle'    
    
    if os.path.exists(pickle_file1) and os.path.exists(pickle_file2):
        Variable_values1 = pickle.load(open(pickle_file1, 'rb')) #load pickle of dimension    
        Variable_values2 = pickle.load(open(pickle_file2, 'rb')) #load pickle of dimension    
        
        plt.figure(figsize=(30,15))  
        for col in Variable_values1.columns:
            if area == 'DC' and int(col) > 10000000: 
                plt.scatter(Variable_values1[col], Variable_values2[col], label=col)
            elif area == 'HC' and int(col) < 10000000 and int(col) >= 7000000:
                plt.scatter(Variable_values1[col], Variable_values2[col], label=col)
            elif area == 'WC' and int(col) >= 2000101 and int(col) < 7000000:
                plt.scatter(Variable_values1[col], Variable_values2[col], label=col)
        plt.legend(loc='best', fontsize=16)
        plt.xlim(xlimits[0], xlimits[1])
        plt.ylim(ylimits[0], ylimits[1])
        plt.xlabel(x_name)
        plt.ylabel(y_name)
        plt.title(title + ' ' + area)
        plt.savefig(Dir + file_name + '_plot.png')
        #pickle.dump(fig, open(Dir + file_name + '_plot.fig.pickle', 'wb'))
        #plt.close()
    else:
        print('One of the dataframes was not available.')

##################################
####    USER-DEFINED REQUEST  ####
##################################
# load basic settings from .txt file
with open("C:/Users/cijzendoornvan/Documents/GitHub/jarkus/jarkus/Settings.txt") as file:
    settings = json.load(file)

Dir_per_variable = settings['Dir_D1']
Dir_relation_plots = settings['Dir_D3']

years_requested = list(range(1965, 2020))
labels_y = [str(yr) for yr in years_requested][0::5]

##################################
####  CREATE RELATION PLOTS   ####
##################################


variables = ['max_TWL_gen', 'DVol_fix_rate']
x_name = 'Total water level (based on general equation)'
y_name = 'Dune volume change rate'
title = 'Dune volume change rate as a function of Total Water Level (based on general equation)'
file_name = 'DVol_vs_TWL'
xlimits = [0, 12]
ylimits = [-30, 30]
area = 'HC'

plot_relation(Dir_per_variable, variables, title, x_name, y_name, Dir_relation_plots, file_name, area, xlimits, ylimits) 

variables = ['max_TWL_gen', 'DF_der_y']
x_name = 'Total water level'
y_name = 'Dune foot elevation'
title = 'Dune elevation as a function of Total Water Level'
file_name = 'DFy_vs_TWL'
xlimits = [0, 12]
ylimits = [-10, 6]
area = 'HC'

plot_relation(Dir_per_variable, variables, title, x_name, y_name, Dir_relation_plots, file_name, area, xlimits, ylimits) 

variables = ['max_TWL_gen', 'B_grad_fix']
x_name = 'Total water level'
y_name = 'Beach slope'
title = 'Beach slope as a function of Total Water Level'
file_name = 'Bgrad_vs_TWL'
xlimits = [0, 12]
ylimits = [-0.1, 0]
area = 'HC'

plot_relation(Dir_per_variable, variables, title, x_name, y_name, Dir_relation_plots, file_name, area, xlimits, ylimits) 


variables = ['B_grad_fix', 'DVol_fix_rate']
x_name = 'Beach slope'
y_name = 'Dune volume change'
title = 'Dune volume change as a function of Beach slope'
file_name = 'DVol_vs_Bgrad'
xlimits = [-0.1, 0]
ylimits = [-75, 75]
area = 'HC'

plot_relation(Dir_per_variable, variables, title, x_name, y_name, Dir_relation_plots, file_name, area, xlimits, ylimits) 

variables = ['DF_der_x', 'DF_fix_x']
x_name = 'DF derivative'
y_name = 'DF fixed'
title = 'DF vs DF'
file_name = 'DF_fix_vs_DF_der'
xlimits = [-300, 300]
ylimits = [-300, 300]
area = 'HC'

plot_relation(Dir_per_variable, variables, title, x_name, y_name, Dir_relation_plots, file_name, area, xlimits, ylimits) 



"""
variables = ['DF_fix_x_normalized', 'DVol_fix_rate']
x_name = 'Dune foot location'
y_name = 'Dune volume change'
title = 'Dune volume change as a function of Beach slope'
file_name = 'DVol_vs_Bgrad'
xlimits = [-0.1, 0]
ylimits = [-75, 75]
area = 'HC'

plot_relation(Dir_per_variable, variables, title, x_name, y_name, Dir_relation_plots, file_name, area, xlimits, ylimits) 
"""