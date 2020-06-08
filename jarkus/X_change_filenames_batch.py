# -*- coding: utf-8 -*-
"""
Created on Mon Oct  7 15:47:25 2019

@author: cijzendoornvan
"""

import os
     
path = "C:/Users/cijzendoornvan/Documents/GitHub/jarkus/jarkus/Figures/17_Zeeuws-Vlaanderen/"
    #input("Enter the directory path where you need to  rename: ")
for filename in os.listdir(path):
    print(filename)
    filename_changed = filename.replace(" ", "")
    print(filename_changed)
    os.rename(os.path.join(path,filename),os.path.join(path,filename_changed))
    