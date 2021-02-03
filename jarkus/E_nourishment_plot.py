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
Created on Tue Feb 11 21:16:01 2020

@author: cijzendoornvan
"""

# This script loads noursihment data and converts it into a plot showing the nourishment volume per time period.

get_ipython().run_line_magic('matplotlib', 'auto') ## %matplotlib auto TO GET WINDOW FIGURE

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
Suppleren = pd.read_excel("C:/Users/cijzendoornvan/Documents/DuneForce/JARKUS/Suppletiedatabase.xlsx")

for i, row in Suppleren.iterrows():
    # Remove Hondsbossche Duinen (35 miljoen)
    if row['Locatie'] == 'HBPZ':
        Suppleren = Suppleren.drop(i)
    elif row['JaarEindUitvoering'] < 1965:
        Suppleren = Suppleren.drop(i)
    elif row['JaarEindUitvoering'] == 2020:
        Suppleren = Suppleren.drop(i)
    # elif row['Locatie'] == 'Hondsbossche- en Pettemer Zeewering':
    #     Suppleren = Suppleren.drop(i)
    # Remove Zandmotor (17 miljoen)
    elif pd.notnull(row['Opmerkingen']):
        if row['Opmerkingen'] == 'zandmotor':
            Suppleren = Suppleren.drop(i)

# Remove Maasvlakte (kustvak 10)
Idx_Maasvlakte = Suppleren[Suppleren.KustVakNummer == 10].index
Suppleren = Suppleren.drop(Idx_Maasvlakte)
# Remove Waddeneilanden
Idx_Wadden = Suppleren[Suppleren.KustVakNummer <= 6].index
Suppleren = Suppleren.drop(Idx_Wadden)

Suppleren['Type_combi'] = np.nan

for i, row in Suppleren.iterrows():
    if row['Type'] == 'vooroeversuppletie':
        Suppleren.loc[i, 'Type_combi'] = 'Subaqeous nourishment'
    elif row['Type'] == 'strandsuppletie':
        Suppleren.loc[i, 'Type_combi'] = 'Beach nourishment'
    elif row['Type'] == 'strand-duinsuppletie':
        Suppleren.loc[i, 'Type_combi'] = 'Beach-dune nourishment'
    elif row['Type'] == 'geulwandsuppletie':
        Suppleren.loc[i, 'Type_combi'] = 'Subaqeous nourishment'
    elif row['Type'] == 'duinverzwaring':
        Suppleren.loc[i, 'Type_combi'] = 'Dune nourishment'
    elif row['Type'] == 'duinsuppletie':
        Suppleren.loc[i, 'Type_combi'] = 'Dune nourishment'
    elif row['Type'] == 'duin':
        Suppleren.loc[i, 'Type_combi'] = 'Dune nourishment'
    elif row['Type'] == 'dijkverzwaring':
        Suppleren.loc[i, 'Type_combi'] = 'Other'
    elif row['Type'] == 'diepe vooroever':
        Suppleren.loc[i, 'Type_combi'] = 'Subaqeous nourishment'
    elif row['Type'] == 'buitendelta':
        Suppleren.loc[i, 'Type_combi'] = 'Other'
    elif row['Type'] == 'anders':
        Suppleren.loc[i, 'Type_combi'] = 'Other'

for i, row in Suppleren.iterrows():
    lastdigit = int(repr(row['JaarEindUitvoering'])[-1])
    if lastdigit == 1 or lastdigit == 6:
        Suppleren.loc[i, 'Jaar_interval'] = row['JaarEindUitvoering'] - 1
    elif lastdigit == 2 or lastdigit == 7:
        Suppleren.loc[i, 'Jaar_interval'] = row['JaarEindUitvoering'] - 2
    elif lastdigit == 3 or lastdigit == 8:
        Suppleren.loc[i, 'Jaar_interval'] = row['JaarEindUitvoering'] - 3
    elif lastdigit == 4 or lastdigit == 9:
        Suppleren.loc[i, 'Jaar_interval'] = row['JaarEindUitvoering'] - 4
    else:
        Suppleren.loc[i, 'Jaar_interval'] = row['JaarEindUitvoering']

Suppleren['Jaar_interval'] = pd.to_datetime(Suppleren['Jaar_interval'], format='%Y')
Suppleren_sum = Suppleren.groupby(['Jaar_interval','Type_combi']).agg(sum_volume = ('Volume (situ)','sum'))
Suppleren_sum['sum_volume'] = Suppleren_sum['sum_volume']/1000000

Suppleren_yearly = Suppleren.groupby(['JaarEindUitvoering','Type_combi']).agg(sum_volume = ('Volume (situ)','sum'), total_length = ('Lengte','sum'))
Suppleren_yearly['sum_volume'] = Suppleren_yearly['sum_volume']/1000000

total_length_coast = 180000
Suppleren_mean = Suppleren.groupby(['JaarEindUitvoering']).agg(sum_volume = ('Volume (situ)','sum'), total_length = ('Lengte','sum'))
Suppleren_mean['mean_volume_m'] = Suppleren_mean['sum_volume']/Suppleren_mean['total_length']
Suppleren_mean['mean_vol_m_overall'] = Suppleren_mean['sum_volume']/total_length_coast

#%%

SLR = pd.read_excel("C:/Users/cijzendoornvan/Documents/DuneForce/JARKUS/zeespiegelstijging_clo.xlsx")


fig, axs = plt.subplots(1, 2, figsize=(25,12))

Suppleren_sum.unstack().plot(kind = 'bar', stacked=True, color = ['#fac203', '#d89002', '#136207', 'grey', '#4169e1'], fontsize=26, ax=axs[1])
years = list(range(1965, 2020, 5))
xtick_labels = [str(yr) for yr in years]
axs[1].set_xticklabels(xtick_labels, rotation=30, horizontalalignment="center", fontsize=22)
axs[1].set_ylabel("Volume (million m$^{3}$)", fontsize=26)
axs[1].set_xlabel("Year of completion", fontsize=26)
legend = axs[1].legend(loc='upper left', title="Nourishment type", labels = ['Beach', 'Beach-dune', 'Dune', 'Other', 'Subaqeous'], fontsize=24)
legend._legend_box.align = "left"
plt.setp(legend.get_title(),fontsize=26)
axs[1].tick_params(axis='both', which='major', labelsize=22)
axs[1].tick_params(axis='x', which='major', rotation=30)

axs[0].plot(SLR['Jaren'], SLR['Trend'], color = '#4169E1')#, marker = 'o', markersize = 8)
axs[0].fill_between(SLR['Jaren'], SLR['Onzekerheid trend Bandbreedte min'], SLR['Onzekerheid trend Bandbreedte max'], color = '#4169E1', alpha = 0.15)
axs[0].scatter(SLR['Jaren'], SLR['Jaargemiddelde 6 kuststations'], color = 'grey', alpha = 0.6)#, marker = 'o', markersize = 8)
axs[0].set_ylabel("Sea level (cm above NAP)", fontsize=26)
axs[0].set_xlabel("Year", fontsize=26)
legend = axs[0].legend(loc='upper left', title = 'Sea level rise', labels = ['Trend of 1.9 mm/yr', 'Trend uncertainty', 'Yearly average of \n6 coastal stations'], fontsize=24)
plt.setp(legend.get_title(),fontsize=26)
legend._legend_box.align = "left"
axs[0].tick_params(axis='both', which='major', labelsize=22)
axs[0].tick_params(axis='x', which='major', rotation=30)

