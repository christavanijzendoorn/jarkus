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

import pandas as pd
Suppleren = pd.read_excel("C:/Users/cijzendoornvan/Documents/DuneForce/JARKUS/Suppletiedatabase.xlsx")

Suppleren_sort = Suppleren.sort_values(by='JaarBeginUitvoering')

Suppleren_sum = Suppleren.groupby('JaarBeginUitvoering')['Volume (situ)'].sum()

years = list(range(1950, 2020, 5))

supll_vols = []
for yr in years:
    vols = Suppleren_sum[Suppleren_sum.index >= yr]
    sup_vol = vols[vols.index < yr+5].sum()
    supll_vols.append(sup_vol/1000000)
    
data_tuples = list(zip(years,supll_vols))
result = pd.DataFrame(data_tuples, columns = ['Years','Nourishment volume (Mm\N{SUPERSCRIPT THREE})'])
results_yrs = result.set_index('Years')
ax = results_yrs.plot(kind='bar', fontsize=30)  
ax.tick_params(axis='x', which='both',length=0, labelsize = 30)  
ax.set_xlabel(' ', fontsize=30)
ax.legend(fontsize=30)