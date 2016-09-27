#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

# prevents white lines from being overlayed over data
sns.set_style('dark')

"""
A little bit about the Hallem and Carlson 2006 dataset:
    -each odor pulse is 500ms long (exceptions?)
    -responses (other than spontaneous) are # of spikes in those 500ms - spontaneous
    -where spontaneous is just count in 500ms without stimulation
    -flies are <4 weeks old (but how young on average?)
    -24 ml/s carrier with 5.9 ml/s odor stream
    -female flies (??)

    TODO: my interpretation of why spontaneous - diff is sometimes negative is that
    the diff is calculated using (in at least the 2004 paper, was unclear in 2006)
    the second (maybe 0.5 sec here) before and the second after (with 500 ms pulse
    in their 2004 paper)
    -if this is the case, there could just be an unusually busy period preceding the
     stimulus, exagerating the inhibition
"""

"""
Fig 1a: raw Hallem and Carlson 2006
"""

# skip glomerulus labels, which are not assigned to each response
# keep the receptor labels, which are assigned to each response
hc06 = pd.read_csv('~/Dropbox/Hallem_Carlson_2006.csv', skiprows=1)

# first column of first row manually set to this in the CSV data file
# might want to use more pandonic way of doing this
hc06 = hc06.set_index(['odor'])
hc06.columns.name = 'receptor'

# way to add the labels with less handles?
fig = plt.figure()
# the 111 means "1x1 grid, first subplot"
ax = fig.add_subplot(111)

# exclude the last row, because those are spontaneous firing rates
delta_orn = hc06.as_matrix()[:-1,:]

# get the spontaneous rates so we can add them back
spont_orn = hc06.loc[hc06.index == 'spontaneous firing rate'].as_matrix().flatten()

# recover actual firing rates (in the 500ms binning windows)
orn = np.empty_like(delta_orn) * np.nan

# TODO numpy way to do this broadcasting?

# for each odor add the spontaneous rates to the difference observed for that odor
# adds the vector of spontaneous rates across all receptors to the odor specific
# vector of reponses across receptors
for i in range(delta_orn.shape[0]):
    orn[i,:] = delta_orn[i,:] + spont_orn

"""
It seems that what Luo et al do is set anything here that would go below zero to zero,
but that might be wrong. Read more carefully, but they might not say.
"""
orn[orn < 0] = 0

cax = ax.matshow(hc06.as_matrix(), cmap=plt.cm.viridis, aspect=0.3) #aspect='auto')

plt.title('Binned ORN responses', fontweight='bold', y=1.01)

axes_font_size = 10
axes_font_weight = 'demi'
x_axes_font = {'fontsize': axes_font_size ,
               'fontweight': axes_font_weight,
               'verticalalignment': 'top',
               'horizontalalignment': 'center'}

y_axes_font = {'fontsize': axes_font_size ,
               'fontweight': axes_font_weight,
               'verticalalignment': 'bottom',
               'horizontalalignment': 'center'}

plt.xlabel('Receptor in recorded cell', x_axes_font)
plt.ylabel('Odorant', y_axes_font)

cbar_font = {'fontsize': axes_font_size ,
             'fontweight': axes_font_weight,
             'verticalalignment': 'top',
             'horizontalalignment': 'center'}

cbar = fig.colorbar(cax, shrink=0.6, aspect=30, pad=0.02)
cbar.set_label('Spike count change in 500ms presentation', **cbar_font)

plt.xticks(np.arange(len(hc06.columns)))
plt.yticks(np.arange(len(hc06.index)))
ax.set_xticklabels(hc06.columns.values, rotation='horizontal')
ax.xaxis.set_ticks_position('bottom')
ax.set_yticklabels(hc06.index.values, fontsize=6)

plt.show()

"""
Fig 1b: 
"""
