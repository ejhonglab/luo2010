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

# TODO exclude pheromone (and other selective) receptors, to compare to what Luo et al
# actually did

# TODO using receptor data now. mostly the same as using glomeruli data, but should perhaps
# be using glomeruli data instead if there is overlapping expression of any of the receptors
# in their dataset, or if any two provide input to a common set of PNs

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

cax = ax.matshow(orn, cmap=plt.cm.viridis, aspect=0.3) #aspect='auto')

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
cbar.set_label('Spike count change in 500ms(?) presentation', **cbar_font)

# keep all columns
# but exclude last odor label (because it is the spontaneous firing rate)
plt.xticks(np.arange(len(hc06.columns)))
plt.yticks(np.arange(len(hc06.index) - 1))
ax.set_xticklabels(hc06.columns.values, rotation='horizontal')
ax.xaxis.set_ticks_position('bottom')
ax.set_yticklabels(hc06.index.values[:-1], fontsize=6)

"""
Fig 1b: simple model PN responses (assuming now 1 receptor -> 1 PN (class). see note above)
"""

# units of Hz in the paper
# TODO is that correct here? (seems so)
rmax = 165
sigma = 12

# model PN responses with no inhibition
pn_no_inh = rmax * orn**1.5 / (sigma**1.5 + orn**1.5)

# TODO add noise a la methods

# TODO make a function out of me. maybe subplot?
fig2 = plt.figure()
ax2 = fig2.add_subplot(111)

cax2 = ax2.matshow(pn_no_inh, cmap=plt.cm.viridis, aspect=0.3, vmin=cbar.vmin, vmax=cbar.vmax)

plt.title('Binned model PN responses (no global inhibition)', fontweight='bold', y=1.01)

plt.xlabel('Receptor in recorded cell', x_axes_font)
plt.ylabel('Odorant', y_axes_font)

cbar2 = fig2.colorbar(cax2, shrink=0.6, aspect=30, pad=0.02)
cbar2.set_label('Spike count change in 500ms(?) presentation', **cbar_font)

# keep all columns
# but exclude last odor label (because it is the spontaneous firing rate)
plt.xticks(np.arange(len(hc06.columns)))
plt.yticks(np.arange(len(hc06.index) - 1))
ax2.set_xticklabels(hc06.columns.values, rotation='horizontal')
ax2.xaxis.set_ticks_position('bottom')
ax2.set_yticklabels(hc06.index.values[:-1], fontsize=6)

"""
Fig 1C - E
"""

m = 0.05
# model PN responses WITH global inhibition (dependent on sum of ORN activity)
# TODO make sure this broadcasting with newaxis is working correctly
pn = rmax * orn**1.5 / (sigma**1.5 + orn**1.5 + (m * np.sum(orn, axis=1)[:, np.newaxis])**1.5)

fig3 = plt.figure()
#fig3.title('Mean firing rates across odors')

a1 = plt.subplot(131)
plt.plot(np.arange(orn.shape[0]), np.mean(orn, axis=1), '.')
plt.title('ORN')
plt.ylabel('Firing rate (spikes/s)')

# TODO this overall activity seems too high maybe?
a2 = plt.subplot(132)
plt.plot(np.arange(pn_no_inh.shape[0]), np.mean(pn_no_inh, axis=1), '.')
plt.title('PN (no inhibition)')
plt.xlabel('Odor')

a3 = plt.subplot(133)
plt.plot(np.arange(pn.shape[0]), np.mean(pn, axis=1), '.')
plt.title('PN')

axs = [a1, a2, a3]
ymax = max([max(a.get_ylim()) for a in axs])

for a in axs:
    a.set_ylim(0, ymax)

plt.show()
