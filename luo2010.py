#!/usr/bin/env python3

import numpy as np
from scipy import linalg 
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

import ipdb


sample_pns_with_replacement = True
# if True, does not add the noise they added in the paper
deterministic = False
# number of trials used to generate "response probability" plots
# am i misunderstanding?
simulated_trials = 1000
exclude_pheromone_receptors = False


if (deterministic or not sample_pns_with_replacement or
    exclude_pheromone_receptors):
    raise NotImplementedError

np.random.seed(1118)

# adapted from a StackOverflow answer by 'doug'
def pca(data, components=None):
    """
    returns: data transformed in 2 dims/columns + regenerated original data
    pass in: data as 2D NumPy array
    """

    # mean center the data
    # could have been screwing things up later
    #data -= data.mean(axis=0)

    # calculate the covariance matrix
    # rowvar=False is equivalent to data being transposed
    # (so output will be pxp as below)
    R = np.cov(data, rowvar=False)

    assert np.allclose(np.cov(data, rowvar=False), 
        np.cov(data - data.mean(axis=0), rowvar=False))

    #print(R.diagonal().sum())

    # using variable names in Wikipedia's first algorithm for computing PCA
    # n observations, p variables
    n = data.shape[0]
    p = data.shape[1]

    assert R.shape == (p, p)
    centered = data - data.mean(axis=0)
    assert np.allclose(1 / (n-1) * np.dot(centered.transpose(), centered), R)

    if components == None:
        components = p

    # want (V^-1)CV = D, where V are eigenvectors
    # D is a (pxp) diagonal matrix of eigenvalues
    # matrix V also (pxp) is made of column vectors (the *right* eigenvectors)
    # only under special conditions are the two each other's transpose
    # but since the covariance matrix is symmetric, they are the transpose of
    # each other

    # TODO manually calculate eigenvectors and compare

    # calculate eigenvectors & eigenvalues of the covariance matrix
    # use 'eigh' rather than 'eig' since R is symmetric, 
    # the performance gain is substantial
    evals, evecs = linalg.eigh(R)

    # test the eigenvectors satisfy identities that eigenvectors should satisfy
    assert np.allclose(np.linalg.inv(evecs).dot(R).dot(evecs), np.diag(evals))
    assert np.allclose(R, evecs.dot(np.diag(evals)).dot(np.linalg.inv(evecs)))

    # each column of evecs is an eigenvector, as it should be
    for i in range(evecs.shape[1]):
        assert np.allclose(R.dot(evecs[:,i]), evals[i] * evecs[:,i])

    # sort eigenvalue in decreasing order
    idx = np.argsort(evals)[::-1]
    evecs = evecs[:,idx]

    # sort eigenvectors according to same index
    evals = evals[idx]

    # select the first n eigenvectors (n is desired dimension
    # of rescaled data array, or dims_rescaled_data)
    evecs = evecs[:, :components]

    #assert np.dot(evecs.T, data.T).T.shape == data.shape

    # carry out the transformation on the data using eigenvectors
    # and return the re-scaled data, eigenvalues, and eigenvectors
    # TODO HOW TO TEST WHETHER IT IS SUPPOSED TO BE EVEC OR TRANPOSE
    # columns should be eigenvectors? can test identity, and then explicitly use
    # same
    # columns for reconstruction?
    #return np.dot(evecs.T, data.T).T, evals, evecs
    return np.dot(data, evecs), evals, evecs


def test_pca(data):
    '''
    test by attempting to recover original data array from
    the eigenvectors of its covariance matrix & comparing that
    'recovered' array with the original data
    '''
    projected, _, eigenvectors = pca(data)

    assert np.allclose(np.dot(eigenvectors, data.T).T, 
        np.dot(data, eigenvectors.T))

    data_recovered = np.dot(projected, np.linalg.inv(eigenvectors))

    assert np.allclose(data, data_recovered)

    # TODO so it is invertible, but need to somehow test the components things
    # are projected on to are indeed the best components (roughly same as
    # premade PCA functions use)

    projected, _, eigenvectors = pca(data, components=3)

    sk_pca = PCA(n_components=3)
    sk_data = sk_pca.fit_transform(data)

    # TODO compare to premade PCA
    print(np.sum(np.abs(sk_pca.inverse_transform(sk_data) - data)))
    print(eigenvectors.shape)
    # TODO it might be a matter of how i am inverting things (if it werent for
    # pinv, this would not be invertible), but my error is way larger at 3
    # components...
    print(np.sum(np.abs(np.dot(projected, np.linalg.pinv(eigenvectors)) 
        - data)))
    print(sk_data - projected)
    #assert np.allclose(sk_sorn, projected)


# prevents white lines from being overlayed over data
sns.set_style('dark')

"""
A little bit about the Hallem and Carlson 2006 dataset:
    -each odor pulse is 500ms long (exceptions?)
    -responses (other than spontaneous) are # of spikes in those 500ms -
     spontaneous
    -where spontaneous is just count in 500ms without stimulation
    -flies are <4 weeks old (but how young on average?)
    -24 ml/s carrier with 5.9 ml/s odor stream
    -female flies (??)

    TODO: my interpretation of why spontaneous - diff is sometimes negative is
    that the diff is calculated using (in at least the 2004 paper, was unclear
    in 2006) the second (maybe 0.5 sec here) before and the second after (with
    500 ms pulse in their 2004 paper)
    -if this is the case, there could just be an unusually busy period preceding
     the stimulus, exagerating the inhibition
"""

# TODO exclude pheromone (and other selective) receptors, to compare to what Luo
# et al actually did

# TODO using receptor data now. mostly the same as using glomeruli data, but
# should perhaps be using glomeruli data instead if there is overlapping
# expression of any of the receptors in their dataset, or if any two provide
# input to a common set of PNs

# skip glomerulus labels, which are not assigned to each response
# keep the receptor labels, which are assigned to each response
# TODO just use drosolf?
# TODO possible to make matrix plotting function not depend on this? maybe use
# df? (so this could either be taken from drosolf or otherwise moved below,
# with rest of numerical code, away from mess of plotting code)
hc06 = pd.read_csv('./Hallem_Carlson_2006.csv', skiprows=1)

# first column of first row manually set to this in the CSV data file
# might want to use more pandonic way of doing this
hc06 = hc06.set_index(['odor'])
hc06.columns.name = 'receptor'
# TODO remove receptors Or33b, Or47b, Or65a, and Or88a (pheromone receptors) as
# in paper
n_pns = len(hc06.columns)
n_odors = len(hc06.index) - 1

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

cbar_font = {'fontsize': axes_font_size ,
             'fontweight': axes_font_weight,
             'verticalalignment': 'top',
             'horizontalalignment': 'center'}


def matrix_plot(mat, title='', xlabel='', ylabel='', matrix_aspect='auto',
    cmap=plt.cm.viridis, cbar_label='', xtickstep=None, ytickstep=None):
    """
    Args:
        xtickstep (int):
        ytickstep (int): 
    """

    fig = plt.figure()
    # the 111 means "1x1 grid, first subplot"
    # TODO still necessary?
    ax = fig.add_subplot(111)
    cax = ax.matshow(mat, cmap=cmap, aspect=matrix_aspect) #aspect='auto')
    plt.title(title, fontweight='bold', y=1.01)
    plt.xlabel(xlabel, x_axes_font)
    plt.ylabel(ylabel, y_axes_font)

    # TODO don't hardcode these values if this function is going to have
    # application beyond ORN / PN data
    if not xtickstep is None:
        plt.xticks(range(20, mat.shape[1], xtickstep))

    if not ytickstep is None:
        plt.yticks(range(2, mat.shape[0], ytickstep))

    cbar = fig.colorbar(cax, shrink=0.6, aspect=30, pad=0.02)
    ax.xaxis.set_ticks_position('bottom')
    cbar.set_label(cbar_label, **cbar_font)
    return fig, ax


# TODO flag for subsequent plots? how many times am i gonna have to do this?
# wrap a somewhat lower level function?
def orn_pn_matrix(mat, title='', xlabel='ORN receptor', luo_style=False):
    # TODO say which way the input dimensions should go / use dataframe
    """
    Args:
        luo_style (defaults to False): If True, uses jet(-like) color map and is
        displayed with odor varying along the horizontal axis.

        If False, will use a more perceptually flat color map and display
        transposed, to be able to read the odor names. (reading glomeruli
        easier?)
        TODO correct. transposing in other case
    """
    if luo_style:
        # TODO does this interfere w/ downstream functions (value passed in)?
        # test.
        mat = mat.T
        options = {
            'cmap': plt.cm.jet,
            'ylabel': 'Odor Index',
            'matrix_aspect': 'auto',
            'xtickstep': 20,
            'ytickstep': 2,
            'cbar_label': 'Firing Rate (Hz)'
        }

    else:
        # TODO TODO TODO how can y-labels both be about odor, if i'm transposing
        # in one of the cases? error?
        options = {
            'cmap': plt.cm.viridis,
            'ylabel': 'Odor',
            'matrix_aspect': 0.3,
            'cbar_label': 'Spike count change in 500ms(?) presentation'
        }
        # TODO should make x-axis font small enough that there is no overlapping
        # text

    # TODO why was fig2's cax also getting vmin=cbar.vim and vmax=cbar.vmax
    # args? (for fig1's cbar)
    # TODO way to get those values in advance, to have plotting of these two be
    # totally independent, for code re-use? (fig1 cbar.vim and cbar.vmax)
    # (if necessary...)

    _, ax = matrix_plot(mat, title=title, **options)

    if not luo_style:
        # keep all columns, but exclude last odor label
        # (because it is the spontaneous firing rate)
        plt.xticks(np.arange(n_pns))
        plt.yticks(np.arange(n_odors))
        # TODO make sure label order is correct / matshow directly for df?
        ax.set_xticklabels(hc06.columns.values, rotation='horizontal')
        ax.set_yticklabels(hc06.index.values[:-1], fontsize=6)


def kc_matrix(mat, title='', xlabel='Odor presented', ylabel='KC index',
              luo_style=False, cbar_label='Response probability'):
    # TODO say which way the input dimensions should go / use dataframe
    """
    Args:
        luo_style (defaults to False): If True, uses jet(-like) color map and is
        displayed with odor varying along the horizontal axis.

        ...
    """
    if luo_style:
        # TODO does this interfere w/ downstream functions (value passed in)?
        # test.
        options = {
            # plt.cm.afmhot may be closer?
            'cmap': plt.cm.hot,
            #'matrix_aspect': 'auto',
            'xtickstep': 20,
            # set to 500 in the paper, but theres no way they are displaying
            # the responses of 2500 cells?
            'ytickstep': 2
        }

    else:
        #mat = mat.T
        options = {
            'cmap': plt.cm.viridis
            #'matrix_aspect': 0.3,
        }
        # TODO should make x-axis font small enough that there is no overlapping
        # text

    # TODO why was fig2's cax also getting vmin=cbar.vim and vmax=cbar.vmax
    # args? (for fig1's cbar)
    # TODO way to get those values in advance, to have plotting of these two be
    # totally independent, for code re-use? (fig1 cbar.vim and cbar.vmax)
    # (if necessary...)

    _, ax = matrix_plot(mat, title=title, **options)

    '''
    if not luo_style:
        # TODO make sure this is correct no matter the possible transposes above
        plt.yticks(np.arange(n_odors))
        # TODO make sure label order is correct / matshow directly for df?
        ax.set_yticklabels(hc06.index.values[:-1], fontsize=6)
    '''

"""
Fig 1a: raw Hallem and Carlson 2006
"""
# exclude the last row, because those are spontaneous firing rates
delta_orn = hc06.as_matrix()[:-1,:]

# TODO just use drosolf?
# get the spontaneous rates so we can add them back
spont_orn = hc06.loc[hc06.index == 'spontaneous firing rate'
        ].as_matrix().flatten()

# recover actual firing rates (in the 500ms binning windows)
# TODO maybe keep it as a df?
orn = np.empty_like(delta_orn) * np.nan

# TODO numpy way to do this broadcasting?

# for each odor add the spontaneous rates to the difference observed for that
# odor adds the vector of spontaneous rates across all receptors to the odor
# specific
# vector of reponses across receptors
for i in range(delta_orn.shape[0]):
    orn[i,:] = delta_orn[i,:] + spont_orn

# It seems that what Luo et al do is set anything here that would go below zero
# to zero, but that might be wrong. Read more carefully, but they might not say.
orn[orn < 0] = 0
orn_matrix_title = 'Average ORN responses'
orn_pn_matrix(orn, title=orn_matrix_title, luo_style=True)
orn_pn_matrix(orn, title=orn_matrix_title)


"""
Fig 1b: simple model PN responses
(assuming now 1 receptor -> 1 PN (class). see note above)
"""

pn_xlabel = 'Cognate ORN receptor'
# units of Hz in the paper
# TODO is that correct here? (seems so)

# the maximum firing rate for PNs (assumed equal for all)
rmax = 165
# in the language of the Hill equation, w/o exponent it is as a Kd dissociation
# constant, and if it had an exponent, it would be like Ka, (Ka)^n = Kd = kd/ka
# Ka = ligand concentration producing half occupation of binding sites
# n > 1 ~= cooperative binding, but i guess here, it just recapitulates
# inflection of ORN -> PN function
sigma = 12
m = 0.05

def pn_responses_and_plots(lateral_inhibition=True):
    """
    """
    # model PN responses with no lateral inhibition
    if lateral_inhibition:
        # activity of any given PN is now dependent on sum of ORN activity
        # TODO make sure this broadcasting with newaxis is working correctly
        pn_responses = rmax * orn**1.5 / (sigma**1.5 + orn**1.5 + 
            (m * np.sum(orn, axis=1)[:,np.newaxis])**1.5)

        # TODO make less verbose? remove "Average"?
        pn_matrix_title = ('Average model PN responses (with lateral ' +
            'inhibition)')

    else:
        pn_responses = rmax * orn**1.5 / (sigma**1.5 + orn**1.5)

        # TODO is this PEP8? or is this a case where backslash is preferred?
        pn_matrix_title = ('Average model PN responses (no lateral ' +
            'inhibition)')

    # TODO TODO TODO add noise a la methods (maybe more important later?)
    # where is it introduced, again?

    # TODO TODO it would be really nice to have a function to group two figs
    # into a subplot..., for easier re-use of figure generating code (beyond
    # this code)

    # the authors used "PN index", but I like this better
    # TODO maybe alter this function to return subplots (within style)?
    orn_pn_matrix(pn_responses, title=pn_matrix_title, xlabel=pn_xlabel,
        luo_style=True)
    orn_pn_matrix(pn_responses, title=pn_matrix_title, xlabel=pn_xlabel)

    return pn_responses

pn_no_inh = pn_responses_and_plots(lateral_inhibition=False)
pn = pn_responses_and_plots()


"""
Fig 1C - E
"""

fig3 = plt.figure()
#fig3.title('Mean firing rates across odors')

a1 = plt.subplot(131)
orn_mean = np.mean(orn, axis=1)
plt.plot(np.arange(orn.shape[0]), orn_mean / np.mean(orn_mean), '.')
plt.title('ORN')
plt.ylabel('Firing rate (spikes/s)')

# this overall activity just seemed to high because i didn't yet normalize
a2 = plt.subplot(132)
pn_no_inh_mean = np.mean(pn_no_inh, axis=1)
plt.plot(np.arange(pn_no_inh.shape[0]), pn_no_inh_mean /
    np.mean(pn_no_inh_mean), '.')
plt.title('PN (no inhibition)')
a2.yaxis.set_ticklabels([])
plt.xlabel('Odor')

a3 = plt.subplot(133)
pn_mean = np.mean(pn, axis=1)
plt.plot(np.arange(pn.shape[0]), pn_mean / np.mean(pn_mean), '.')
plt.title('PN')
a3.yaxis.set_ticklabels([])

# get max ylim so we can rescale all subplots to have same max y limit
axs = [a1, a2, a3]
ymax = max([max(a.get_ylim()) for a in axs])

for a in axs:
    a.set_ylim(0, ymax)
    a.xaxis.set_ticklabels([])


"""
Fig 1F - H: skree plots of principal components of odor responses
skree plot = % variance "explained" as a function of the principal component
number

Note: they say "percentage of variances from a PCA analysis of the response used
in C-E" which seems to mean of the average responses. It is unclear that this is
meaningful, as apart from normalizing in the antennal lobe, the sum of the ORN
response is not really what is important. Is the sum of the PN response very
important, or always held approx constant?

I guess they are treating odors as observations?

"""
# TODO if i don't find code for these plots on old desktop, at least fix
# scale on skree plots (same scale on all). that might be only difference.

# 're-scaled' data, eigenvalues, and eigenvectors
sorn, orn_eval, orn_evec = pca(orn)
snlpn, nlpn_eval, nlpn_evec = pca(pn_no_inh)
spn, pn_eval, pn_evec = pca(pn)

"""
"PCA replaces original variables with new variables, called principal
 components, which are orthogonal (i.e. they have zero covariations) and have
 variances (called eigenvalues)..."

 The diagonal sums of original covariance matrix and covariance matrix of PCs, a
 diagonal matrix, are equal. This quantity is called the 'total variability.'
 Off-diagonal sum of covariance matrix is of course not guaranteed to be zero.
"""

'''
# can PCA perfectly reconstruct even random data with all components?
rand = np.random.rand(orn.shape[0], orn.shape[1])
rand_pca = PCA(n_components=rand.shape[1])
sk_rand = rand_pca.fit_transform(rand)
rand_reconstructed = rand_pca.inverse_transform(sk_rand)
# yes, it can
assert np.allclose(rand, rand_reconstructed)

# one less component, and it can't
rand_pca = PCA(n_components=(rand.shape[1] - 1))
sk_rand = rand_pca.fit_transform(rand)
rand_reconstructed = rand_pca.inverse_transform(sk_rand)
assert not np.allclose(rand, rand_reconstructed)
'''

# PCA from sklearn to compare output against
# not reducing # of components
sk_pca = PCA(n_components=orn.shape[1])
sk_sorn = sk_pca.fit_transform(orn)

# hmmm. well the norms are the same to high precision, despite different vectors
# ...why?
print(np.linalg.norm(orn_evec))
print(np.linalg.norm(sk_pca.components_))
#print(np.linalg.norm(orn_evec - sk_pca.components_))

print(orn_eval)
print(sk_pca.explained_variance_)
# Verdict: close, but always off a little bit. usually after 2 significant
# digits.  seems low by typical computer standards though... what explains the
# difference?

# works just fine
# assert np.allclose(sk_pca.inverse_transform(sk_sorn), orn)

# PCA should maintain the total variance after change of basis to that of the
# eigenvectors
assert np.isclose(np.cov(orn.T).diagonal().sum(),
    np.cov(sorn.T).diagonal().sum())
assert np.isclose(np.cov(pn_no_inh.T).diagonal().sum(),
    np.cov(snlpn.T).diagonal().sum())
assert np.isclose(np.cov(pn.T).diagonal().sum(), np.cov(spn.T).diagonal().sum())

# and off diagonal elements should be zero
# which means the sum of the whole matrix should be the sum of the diagonal
assert np.isclose(np.cov(sorn.T).diagonal().sum(), np.cov(sorn.T).sum())
assert np.isclose(np.cov(snlpn.T).diagonal().sum(), np.cov(snlpn.T).sum())
assert np.isclose(np.cov(spn.T).diagonal().sum(), np.cov(spn.T).sum())
# TODO move above in to test_PCA

# has its own assertion
test_pca(orn)
test_pca(pn_no_inh)
test_pca(pn)

# if our PCA is working correctly, generate the Skree plots
plt.figure()
plt.title('Fraction of total variance along each PC')
plt.xlabel('n-th largest eigenvalue of eigenvectors')
plt.ylabel("Fraction of total variance 'explained'")

plt.subplot(131)
plt.plot(orn_eval / orn_eval.sum(), '.')
plt.subplot(132)
plt.plot(nlpn_eval / nlpn_eval.sum(), '.')
plt.subplot(133)
plt.plot(pn_eval / pn_eval.sum(), '.')


# TODO move down?
def noisy_pns(trials=1):
    """
    """
    # TODO use trials to return tensor of activations, for vectorizing whole
    # simulation
    if trials != 1:
        raise NotImplementedError

    # TODO should i use the noisy PN responses for PN response plot too? seed?
    # for all subsequent transformations?
    # described in Methods section of main text
    # see: Sensory processing in the Drosophila antennal lobe increases the
    # reliability and separability of ensemble odor representations (Bhandawhat
    # et al., 2007) for possible justification
    sigma_pn_noise_hz = 10
    alpha_pn_noise_hz = 0.025

    # TODO why this function? what the deterministic part of tanh term look
    # like?
    # they didn't say the noise was normal, they just said zero mean and unit
    # variance, but i assumed
    # r_pn in their equations (after transformation to add noise)
    # some experimental justification for this noise distribution?
    return (pn + (sigma_pn_noise_hz * np.tanh(alpha_pn_noise_hz * pn) *
        np.random.normal(loc=0.0, scale=1.0, size=pn.shape))).T

"""
Fig 2: Responses of model LHNs

'...we constructed 110 LHNs, each selective for a different one of the 110
odorants... We are not suggesting that each of these odors generates an innate
behavbiort. Instead, the model LHNs are used...to demonstrate how...selective
LHNs can be constructed.'
"""




"""
Fig 3: Model KC responses
"""
n_kcs = 2500

# think about naming conventions here...
# TODO rewrite to allow for distribution, rather than integer n_pns_per_kc
# input, or make two functions
def pn_to_kc_inputs(n_pns_per_kc=5, n_kcs=n_kcs, verbose=False):
    """
    """
    # from methods: "we chose the n synaptic connections for the KCs randomly
    # and drew their weights, denoted by the vector w, from a uniform
    # distribution between 0 and 1."

    # TODO maybe make this more accurate, using information from subsequent
    # studies?  some stuff that could help exists, right?
    # TODO break into function? maybe just when i copy into drosolf
    n_pns_per_kc = 5
    # TODO w/ replacement or not? need diff function to do w/o replacement?
    nonzero_weights = np.random.randint(n_pns, size=(n_kcs, n_pns_per_kc))
    # w in their equations (transposed?)
    pn_to_kc_weights = np.zeros((n_kcs, n_pns))
    #pn_to_kc_weights[nonzero_weights] = np.random.uniform(
    #    size=nonzero_weights.shape)

    # TODO do different trials of theirs only differ with the activity of the
    # PNs, or do they also re-draw the PN inputs to the KCs?

    # Sampling glomeruli WITH REPLACEMENT. Not obvious whether Luo et al. sample
    # with replacement or not.
    counts = dict()
    # TODO how to vectorize this?
    for k in range(n_kcs):
        distinct_pn_inputs = set()
        for i in range(n_pns_per_kc):
            # so that if we same from the same glomerulus twice, we increase the
            # weight
            pn_to_kc_weights[k, nonzero_weights[k,i]] += np.random.uniform()
            distinct_pn_inputs.add(nonzero_weights[k,i])

        if verbose:
            n_distinct_inputs = len(distinct_pn_inputs)
            if n_distinct_inputs in counts:
                counts[n_distinct_inputs] += 1
            else:
                counts[n_distinct_inputs] = 1

    if verbose:
        print('counts:', counts)

    # same?
    '''
    pn_to_kc_weights2 = np.zeros((n_kcs, n_pns))
    for i in range(n_pns_per_kc):
        pn_to_kc_weights2[:, nonzero_weights[:,i]] = 
    '''
    return pn_to_kc_weights


def kc_activations(pns=None, pn_to_kc_weights=None, checks=False):
    """
    Args:
        pns ():
        pn_to_kc_weights (): 
        checks (bool): 
    """
    #print('pns.shape:', pns.shape)

    # from SI: (.T to indicate transpose, * for matrix multiplication) "the
    # total KC input in our model is I=W.T * r_pn - v * r_in, with r_in the
    # firing rate of one or more globally acting interneurons connected to each
    # KC through a synapse of strength v and driven by PNs through synapses
    # W_in.T so that r_in = W_in.T * r_pn. Defining r_hat = <r_pn> / |<r_pn>|,
    # where the brackets denote an average across odors, we set v = W.T * r_hat
    # and W_in = r_hat. Then,
    # I = W.T * r_pn - v * r_in = W.T * (r_pn - (r_hat.T * r_pn) * r_hat),
    # which removes the projection of the PN rates along the direction of their
    # mean. Note that if we average over all odors,
    # <I> = W.T * <r_pn> - v * <r_in> = 0"
    # TODO is that last consequence (just above) sensible?
    # TODO is this really subtracting first PC?
    # TODO is this model totally linear? (i guess this is all before some
    # threshold?)

    # TODO TODO vectorize to include a trials dimension for these calculations,
    # if possible

    odor_averaged_pn_responses = np.mean(noisy_pn, axis=1)
    if checks:
        assert len(odor_averaged_pn_responses.shape) == 1
        assert odor_averaged_pn_responses.shape[0] == n_pns

    odor_averaged_pn_responses = np.expand_dims(odor_averaged_pn_responses,
        axis=1)
    #print(odor_averaged_pn_responses)
    #print(odor_averaged_pn_responses.shape)

    # r_hat in their equations
    # TODO is their denominator definitely this norm? probably?
    normalized_pn_responses = (odor_averaged_pn_responses /
        np.linalg.norm(odor_averaged_pn_responses))
    #print('normalized_pn_responses.shape:', normalized_pn_responses.shape)

    # w_in (transposed?) in their equations
    pn_to_inh_weights = normalized_pn_responses
    #print('pn_to_inh_weights.shape:', pn_to_inh_weights.shape)

    # r_in in their equations
    inhibitory_neurons_activation = np.dot(pn_to_inh_weights.T, noisy_pn)
    #print('inhibitory_neurons_activation.shape:',
    #    inhibitory_neurons_activation.shape)

    #print('pn_to_kc_weights.shape:', pn_to_kc_weights.shape)
    # v in their equations
    # TODO correct? scalar or not?
    inhibition_strength = np.dot(pn_to_kc_weights, normalized_pn_responses)
    #print('inhibition_strength.shape:', inhibition_strength.shape)

    kc_activation = (np.dot(pn_to_kc_weights, noisy_pn) - 
        np.dot(inhibition_strength, inhibitory_neurons_activation))

    if checks:
        # checking this equals their equivalent form, largely to gaurd against
        # having made dimension mismatch errors
        synonym_kc_activation = np.dot(pn_to_kc_weights, (noisy_pn -
            np.dot(np.dot(odor_averaged_pn_responses.T, noisy_pn).T,
            odor_averaged_pn_responses.T).T))

        print('kc_activation.shape:', kc_activation.shape)
        '''
        print(synonym_kc_activation.shape)

        print(kc_activation[0,:])
        print(synonym_kc_activation[0,:])

        print(kc_activation[-1,:])
        print(synonym_kc_activation[-1,:])
        # TODO recheck above math. identify errors.
        assert np.allclose(kc_activation, synonym_kc_activation)
        '''

        # my original attempt:
        #np.dot(pn_to_kc_weights, (noisy_pn -
        #    np.dot(np.dot(odor_averaged_pn_responses.T, noisy_pn),
        #    odor_averaged_pn_responses)))

        odor_averaged_kc_activation = np.mean(kc_activation, axis=1)
        assert len(odor_averaged_kc_activation.shape) == 1
        assert odor_averaged_kc_activation.shape[0] == n_kcs
        # their assertion (do algebra to get this consequence)
        assert np.allclose(odor_averaged_kc_activation, 0.0)

    return kc_activation


# TODO probably turn this into a function like noisy_pns
pn_to_kc_weights = pn_to_kc_inputs(n_pns_per_kc=5)
kc_activation_trials = []
for t in range(simulated_trials):
    if t == 0:
        checks = True
    else:
        checks = False

    noisy_pn = noisy_pns()
    kc_activation_trials.append(kc_activations(pns=noisy_pn, 
        pn_to_kc_weights=pn_to_kc_weights, checks))
kc_activation_trials = np.stack(kc_activation_trials)
print('kc_activation_trials.shape:', kc_activation_trials.shape)

# 3A: "response probabilities" of model KCs, each receiving input from n PNs and
# global inhibition
# TODO what is this actually a plot of? the bars are too wide to actually fit
# 2500 cells, unless maybe light colors overwrite neighboring darker colors, and
# everything is plotted much wider than it should be...
# like, this figure is just a hair under 40mm and 40mm / 2500 = 0.016mm,
# yet each light bar is about 0.7mm wide (maybe a little less, >= 0.65mm)
# which only leaves room for about 67 cells, best case
# a random sample would make sense... is that what it is?

kc_threshold_percentile = 0.95
# TODO should threshold only be determined after taking a number of "trials"?
# TODO so is this threshold used for all further experiments on simulated KCs?
# TODO and if i'm not going to use a bunch of trials to calculate the threshold
# should i at least check whether the 0.95/0.05 proportion holds in the
# population of simulations?
kc_response_threshold = np.sort(kc_activation.flatten())[
    int(round(kc_threshold_percentile * kc_activation.size))]
print('kc_response_threshold:', kc_response_threshold)

assert np.isclose(np.sum(kc_activation < kc_response_threshold) 
    / kc_activation.size, kc_threshold_percentile)
assert np.isclose(np.sum(kc_activation >= kc_response_threshold) 
    / kc_activation.size, 1 - kc_threshold_percentile)

kc_response_probability = np.mean(kc_activation_trials > kc_response_threshold,
    axis=0)

# i think this is about how many there are
#apparent_number_kcs_plotted = 80
apparent_number_kcs_plotted = 110
kcs_to_plot = np.random.choice(n_kcs, apparent_number_kcs_plotted,
    replace=False)

# take the random sample within the plotting function?
kc_matrix(kc_response_probability[kcs_to_plot, :], luo_style=True)
kc_matrix(kc_response_probability[kcs_to_plot, :])


# 3B: the number of missed odors as a function of # of PNs each KC receives
# input from
pn_to_kc_connections = np.arange(20) + 1

# delete me
plt.close('all')

missed_odors = np.arange(20)
silent_kcs = np.arange(20)

def three_b(xs, ys, ylabel, title=''):
    _ = plt.figure()
    plt.plot(xs, ys, 'r.')
    plt.title(title, fontweight='bold', y=1.01)
    plt.xlabel('Number of PN to KC connections', x_axes_font)
    plt.ylabel(ylabel, y_axes_font)

three_b(pn_to_kc_connections, missed_odors, 'Missed Odors')
three_b(pn_to_kc_connections, silent_kcs, 'Silent KCs')


"""
Fig 4: LHN and KC responses for different concentrations
"""


"""
Fig 5: Effect of feedforward nonlinearity and lateral suppression on LHN and KC
responses.
"""




plt.show()
