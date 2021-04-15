#!/usr/bin/env python3

import numpy as np
from scipy import linalg 
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

import drosolf.orns


# TODO argument for keeping configuration up top, is that they don't need to be
# passed to each function, and any new function here can have access to them by
# default? downsides besides style? alternatives? just suck it up and pass a
# configuration dict to each function?
n_kcs = 2500
sample_pns_with_replacement = True

# If True, does not add the noise they added in the paper.
deterministic = False

# number of trials used to generate "response probability" plots
# am i misunderstanding?
simulated_trials = 100
exclude_pheromone_receptors = False

# TODO maybe default to this? to reduce deviation from mean performance seed /
# general lack of averaging may cause? experiment / discuss
regenerate_connectivity_each_trial = False

# How literally should I take "even though we used the same threshold for *all*
# KCs"? a threshold defined as the 95th percentile of the responses, after
# building the PN->KC weights and generating activations for however many
# trials? I would hope it's the former? Thought this is one possible difference.
all_use_five_input_threshold = False

###############################################################################
# PN properties
###############################################################################
# the maximum firing rate for PNs (assumed equal for all)
rmax = 165
# in the language of the Hill equation, w/o exponent it is as a Kd dissociation
# constant, and if it had an exponent, it would be like Ka, (Ka)^n = Kd = kd/ka
# Ka = ligand concentration producing half occupation of binding sites
# n > 1 ~= cooperative binding, but i guess here, it just recapitulates
# inflection of ORN -> PN function
sigma = 12
m = 0.05

# described in Methods section of main text
# see: Sensory processing in the Drosophila antennal lobe increases the
# reliability and separability of ensemble odor representations (Bhandawhat
# et al., 2007) for possible justification
sigma_pn_noise_hz = 10
alpha_pn_noise_hz = 0.025
###############################################################################

if (deterministic or not sample_pns_with_replacement or
    exclude_pheromone_receptors):
    raise NotImplementedError

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

# TODO TODO refactor this stuff out. use drosolf
"""
A little bit about the Hallem and Carlson 2006 dataset:
    -each odor pulse is 500ms long (exceptions?)
    -responses (other than spontaneous) are # of spikes in those 500ms -
     spontaneous
    -where spontaneous is just count in 500ms without stimulation
    -flies are <4 weeks old (but how young on average?)
    -24 ml/s carrier with 5.9 ml/s odor stream
    -female flies (??)
"""
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
# TODO TODO refactor configuration. dict?
n_pns = len(hc06.columns)
n_odors = len(hc06.index) - 1

# exclude the last row, because those are spontaneous firing rates
delta_orn = hc06.as_matrix()[:-1,:]

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


# TODO move to a test directory?
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

def fig_one_a():
    orn_matrix_title = 'Average ORN responses'
    orn_pn_matrix(orn, title=orn_matrix_title, luo_style=True)
    orn_pn_matrix(orn, title=orn_matrix_title)

def pn_responses_and_plots(lateral_inhibition=True):
    """
    Fig 1b: simple model PN responses
    (assuming now 1 receptor -> 1 PN (class). see note above)
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
        pn_matrix_title = ('Average model PN responses (no lateral ' +
            'inhibition)')
    # TODO TODO it would be really nice to have a function to group two figs
    # into a subplot..., for easier re-use of figure generating code (beyond
    # this code)

    # the authors used "PN index", but I like this better
    # TODO maybe alter this function to return subplots (within style)?
    pn_xlabel = 'Cognate ORN receptor'
    orn_pn_matrix(pn_responses, title=pn_matrix_title, xlabel=pn_xlabel,
        luo_style=True)
    orn_pn_matrix(pn_responses, title=pn_matrix_title, xlabel=pn_xlabel)

    return pn_responses

def fig_one_b():
    pn_no_inh = pn_responses_and_plots(lateral_inhibition=False)
    pn = pn_responses_and_plots()
    # TODO TODO are PN responses actually more decorrelated than the ORNs in
    # this model? or is it only subtracting the PN mean at the level of the KC
    # input that does any decorrelation? where does the decorrelation come from?
    # and which papers support this again?
    # (they claim it in the last paragraph before the "Concentration dependence"
    # section)

def fig_one_c_thru_e(orn, pn_no_inh, pn):
    # TODO refactor / at least add inner plotting fn
    """
    Fig 1C - E
    """
    fig3 = plt.figure()
    #fig3.title('Mean firing rates across odors')

    # TODO relative to averages like in paper
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


def fig_one_f_thru_h(orn, pn_no_inh, pn):
    """
    Fig 1F - H: skree plots of principal components of odor responses
    skree plot = % variance "explained" as a function of the principal component
    number

    Note: they say "percentage of variances from a PCA analysis of the response
    used in C-E" which seems to mean of the average responses. It is unclear
    that this is meaningful, as apart from normalizing in the antennal lobe, the
    sum of the ORN response is not really what is important. Is the sum of the
    PN response very important, or always held approx constant?

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

     The diagonal sums of original covariance matrix and covariance matrix of
     PCs, a diagonal matrix, are equal. This quantity is called the 'total
     variability.' Off-diagonal sum of covariance matrix is of course not
     guaranteed to be zero.
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

    # hmmm. well the norms are the same to high precision, despite different
    # vectors ...why?
    print(np.linalg.norm(orn_evec))
    print(np.linalg.norm(sk_pca.components_))
    #print(np.linalg.norm(orn_evec - sk_pca.components_))

    print(orn_eval)
    print(sk_pca.explained_variance_)
    # Verdict: close, but always off a little bit. usually after 2 significant
    # digits.  seems low by typical computer standards though... what explains
    # the difference?

    # works just fine
    # assert np.allclose(sk_pca.inverse_transform(sk_sorn), orn)

    # PCA should maintain the total variance after change of basis to that of
    # the eigenvectors
    assert np.isclose(np.cov(orn.T).diagonal().sum(),
        np.cov(sorn.T).diagonal().sum())
    assert np.isclose(np.cov(pn_no_inh.T).diagonal().sum(),
        np.cov(snlpn.T).diagonal().sum())
    assert np.isclose(np.cov(pn.T).diagonal().sum(),
                      np.cov(spn.T).diagonal().sum())

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
def noisy_pns(alpha=alpha_pn_noise_hz, sigma=sigma_pn_noise_hz, trials=1):
    # TODO follow consistent pattern to other functions, re: taking pn response
    # argument, but otherwise generating it? way to have less boilerplate?
    """
    """
    # TODO use trials to return tensor of activations, for vectorizing whole
    # simulation
    if trials != 1:
        raise NotImplementedError

    # TODO why this function? what the deterministic part of tanh term look
    # like?
    # they didn't say the noise was normal, they just said zero mean and unit
    # variance, but i assumed
    # r_pn in their equations (after transformation to add noise)
    # some experimental justification for this noise distribution?

    # TODO plot version of resonses w/ noise added for sanity checking?
    return (pn + (sigma * np.tanh(alpha * pn) *
        np.random.normal(loc=0.0, scale=1.0, size=pn.shape))).T

# think about naming conventions here...
# TODO rewrite to allow for distribution, rather than integer n_pns_per_kc
# input, or make two functions
def pn_to_kc_inputs(n_pns_per_kc=5, n_kcs=n_kcs, verbose=False):
    """
    """
    # from methods: "we chose the n synaptic connections for the KCs randomly
    # and drew their weights, denoted by the vector w, from a uniform
    # distribution between 0 and 1."
    # TODO maybe make this connectivity more accurate, using information from
    # subsequent studies?  some stuff that could help exists, right?

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

def kc_activations(n_kcs=n_kcs, pns=None, n=None, pn_to_kc_weights=None,
                   inhibition=True, checks=False):
    """
    Args:
        pns (np.ndarray): (optional) If passed, model KC responses are computed
            with these PN responses. By default, model PNs responses are
            generated with the function noisy_pns().

            TODO say which dimensions / labels are expected

        n (int): The number of PNs each KC will receive input from, each weight
            drawn from the uniform distribution on [0,1). Only pass either this
            or pn_to_kc_weights. If neither is passed, the function behaves as
            if it had n set to 5, as the parameterization used for most of the
            paper.

        pn_to_kc_weights (np.ndarray): The PN to KC weights to use, if
            predefined. Only pass either this or n.

        inhibition (bool): Whether to include their "global" inhibition term
            False should be able to recapitulate S2, when fed through the rest
            of the analysis.

        checks (bool): Whether to check dimensions and some of the identities in
            the supplement.
    """
    if pns is None:
        pns = noisy_pns()

    if pn_to_kc_weights is None:
        if n is None:
            # the value the paper ultimately settled on for most of the figures
            n = 5
        pn_to_kc_weights = pn_to_kc_inputs(n_pns_per_kc=n)
        
    elif not (n is None or pn_to_kc_weights is None):
        raise ValueError('ambiguous. only pass in either n or pn_to_kc_weights')

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

    odor_averaged_pn_responses = np.mean(pns, axis=1)
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
    inhibitory_neurons_activation = np.dot(pn_to_inh_weights.T, pns)
    #print('inhibitory_neurons_activation.shape:',
    #    inhibitory_neurons_activation.shape)

    #print('pn_to_kc_weights.shape:', pn_to_kc_weights.shape)
    # v in their equations
    # TODO correct? scalar or not?
    inhibition_strength = np.dot(pn_to_kc_weights, normalized_pn_responses)
    #print('inhibition_strength.shape:', inhibition_strength.shape)

    kc_activation = (np.dot(pn_to_kc_weights, pns) - 
        np.dot(inhibition_strength, inhibitory_neurons_activation))

    if checks:
        # checking this equals their equivalent form, largely to gaurd against
        # having made dimension mismatch errors
        synonym_kc_activation = np.dot(pn_to_kc_weights, (pns -
            np.dot(np.dot(odor_averaged_pn_responses.T, pns).T,
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
        # TODO if this inhibition is not equivalent to "subtracting first PC"
        # then prove it. compute PCs, calculate with that, show different.

        # my original attempt:
        #np.dot(pn_to_kc_weights, (pns -
        #    np.dot(np.dot(odor_averaged_pn_responses.T, pns),
        #    odor_averaged_pn_responses)))

        odor_averaged_kc_activation = np.mean(kc_activation, axis=1)
        assert len(odor_averaged_kc_activation.shape) == 1
        assert odor_averaged_kc_activation.shape[0] == n_kcs
        # their assertion (do algebra to get this consequence)
        assert np.allclose(odor_averaged_kc_activation, 0.0)

    return kc_activation

# TODO make another fn called kc_responses that chains the above with this?
def calc_response_prob(trial_fn=lambda: kc_activations(),
                       simulated_trials=simulated_trials,
                       response_threshold=None):
    """
    Args:
        trial_fn (callable): Each call should return an independent simulated
                             trial. If trial_fn_parameter is None
        simulated_trials (int): number of trials to generate
                                -maybe rename?
        response_threshold (number): 

    Returns:
        
    """
    # was it reasonable to not generate the pn_to_kc_weights here?
    # seems to go hand-in-hand with not being able to control the checks arg
    # here...
    trials = []
    for t in range(simulated_trials):
        if t == 0:
            checks = True
        else:
            checks = False
        trials.append(trial_fn())
    trials = np.stack(trials)

    if response_threshold is None:
        threshold_percentile = 0.95
        # TODO allow suppression
        print('determining inverse-CDF of {} for response threshold...'.format(
            threshold_percentile))
        response_threshold = np.sort(trials.flatten())[
            int(round(threshold_percentile * trials.size))]

        assert np.isclose(np.sum(trials < response_threshold) 
            / trials.size, threshold_percentile)
        assert np.isclose(np.sum(trials >= response_threshold) 
            / trials.size, 1 - threshold_percentile)
        return_none = False

    else:
        # Want to return None for response threshold in this case, to not give
        # the false impression it was calculated in here.
        return_none = True

    response_probability = np.mean(trials > response_threshold, axis=0)
    return response_probability, None if return_none else response_threshold

def responders(response_probability):
    """
    """
    # not measuring the same thing as kc_response_threshold above
    response_criteria = 0.50
    # "we define a neuron as responding if it receives an above-threshold input
    # in at least 50% of trials." "fairly stringent"

    # TODO why was this working with kc_response_probability (a typo), when that
    # variable is not defined until below? true, that variable should exist by
    # the time this function is called... so i guess i misunderstood how
    # Python's scoping works?
    responses_above_criteria = response_probability >= response_criteria
    return responses_above_criteria

def responses_along_axis(responses, axis, expected_size):
    """
    Args:
        responses (array-like): a boolean type array to be checked

        axis (int): axis along which to check for a complete lack of responses

        expected_size (int): for checking result it expected size. somewhat
        tautological / equivalent to checking dimensions of responses.
    """
    assert responses.dtype == np.dtype('bool')
    no_responses = np.logical_not(np.any(responses, axis=axis))
    assert no_responses.size == expected_size
    return np.sum(no_responses)

def missed_odors(responses):
    """
    """
    # TODO use variable for the odor / kc axis?
    return responses_along_axis(responses, 0, n_odors)

def silent_kcs(responses):
    """
    Args:
        responses 
    """
    return responses_along_axis(responses, 1, n_kcs)

def kc_response_summary(responses, weights=None):
    """
    """
    raise NotImplementedError

def vary_n_pns_per_kc(verbose=True):
    """Seeing how "quality" of sparse representation varies as the number of PN
    inputs to the KCs, with quality achieved by minimizing both missed odors
    and silent KCs, as defined above.
    """
    pn_to_kc_connections = [n + 1 for n in range(20)]
    n_missed_odors = []
    n_silent_kcs = []

    if all_use_five_input_threshold:
        ok_input_number = 5
        # move 5 to the front of the list, so we can calculate its threshold and
        # save it for models using a different number of PN inputs to each KC
        assert ok_input_number in pn_to_kc_connections
        pn_to_kc_connections.remove(ok_input_number)
        pn_to_kc_connections.insert(0, ok_input_number)

    response_threshold = None
    # TODO TODO maybe plot weight matrices as a crude debugging step? +
    # activations + distributions of # responders to each odor + # odors evoking
    # a response across cells? + print different thresholds (+ distribution of
    # activations?)
    for n in pn_to_kc_connections:
        if regenerate_connectivity_each_trial:
            # kc_activations will generate the weights when pn_to_kc_weights is
            # None (the default)
            pn_kc_weights = None
        else:
            pn_kc_weights = pn_to_kc_inputs(n_pns_per_kc=n)

        print(('simulating {} trials of KC activation to Hallem odors, each ' +
               'KC receiving {} input(s).').format(simulated_trials, n))

        # TODO maybe refactor again? i feel like i've make my control over when
        # to check kind of cumbersome. maybe likewise for the weights, but i do
        # like having default for everything, so that each function can pretty
        # much just be called on it's own, to make it easier for someone to play
        # around with the functions...
        kc_response_probability, last_response_threshold = calc_response_prob(
            trial_fn=lambda: kc_activations(pn_to_kc_weights=pn_kc_weights,
                                            checks=True if n == 0 else False),
            response_threshold=response_threshold)


        # because this is used for most of the plots
        if n == 5:
            five_inputs_kc_p_response = kc_response_probability
            if all_use_five_input_threshold:
                print('using response threshold determined with 5 input ' + 
                      'model for all other models!')
                response_threshold = last_response_threshold

        responses = responders(kc_response_probability)
        n_missed_odors.append(missed_odors(responses))
        n_silent_kcs.append(silent_kcs(responses))

    # TODO break into two functions, one ending here?

    # 3A: "response probabilities" of model KCs, each receiving input from n PNs
    # and global inhibition
    # TODO what is this actually a plot of? the bars are too wide to actually
    # fit 2500 cells, unless maybe light colors overwrite neighboring darker
    # colors, and everything is plotted much wider than it should be...
    # like, this figure is just a hair under 40mm and 40mm / 2500 = 0.016mm,
    # yet each light bar is about 0.7mm wide (maybe a little less, >= 0.65mm)
    # which only leaves room for about 67 cells, best case
    # a random sample would make sense... is that what it is?

    # i think this is about how many there are
    apparent_number_kcs_plotted = 80
    kcs_to_plot = np.random.choice(n_kcs, apparent_number_kcs_plotted,
        replace=False)

    # take the random sample within the plotting function?
    kc_matrix(five_inputs_kc_p_response[kcs_to_plot, :], luo_style=True)
    kc_matrix(five_inputs_kc_p_response[kcs_to_plot, :])

    # 3B: the number of missed odors as a function of # of PNs each KC receives
    # input from

    # TODO TODO TODO i was able to roughly match the silent KC graph, but the
    # missed odors is all flat at zero, with 1 missed odor at 1 input per KC.
    # why is that?  do i need to use the same threshold (determined at n=5) for
    # everything, or something?
    print('n_missed_odors:', n_missed_odors)
    print('n_silent_kcs:', n_silent_kcs)

    # TODO TODO TODO look at distribution of number of inputs required for the
    # threshold response (possible?)? and distribution not leading to a
    # response?  (maybe the uniform distribution has some subtle consequences?)

    # TODO do i also get an average of ~125 cells responding per odor ("and min
    # of 2 cells", over all 110 odors) (this is all for the 5 input case)

    def three_b(xs, ys, ylabel, title=''):
        _ = plt.figure()
        plt.plot(xs, ys, 'r.')
        plt.title(title, fontweight='bold', y=1.01)
        plt.xlabel('Number of PN to KC connections', x_axes_font)
        plt.ylabel(ylabel, y_axes_font)
        # TODO fix x tick marks to only display integers

    three_b(pn_to_kc_connections, n_missed_odors, 'Missed Odors')
    three_b(pn_to_kc_connections, n_silent_kcs, 'Silent KCs')


if __name__ == '__main__':
    sns.set_style('dark')
    np.random.seed(1118)
    """
    Fig 1a: raw Hallem and Carlson 2006
    """

    """
    Fig 2: Responses of model LHNs

    '...we constructed 110 LHNs, each selective for a different one of the 110
    odorants... We are not suggesting that each of these odors generates an
    innate behavbiort. Instead, the model LHNs are used...to demonstrate
    how...selective LHNs can be constructed.'
    """

    """
    Fig 3: Model KC responses
    """
    vary_n_pns_per_kc()

    """
    Fig 4: LHN and KC responses for different concentrations
    """


    """
    Fig 5: Effect of feedforward nonlinearity and lateral suppression on LHN and
    KC responses.
    """
    plt.show()
