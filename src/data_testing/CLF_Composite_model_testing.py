#! /usr/bin/env python
# -*- coding: utf-8 -*-

# Victor Calderon
# Created      : 2019-01-20
# Last Modified: 2019-01-20
# Vanderbilt University
from __future__ import absolute_import, division, print_function 
__author__     = ['Victor Calderon']
__copyright__  = ["Copyright 2018 Victor Calderon, "]
__email__      = ['victor.calderon@vanderbilt.edu']
__maintainer__ = ['Victor Calderon']
"""
Compares different types of scatter using the CLF formalism to compute.

"""
# Importing Modules
from cosmo_utils       import mock_catalogues as cm
from cosmo_utils       import utils           as cu
from cosmo_utils.utils import file_utils      as cfutils
from cosmo_utils.utils import file_readers    as cfreaders
from cosmo_utils.utils import work_paths      as cwpaths
from cosmo_utils.utils import stats_funcs     as cstats
from cosmo_utils.utils import geometry        as cgeom
from cosmo_utils.mock_catalogues import catls_utils as cmcu
from cosmo_utils.mock_catalogues import abundance_matching as cuam

from src.ml_tools import ReadML

import numpy as num
import math
import os
import sys
import pandas as pd
import matplotlib
matplotlib.use( 'Agg' )
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
plt.rc('text', usetex=True)
import seaborn as sns
#sns.set()
from tqdm import tqdm

# Halotools
from halotools.empirical_models import PrebuiltHodModelFactory
from halotools.sim_manager import CachedHaloCatalog, DownloadManager

import astropy.cosmology   as astrocosmo
import astropy.constants   as ac
import astropy.units       as u
import astropy.table       as astro_table
import astropy.coordinates as astrocoord
import hmf

# Extra-modules
import argparse
from argparse import ArgumentParser
from argparse import HelpFormatter
from operator import attrgetter
from tqdm import tqdm
from datetime import datetime
from scipy.interpolate import interp1d

## Functions

# Extra-modules
import argparse
from argparse import ArgumentParser
from argparse import HelpFormatter
from operator import attrgetter
from tqdm import tqdm

## Functions
class SortingHelpFormatter(HelpFormatter):
    def add_arguments(self, actions):
        """
        Modifier for `argparse` help parameters, that sorts them alphabetically
        """
        actions = sorted(actions, key=attrgetter('option_strings'))
        super(SortingHelpFormatter, self).add_arguments(actions)

def _str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def _check_pos_val(val, val_min=0):
    """
    Checks if value is larger than `val_min`

    Parameters
    ----------
    val : `int` or `float`
        Value to be evaluated by `val_min`

    val_min: `float` or `int`, optional
        minimum value that `val` can be. This value is set to `0` by default.

    Returns
    -------
    ival : `float`
        Value if `val` is larger than `val_min`

    Raises
    -------
    ArgumentTypeError : Raised if `val` is NOT larger than `val_min`
    """
    ival = float(val)
    if ival <= val_min:
        msg  = '`{0}` is an invalid input!'.format(ival)
        msg += '`val` must be larger than `{0}`!!'.format(val_min)
        raise argparse.ArgumentTypeError(msg)

    return ival

def get_parser():
    """
    Get parser object for `eco_mocks_create.py` script.

    Returns
    -------
    args: 
        input arguments to the script
    """
    ## Define parser object
    description_msg = 'Description of Script'
    parser = ArgumentParser(description=description_msg,
                            formatter_class=SortingHelpFormatter,)
    ## 
    parser.add_argument('--version', action='version', version='%(prog)s 1.0')
    ## Cosmology Choice
    parser.add_argument('-cosmo',
                        dest='cosmo_choice',
                        help='Choice of Cosmology',
                        type=str,
                        choices=['LasDamas', 'Planck'],
                        default='LasDamas')
    ## Removes the file for the "Halo Mass Function"
    parser.add_argument('-remove_hmf',
                        dest='remove_hmf',
                        help='Delete HMF file',
                        type=_str2bool,
                        default=False)
    ## Halo mass function (HMF)
    parser.add_argument('-hmf',
                        dest='hmf_model',
                        help='Halo Mass Function choice',
                        type=str,
                        choices=['warren','tinker08'],
                        default='warren')
    # Luminosity cut for galaxies in log(L) space
    parser.add_argument('-log_lum_cut',
                        dest='log_lum_cut',
                        help='Luminosity cut for galaxies in log(L)',
                        type=_check_pos_val,
                        default=9.0)
    # Fiducial value for `sigma_c`
    parser.add_argument('-sigma_c_fid',
                        dest='sigma_c_fid',
                        help='Fiducial value for sigma_c',
                        type=_check_pos_val,
                        default=0.1417)
    ## Program message
    parser.add_argument('-progmsg',
                        dest='Prog_msg',
                        help='Program message to use throught the script',
                        type=str,
                        default=cfutils.Program_Msg(__file__))
    ## Parsing Objects
    args = parser.parse_args()

    return args


def directory_skeleton(proj_dict):
    """
    Creates the directory skeleton for the current project

    Parameters
    ----------
    param_dict : `dict`
        Dictionary with `project` variables

    proj_dict : `dict`
        Dictionary with info of the project that uses the
        `Data Science` Cookiecutter template.

    Returns
    ---------
    proj_dict : `dict`
        Dictionary with current and new paths to project directories
    """
    ## Output data - directory
    # outdir  = os.path.join( proj_dict['proc_dir'],
    #                         'clf_models')
    ## Figure directory
    fig_dir = os.path.join( proj_dict['plot_dir'],
                            'clf_models')
    # HMF directory
    mf_dir = os.path.join(  proj_dict['int_dir'],
                            'MF')
    # Creating directory
    dir_arr = [fig_dir, mf_dir]
    for dir_ii in dir_arr:
        if not (os.path.exists(dir_ii)):
            cfutils.Path_Folder(dir_ii)
    #
    # Adding to main project dictionary
    proj_dict['fig_dir'] = fig_dir
    proj_dict['mf_dir' ] = mf_dir

    return proj_dict

## --------- Cosmology and Halo Mass Function ------------##

def cosmo_create(cosmo_choice='Planck', H0=100., Om0=0.25, Ob0=0.04,
    Tcmb0=2.7255):
    """
    Creates instance of the cosmology used throughout the project.

    Parameters
    ----------
    cosmo_choice: string, optional (default = 'Planck')
        choice of cosmology
        Options:
            - Planck: Cosmology from Planck 2015
            - LasDamas: Cosmology from LasDamas simulation

    h: float, optional (default = 1.0)
        value for small cosmological 'h'.

    Returns
    ----------                  
    cosmo_obj: astropy cosmology object
        cosmology used throughout the project
    """
    ## Checking cosmology choices
    cosmo_choice_arr = ['Planck', 'LasDamas']
    assert(cosmo_choice in cosmo_choice_arr)
    ## Choosing cosmology
    if cosmo_choice == 'Planck':
        cosmo_model = astrocosmo.Planck15.clone(H0=H0)
    elif cosmo_choice == 'LasDamas':
        cosmo_model = astrocosmo.FlatLambdaCDM(H0=H0, Om0=Om0, Ob0=Ob0, 
            Tcmb0=Tcmb0)
    ## Cosmo Paramters
    cosmo_params         = {}
    cosmo_params['H0'  ] = cosmo_model.H0.value
    cosmo_params['Om0' ] = cosmo_model.Om0
    cosmo_params['Ob0' ] = cosmo_model.Ob0
    cosmo_params['Ode0'] = cosmo_model.Ode0
    cosmo_params['Ok0' ] = cosmo_model.Ok0
    ## HMF Cosmo Model
    cosmo_hmf = hmf.cosmo.Cosmology(cosmo_model=cosmo_model)

    return cosmo_model, cosmo_hmf

def hmf_calc(cosmo_model, cosmo_choice, proj_dict, 
    Mmin=10, Mmax=16, dlog10m=1e-3, hmf_model='warren',
    remove_hmf=False, delimiter=',', ext='csv', 
    Prog_msg = '1>>> '):#Prog_msg=cu.Program_Msg(__file__)):
    """
    Creates file with the desired mass function

    Parameters
    ----------
    cosmo_obj: astropy cosmology object
        cosmology used throughout the project

    cosmo_choice: string
        string of the option of cosmology used throughout the project

    hmf_out: string
        path to the output file for the halo mass function

    Mmin: float, optional (default = 10)
        minimum halo mass to evaluate

    Mmax: float, optional (default = 15)
        maximum halo mass to evaluate

    dlog10m: float, optional (default = 1e-2)


    hmf_model:
    
    remove_hmf: boolean, optional (default = False)
        option to delete file if it exists

    delimiter: string, optional (default = ',')
        delimiter used for creating/reading cosmo file

    ext: string, optional (default = 'txt')
        extension of output file

    Returns
    ----------
    hmf_pd: pandas DataFrame
        DataFrame of `log10 masses` and `cumulative number densities` for 
        halos of mass > M.
    """
    ## HMF Output file
    hmf_outfile = '{0}/{1}_H0_{2}_HMF_{3}.{4}'.format(
        proj_dict['mf_dir'],
        cosmo_choice,
        cosmo_model.H0.value,
        hmf_model,
        ext)
    if remove_hmf and os.path.exists(hmf_outfile):
        # Removing file
        os.remove(hmf_outfile)
    ## Check if file exists
    if not os.path.exists(hmf_outfile):
        ## Halo mass function - Fitting function
        if hmf_model == 'warren':
            hmf_choice_fit = hmf.fitting_functions.Warren
        elif hmf_model == 'tinker08':
            hmf_choice_fit = hmf.fitting_functions.Tinker08
        else:
            msg = '{0} hmf_model `{1}` not supported! Exiting'.format(
                Prog_msg, hmf_model)
            raise ValueError(msg)
        # Calculating HMF
        mass_func = hmf.MassFunction(Mmin=Mmin, Mmax=Mmax, dlog10m=dlog10m,
            cosmo_model=cosmo_model, hmf_model=hmf_choice_fit)
        ## Log10(Mass) and cumulative number density of haloes
        # HMF Pandas DataFrame
        hmf_pd = pd.DataFrame({ 'logM':num.log10(mass_func.m), 
                                'ngtm':mass_func.ngtm})
        # Saving to output file
        hmf_pd.to_csv(hmf_outfile, sep=delimiter, index=False,
            columns=['logM','ngtm'])
    else:
        ## Reading output file
        hmf_pd = pd.read_csv(hmf_outfile)

    return hmf_pd

## --------- CLF Models ------------##
def clf_models_analysis_main(proj_dict, param_dict, sigma_c_init=0.0,
    sigma_c_final=2, sigma_c_int=0.1):
    """
    Computes the various CLF models that have different values of ``sigma_c``
    for the CLF (conditional luminosity function)

    Parameters
    -------------
    proj_dict : `dict`
        Dictionary with the paths of the project.

    param_dict : `dict`
        Dictionary containing the variables for the project.

    sigma_c_init : `float`, optional
        Initial value for ``sigma_c`` to test. This variable is set to
        ``0.1`` by default.

    sigma_c_final : `float`, optional
        Final value for ``sigma_c`` to test. This variable is set to
        ``0.3`` by default.

    sigma_c_int : `float`, optional
        Value for the interval for ``sigma_c``. This variable is set to
        ``0.1`` by default.

    Returns
    ---------
    clf_models_dict : `dict`
        Dictionary with the final results for each of the various CLF
        models.
    """
    ## Array of ``sigma_c`` values
    sigma_c_arr = num.arange(   sigma_c_init,
                                sigma_c_final + 0.5*sigma_c_int,
                                sigma_c_int)
    sigma_c_arr = num.round(sigma_c_arr, 3)
    # Insert `fiducial` value
    sigma_c_arr = num.insert(sigma_c_arr, len(sigma_c_arr),
        param_dict['sigma_c_fid'])
    # Downloading Halo catalogue
    try:
        halocat = CachedHaloCatalog(simname='bolplanck', halo_finder='rockstar',
            redshift=0.0)
    except:
        dman = DownloadManager()
        dman.download_processed_halo_table('bolplanck', 'rockstar', 0.0,
            overwrite=True, ignore_nearby_redshifts=True)
        # Reading in `halocat`
        halocat = CachedHaloCatalog(simname='bolplanck', halo_finder='rockstar',
            redshift=0.0)
    ## Adding new column to main `halocat`
    try:
        halotable = halocat.halo_table
        halotable['halo_m180b'] = halotable['halo_m200b']
        halocat.halo_table = halotable
    except:
        pass
    # Saving halo catalogue to `param_dict`
    param_dict['halocat'    ] = halocat
    param_dict['Lbox'       ] = halocat.Lbox
    param_dict['halocat_vol'] = num.product(halocat.Lbox)
    # Initialize dictionary
    clf_models_dict = {}
    # Looping over CLF models
    for ii, sigma_c_ii in enumerate(tqdm(sigma_c_arr)):
        # Running CLF analysis
        clf_models_dict[sigma_c_ii] = clf_models_analysis_calc(sigma_c_ii,
            halocat, param_dict)

    return clf_models_dict

def clf_models_analysis_calc(sigma_c_ii, halocat, param_dict):
    """
    Computes the CLF models and computes relevant statistics for such model

    Parameters
    -----------
    sigma_c_ii : `float`
        Value of the scatter in ``log(L)`` for central galaxies in the
        ``CLF`` formalism.

    halocat : `halotools.sim_manager.cached_halo_catalog.CachedHaloCatalog`
        Halo catalogue that contains information about the dark matter
        halos in the simulations. This corresponds to the halo catalogue
        that was previously downloaded.

    param_dict : `dict`
        Dictionary containing the variables for the project.

    Returns
    ----------
    clf_ii_model_dict : `dict`
        Dictionary with the set of measurements and statistics for the given
        CLF model.
    """
    # Array of halo masses
    mhalo_arr = num.logspace(9, 15, 1000)
    # Initializing `clf_ii_model_dict` dictionary
    clf_ii_model_dict = {}
    # Initializing model
    clf_ii_model = PrebuiltHodModelFactory('cacciato09',
                        threshold=param_dict['log_lum_cut'])
    # Modifying `sigma_c` value
    clf_ii_model.param_dict['sigma'] = sigma_c_ii
    # Modifying model with LasDamas values
    clf_ii_model.param_dict['log_L_0'] = 9.952
    clf_ii_model.param_dict['log_M_1'] = 11.08
    clf_ii_model.param_dict['gamma_1'] = 3.510
    clf_ii_model.param_dict['gamma_2'] = 0.245
    clf_ii_model.param_dict['a_1']     = 0.5858
    clf_ii_model.param_dict['a_2']     = 0.8795
    clf_ii_model.param_dict['log_M_2'] = 14.30
    clf_ii_model.param_dict['b_0']     = -0.6954
    clf_ii_model.param_dict['b_1']     = 0.8672
    clf_ii_model.param_dict['b_2']     = -0.05834
    clf_ii_model.param_dict['delta_1'] = 0.0
    clf_ii_model.param_dict['delta_2'] = 0.0
    # Populating model
    clf_ii_model.populate_mock(halocat)
    # Luminosity
    median_lum = clf_ii_model.median_prim_galprop_centrals(prim_haloprop = mhalo_arr)
    # Mean number of centrals and satellites
    mean_cens = clf_ii_model.mean_occupation_centrals(prim_haloprop=mhalo_arr)
    mean_sats = clf_ii_model.mean_occupation_satellites(prim_haloprop=mhalo_arr)
    # CLF for centrals and satellites
    clf_cens  = clf_ii_model.clf_centrals(mhalo_arr)
    clf_sats  = clf_ii_model.clf_satellites(mhalo_arr)
    # Abundance matching for central galaxies
    galaxy_table_ii = clf_ii_model.mock.galaxy_table
    # Halo mass and HAM mass
    gal_logL_merged_ii = galaxy_table_HAM(galaxy_table_ii, param_dict)
    ##
    ## Saving values to dictionary
    clf_ii_model_dict['mhalo_arr'] = mhalo_arr
    clf_ii_model_dict['mean_cens'] = mean_cens
    clf_ii_model_dict['mean_sats'] = mean_sats
    clf_ii_model_dict['clf_cens' ] = clf_cens
    clf_ii_model_dict['clf_sats' ] = clf_sats
    clf_ii_model_dict['gal_pd'   ] = gal_logL_merged_ii

    return clf_ii_model_dict

## Computes HAM on central galaxies
def galaxy_table_HAM(galaxy_table_ii, param_dict):
    """
    Computes HAM on central galaxies and compares to the actual halo mass
    of the central galaxy.

    Parameters
    ------------
    galaxy_table_ii : `astropy.table.table.Table`
        Table containing both `galaxy` and `halo` information for each
        of the galaxies in the sample.

    param_dict : `dict`
        Dictionary containing the variables for the project.

    Returns
    ----------
    gal_logL_merged : `pandas.DataFrame`
        DataFrame containing original information + HAM masses
    """
    # HMF DataFrame
    hmf_pd   = param_dict['hmf_pd']
    hmf_dict = param_dict['hmf_dict']
    # Interpolation of HMF
    hmf_interp = interp1d(hmf_dict['dens'], hmf_dict['var'])
    # 
    # Columns to keep
    cols_keep = ['luminosity', 'gal_type', 'halo_mvir', 'log_Mvir']
    # Converting to `pandas.DataFrame`
    gal_tab_ii_pd = galaxy_table_ii.to_pandas()
    # Converting `halo_mvir` to `log`
    gal_tab_ii_pd.loc[:, 'log_Mvir'] = gal_tab_ii_pd['halo_mvir'].apply(num.log10)
    # Extracting central galaxies with their halo mass and luminosity
    gal_mod_ii = gal_tab_ii_pd.loc[(gal_tab_ii_pd['gal_type'] == 'centrals') &
                    (gal_tab_ii_pd['luminosity'] >= param_dict['log_lum_cut']),
                    cols_keep].reset_index(drop=True)
    # Computing log(L)
    gal_mod_ii.loc[:, 'logL'] = gal_mod_ii['luminosity'].apply(num.log10)
    # Sorted DataFrame
    gal_mod_sorted_ii = gal_mod_ii.sort_values('logL', ascending=True)
    # Unique set of `log(L)` values
    gal_mod_sorted_logL_arr = gal_mod_sorted_ii['logL'].values
    logL_unq_sort_arr       = num.sort(num.unique(gal_mod_sorted_logL_arr))
    # Calculating ranking
    logL_unq_sort_rank = num.arange(0, len(logL_unq_sort_arr))
    # Creating dictionary with the ranking and `log(L)` values
    logL_rank_dict = dict(zip(logL_unq_sort_arr, logL_unq_sort_rank))
    # Assigning ranks to each galaxy
    gal_mod_sorted_rank = [logL_rank_dict[xx] for xx in gal_mod_sorted_logL_arr]
    gal_mod_sorted_rank = num.asarray(gal_mod_sorted_rank)
    # Assigning it to DataFrame
    gal_mod_sorted_ii.loc[:, 'rank'] = gal_mod_sorted_rank
    ##
    ## --- Abundance Matching --- ##
    ##
    # Figuring out how many are larger than some `log(L)` value
    counts, bin_edges = num.histogram(gal_mod_sorted_logL_arr,
                            bins=logL_unq_sort_arr)
    # Fixing last bin
    lum_last_count = len(num.where(gal_mod_sorted_logL_arr == logL_unq_sort_arr[-1])[0])
    counts[-1]    -= lum_last_count
    counts         = num.insert(counts, len(counts), 0)
    counts        += 1
    # Cumulative histogram
    cumu_counts    = num.cumsum(counts[::-1])[::-1]
    ndens_cumu     = cumu_counts / param_dict['halocat_vol']
    # Assigning densities to each unique luminosity
    Mh_am          = hmf_interp(ndens_cumu)
    # Creating DataFrame and joining with original galaxy DataFrame
    ndens_cumu_unq_pd = pd.DataFrame({  'Mh_am': Mh_am,
                                        'ndens_cumu': ndens_cumu,
                                        'logL_unq_sort_arr': logL_unq_sort_arr})
    ndens_cumu_unq_pd.sort_values('logL_unq_sort_arr', ascending=True,
        inplace=True)
    # Creating ranking for each unique luminosity
    ndens_cumu_unq_pd.loc[:, 'rank'] = num.arange(len(ndens_cumu_unq_pd))
    # Joining both DataFrames
    gal_logL_merged = pd.merge( gal_mod_sorted_ii,
                                ndens_cumu_unq_pd,
                                left_on='rank',
                                right_on='rank',
                                how='left')
    # Calculating fractional difference
    mass_frac_diff = gal_logL_merged['Mh_am'] - gal_logL_merged['log_Mvir']
    mass_frac_diff /= gal_logL_merged['log_Mvir']
    mass_frac_diff *= 100.
    # Assigning it to main DataFrame
    gal_logL_merged.loc[:, 'mass_frac_diff'] = mass_frac_diff.values
    # Dropping certain columns
    drop_cols = ['logL_unq_sort_arr', 'ndens_cumu', 'rank', 'halo_mvir',
                    'gal_type', 'luminosity']
    gal_logL_merged.drop(drop_cols, axis=1, inplace=True)


    return gal_logL_merged

## Plotting HOD's
def hod_models_plotting(clf_models_dict, proj_dict, fig_fmt='pdf',
    figsize=(15,6), fig_number=1):
    """
    Plots the HOD for `centrals` and `satellite` galaxies.

    Parameters
    -----------
    clf_models_dict : `dict`
        Dictionary with the final results for each of the various CLF
        models.

    proj_dict : `dict`
        Dictionary with the paths of the project.

    fig_fmt : `str`, optional (default = 'pdf')
        extension used to save the figure

    figsize : `tuple`, optional
        Size of the output figure. This variable is set to `(10,6)` by
        default.

    fig_number : `int`, optional
        Number of figure in the workflow. This variable is set to `1`
        by default.
    """
    file_msg = cfutils.Program_Msg('./')
    # Matplotlib option
    matplotlib.rcParams['axes.linewidth'] = 2.5
    matplotlib.rcParams['axes.edgecolor'] = 'black'
    # Figure figure
    fname = os.path.join(   proj_dict['fig_dir'],
                            'Figure_{0}_clf_models.{1}'.format(fig_number,
                                fig_fmt))
    # Number of models
    clf_models = num.sort(list(clf_models_dict.keys()))
    n_models   = len(clf_models)
    # Labels
    xlabel = r'\boldmath$\log M_{\textrm{vir}}\left[h^{-1} M_{\odot}\right]$'
    ylabel = r'\boldmath$\langle N \rangle$'
    # Fontsize
    label_size = 14
    ##
    ## Figure details
    plt.clf()
    plt.close()
    fig, axes = plt.subplots(ncols=2, nrows=1, figsize=figsize, sharex=True,
        sharey=True, facecolor='white')
    try:
        axes = axes.flatten()
    except:
        axes = [axes]
    # Limits
    mhalo_lim = (10.5, 14.9)
    ylim      = (0.1, 10)
    # Colors
    cm       = plt.cm.get_cmap('viridis')
    cm_arr   = [cm(kk/float(n_models)) for kk in range(n_models)]
    propssfr = dict(boxstyle='round', facecolor='white', alpha=0.7)
    # Looping over CLF models
    for ii, clf_ii in enumerate(tqdm(clf_models)):
        if (ii == 0):
            for ax_ii in axes:
                ax_ii.set_facecolor('white')
        cen_ax = axes[0]
        sat_ax = axes[1]
        ## CLF Model
        clf_model_ii = clf_models_dict[clf_ii]
        # Halo mass
        mhalo_arr = clf_model_ii['mhalo_arr']
        log_mhalo_arr = num.log10(mhalo_arr)
        ## ------ Centrals ------ ##
        # Mean occupation numbers
        mean_cens_ii = clf_model_ii['mean_cens']
        # Plotting occupation numbers
        cen_ax.plot(log_mhalo_arr, mean_cens_ii, color=cm_arr[ii],
            label=r'\boldmath$\sigma_{\textrm{c}}$ :' + '{0}'.format(clf_ii),
            linestyle='-', marker='o')
        ## ------ Satellites ------ ##
        # Mean occupation numbers
        mean_sats_ii = clf_model_ii['mean_sats']
        # Plotting occupation numbers
        sat_ax.plot(log_mhalo_arr, mean_sats_ii, color=cm_arr[ii],
            label=r'\boldmath$\sigma_{\textrm{c}}$ :' + '{0}'.format(clf_ii),
            linestyle='-')
        # Axes limits
        cen_ax.set_xlim(mhalo_lim)
        cen_ax.set_ylim(ylim)
        # Axes labels
        if (ii == 0):
            cen_ax.set_xlabel(xlabel, fontsize=label_size)
            sat_ax.set_xlabel(xlabel, fontsize=label_size)
            cen_ax.set_ylabel(ylabel, fontsize=label_size)
            plt.setp(sat_ax.get_yticklabels(), visible=False)
        # Logspace
        cen_ax.set_yscale('log')
        sat_ax.set_yscale('log')
    # Galaxy type text
    cen_ax.text(0.60, 0.10, 'Centrals', transform=cen_ax.transAxes,
        verticalalignment='top', color='black',
        bbox=propssfr, weight='bold', fontsize=10)
    sat_ax.text(0.60, 0.10, 'Satellites', transform=sat_ax.transAxes,
        verticalalignment='top', color='black',
        bbox=propssfr, weight='bold', fontsize=10)
    # Legend
    axes[0].legend(loc='upper left', ncol=3, numpoints=1, frameon=False,
        prop={'size': 14})
    axes[1].legend(loc='upper left', ncol=3, numpoints=1, frameon=False,
        prop={'size': 14})
    # Horizontal line
    cen_ax.axhline(y=0.5, color='black', linestyle='--', zorder=10)
    sat_ax.axhline(y=0.5, color='black', linestyle='--', zorder=10)
    # Removing any spacings
    plt.subplots_adjust(wspace=0., hspace=0.)
    ##
    ## Saving figure
    if (fig_fmt == 'pdf'):
        plt.savefig(fname, bbox_inches='tight', rasterize=True)
    else:
        plt.savefig(fname, bbox_inches='tight', dpi=400)
    ##
    ##
    # print('{0} Figure saved as: {1}'.format(file_msg, fname))
    print('{0} CLF Models Figure saved as: {1}'.format(file_msg, fname))
    plt.clf()
    plt.close()

## Plotting HOD's
def mass_frac_diff_models_plotting(clf_models_dict, proj_dict, param_dict,
    arr_len=10, bin_statval='average', fig_fmt='pdf', figsize=(8,8),
    fig_number=2, plot_shades=False, plot_opt='m_ham'):
    """
    Plots the HOD for `centrals` and `satellite` galaxies.

    Parameters
    -----------
    clf_models_dict : `dict`
        Dictionary with the final results for each of the various CLF
        models.

    proj_dict : `dict`
        Dictionary with the paths of the project.

    param_dict : `dict`
        Dictionary containing the variables for the project.

    arr_len : `int`, optional
        Minimum number of elements in bins. This variable is set to `0`
        by default.

    bin_statval : `str`, optional
        Option for where to plot the bin values. This variable is set
        to `average` by default.

        Options:
        - 'average': Returns the x-points at the average x-value of the bin
        - 'left'   : Returns the x-points at the left-edge of the x-axis bin
        - 'right'  : Returns the x-points at the right-edge of the x-axis bin

    fig_fmt : `str`, optional (default = 'pdf')
        extension used to save the figure

    figsize : `tuple`, optional
        Size of the output figure. This variable is set to `(10,6)` by
        default.

    fig_number : `int`, optional
        Number of figure in the workflow. This variable is set to `1`
        by default.

    plot_shades : `bool`
        If `True`, it plots the error/shades of :math:`1\\sigma` deviations
        from the mean for each of the CLF models. This variable is set to
        `False` by default.

    plot_opt : {``m_ham``, ``mvir``}, `str`, optional
        Option for which `mass` to plot on the ``x-axis``. This variable is
        set to ``m_ham`` by default.

        Options:
            - ``m_ham`` : Plots the `estimated` mass from ``HAM``.
            - ``mvir`` : Plots the `virial` halo mass of the galaxy's halo.

    """
    file_msg = cfutils.Program_Msg('./')
    # Matplotlib option
    matplotlib.rcParams['axes.linewidth'] = 2.5
    matplotlib.rcParams['axes.edgecolor'] = 'black'
    # Figure figure
    fname = os.path.join(   proj_dict['fig_dir'],
                            'Figure_{0}_mass_frac_diff_models_{1}.{2}'.format(
                                fig_number, plot_opt, fig_fmt))
    # Number of models
    clf_models = num.sort(list(clf_models_dict.keys()))
    n_models   = len(clf_models)
    # Labels
    if (plot_opt == 'm_ham'):
        xlabel = r'\boldmath$\log M_{\mathrm{HAM}}\left[ h^{-1} M_{\odot}\right]$'
    elif (plot_opt == 'mvir'):
        xlabel = r'\boldmath$\log M_{\mathrm{vir}}\left[ h^{-1} M_{\odot}\right]$'
    ylabel = r'Frac. Difference \boldmath$[\%]$'
    # Fontsize
    label_size   = 14
    alpha        = 0.6
    zorder_mass  = 10
    zorder_shade = zorder_mass - 1
    zorder_ml    = zorder_mass + 1
    ##
    ## Figure details
    plt.clf()
    plt.close()
    fig, axes = plt.subplots(ncols=1, nrows=1, figsize=figsize, sharex=True,
        sharey=True, facecolor='white')
    try:
        axes = axes.flatten()
    except:
        axes = [axes]
    # Limits
    mhalo_lim = (10.5, 14.9)
    ylim      = (-10, 10)
    # Colors
    cm       = plt.cm.get_cmap('viridis')
    cm_arr   = [cm(kk/float(n_models)) for kk in range(n_models)]
    propssfr = dict(boxstyle='round', facecolor='white', alpha=0.7)
    ## Modifying axes
    cen_ax = axes[0]
    cen_ax.set_facecolor('white')
    # Looping over CLF models
    for ii, clf_ii in enumerate(tqdm(clf_models)):
        ## CLF Model
        clf_model_ii = clf_models_dict[clf_ii]
        # Galaxy DataFame
        gal_pd_ii = clf_model_ii['gal_pd']
        # Extracting arrays
        frac_diff_ii = gal_pd_ii['mass_frac_diff'].values
        mh_am_ii     = gal_pd_ii['Mh_am'].values
        mvir_ii      = gal_pd_ii['log_Mvir'].values
        # Choosing which mass to plot
        if (plot_opt == 'm_ham'):
            mass_selected = mh_am_ii
        elif (plot_opt == 'mvir'):
            mass_selected = mvir_ii
        ##
        ## Computing statistics
        (   x_stat_arr,
            y_stat_arr,
            y_std_arr,
            y_std_err) = cstats.Stats_one_arr(  mass_selected,
                                                frac_diff_ii,
                                                base=0.4,
                                                arr_len=arr_len,
                                                bin_statval=bin_statval)
        y1 = y_stat_arr - y_std_arr
        y2 = y_stat_arr + y_std_arr
        ##
        ## Fractional difference
        if (clf_ii == param_dict['sigma_c_fid']):
            linestyle = '-.'
            label = 'Fiducial'
        else:
            linestyle = '-'
            label = r'\boldmath$\sigma_{\textrm{c}}$ :' + '{0}'.format(clf_ii)
        # Plotting relation
        cen_ax.plot(x_stat_arr, y_stat_arr, color=cm_arr[ii],
            label=label, linestyle=linestyle, marker='o')
        # Shadings
        if plot_shades:
            cen_ax.fill_between(x_stat_arr, y1, y2, color=cm_arr[ii],
                alpha=alpha, zorder=zorder_shade)
    # Axes limits
    cen_ax.set_xlim(mhalo_lim)
    cen_ax.set_ylim(ylim)
    # Axes labels
    cen_ax.set_xlabel(xlabel, fontsize=label_size)
    cen_ax.set_ylabel(ylabel, fontsize=label_size)
    # Galaxy type text
    cen_ax.text(0.06, 0.80, 'Centrals', transform=cen_ax.transAxes,
        verticalalignment='top', color='black',
        bbox=propssfr, weight='bold', fontsize=20)
    # Legend
    if (plot_opt == 'm_ham'):
        leg_pos = 'lower right'
    elif (plot_opt == 'mvir'):
        leg_pos = 'upper right'
    cen_ax.legend(loc=leg_pos, ncol=3, numpoints=1, frameon=False,
        prop={'size': 14})
    # Horizontal line
    cen_ax.axhline(y=0.5, color='black', linestyle='--', zorder=10)
    # Removing any spacings
    plt.subplots_adjust(wspace=0., hspace=0.)
    ##
    ## Saving figure
    if (fig_fmt == 'pdf'):
        plt.savefig(fname, bbox_inches='tight', rasterize=True)
    else:
        plt.savefig(fname, bbox_inches='tight', dpi=400)
    ##
    ##
    # print('{0} Figure saved as: {1}'.format(file_msg, fname))
    print('{0} CLF Models Figure saved as: {1}'.format(file_msg, fname))
    plt.clf()
    plt.close()

## --------- Main Function ------------##

def main(args):
    """
    Compares various CLF models
    """
    ## Starting time
    start_time = datetime.now()
    ## Reading all elements and converting to python dictionary
    param_dict = vars(args)
    ## Creating Folder Structure
    proj_dict  = cwpaths.cookiecutter_paths('./')
    proj_dict  = directory_skeleton(proj_dict)
    # Cosmological model
    (   cosmo_model,
        cosmo_hmf  ) = cosmo_create(cosmo_choice='Planck')
    # Assigning the cosmological model to `param_dict`
    param_dict['cosmo_model'] = cosmo_model
    param_dict['cosmo_hmf'  ] = cosmo_hmf
    # Halo mass function
    hmf_pd = hmf_calc(param_dict['cosmo_model'], param_dict['cosmo_choice'],
        proj_dict, Mmin=6., Mmax=16.01, dlog10m=1.e-3,
        remove_hmf=param_dict['remove_hmf'], 
        hmf_model=param_dict['hmf_model'])
    hmf_pd.rename(columns=dict(zip(hmf_pd.columns.values, ['var', 'dens'])),
        inplace=True)
    # To dictionary
    hfm_dict = hmf_pd.to_dict('list')
    for key in hfm_dict.keys():
        hfm_dict[key] = num.asarray(hfm_dict[key])
    # Assigning the HMF to `param_dict`
    param_dict['hmf_pd'] = hmf_pd
    param_dict['hmf_dict'] = hfm_dict
    ##
    ## Computing various CLF models
    clf_models_dict = clf_models_analysis_main(proj_dict, param_dict)
    # HOD Plotting
    hod_models_plotting(clf_models_dict, proj_dict)
    # Fractional difference between HAM and actual halo mass
    mass_frac_diff_models_plotting(clf_models_dict, proj_dict,
        param_dict, plot_opt='mvir')
    mass_frac_diff_models_plotting(clf_models_dict, proj_dict,
        param_dict, plot_opt='m_ham')

# Main function
if __name__=='__main__':
    ## Input arguments
    args = get_parser()
    # Main Function
    main(args)
