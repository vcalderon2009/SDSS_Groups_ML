#! /usr/bin/env python
# -*- coding: utf-8 -*-

# Victor Calderon
# Created      : 2018-10-02
# Last Modified: 2018-10-02
# Vanderbilt University
from __future__ import absolute_import, division, print_function 
__author__     = ['Victor Calderon']
__copyright__  = ["Copyright 2018 Victor Calderon, "]
__email__      = ['victor.calderon@vanderbilt.edu']
__maintainer__ = ['Victor Calderon']
"""
Script that grabs the output catalogue of the *real* SDSS data, and
produces sets of plots. The set of plots includes:
    - Comparison between HAM, DYN, and predicted group masses for galaxies.
    - Standard error as function of measured mass.
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

from src.ml_tools import ReadML

import numpy as num
import os
import sys
import pandas as pd
import matplotlib
matplotlib.use( 'Agg' )
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import matplotlib.gridspec as gridspec
from matplotlib.colors import LogNorm
from matplotlib.ticker import NullFormatter
# plt.rc('text', usetex=True)
import seaborn as sns
#sns.set()
from tqdm import tqdm

from datetime import datetime

import astropy.constants as ac
import astropy.units     as u

# Extra-modules
import argparse
from argparse import ArgumentParser
from argparse import HelpFormatter
from operator import attrgetter
from tqdm import tqdm

## Functions

## ------------------------- Initial Functions ------------------------------#

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
    description_msg = """
    Script that grabs the output catalogue of the *real* SDSS data, and
    produces sets of plots. The set of plots includes:
        - Comparison between HAM, DYN, and predicted group masses for galaxies.
        - Standard error as function of measured mass.
    """
    parser = ArgumentParser(description=description_msg,
                            formatter_class=SortingHelpFormatter,)
    ## 
    parser.add_argument('--version', action='version', version='%(prog)s 1.0')

    ## Number of HOD's to create. Dictates how many different types of
    ##      mock catalogues to create
    parser.add_argument('-hod_model_n',
                        dest='hod_n',
                        help="Number of distinct HOD model to use.",
                        type=int,
                        choices=range(0, 10),
                        metavar='[0-10]',
                        default=0)
    ## Type of dark matter halo to use in the simulation
    parser.add_argument('-halotype',
                        dest='halotype',
                        help='Type of the DM halo.',
                        type=str,
                        choices=['so', 'fof'],
                        default='so')
    ## CLF/CSMF method of assigning galaxy properties
    parser.add_argument('-clf_method',
                        dest='clf_method',
                        help="""
                        Method for assigning galaxy properties to mock
                        galaxies. Options:
                        (1) = Independent assignment of (g-r), sersic, logssfr
                        (2) = (g-r) decides active/passive designation and
                        draws values independently.
                        (3) (g-r) decides active/passive designation, and
                        assigns other galaxy properties for that given
                        galaxy.
                        """,
                        type=int,
                        choices=[1, 2, 3],
                        default=1)
    ## Random Seed for CLF
    parser.add_argument('-clf_seed',
                        dest='clf_seed',
                        help='Random seed to be used for CLF',
                        type=int,
                        metavar='[0-4294967295]',
                        default=1235)
    ## Difference between galaxy and mass velocity profiles (v_g-v_c)/(v_m-v_c)
    parser.add_argument('-dv',
                        dest='dv',
                        help="""
                        Difference between galaxy and mass velocity profiles
                        (v_g-v_c)/(v_m-v_c)
                        """,
                        type=_check_pos_val,
                        default=1.0)
    ## Luminosity sample to analyze
    parser.add_argument('-sample',
                        dest='sample',
                        help='SDSS Luminosity sample to analyze',
                        type=str,
                        choices=['19', '20', '21'],
                        default='19')
    ## SDSS Type
    parser.add_argument('-abopt',
                        dest='catl_type',
                        help='Type of Abund. Matching used in catalogue',
                        type=str,
                        choices=['mr', 'mstar'],
                        default='mr')
    ## Cosmology Choice
    parser.add_argument('-cosmo',
                        dest='cosmo_choice',
                        help='Choice of Cosmology',
                        type=str,
                        choices=['LasDamas', 'Planck'],
                        default='LasDamas')
    ## Minimum of galaxies in a group
    parser.add_argument('-nmin',
                        dest='nmin',
                        help='Minimum number of galaxies in a galaxy group',
                        type=int,
                        choices=range(2, 1000),
                        metavar='[1-1000]',
                        default=2)
    ## Factor by which to evaluate the distance to closest cluster.
    parser.add_argument('-mass_factor',
                        dest='mass_factor',
                        help="""
                        Factor by which to evaluate the distance to closest
                        cluster""",
                        type=int,
                        choices=range(2, 100),
                        metavar='[2-100]',
                        default=10)
    ## Total number of properties to predict. Default = 1
    parser.add_argument('-n_predict',
                        dest='n_predict',
                        help="""
                        Number of properties to predict. Default = 1""",
                        type=int,
                        choices=range(1, 4),
                        default=1)
    ## Option for Shuffling dataset when separating
    ## `training` and `testing` sets
    parser.add_argument('-shuffle_opt',
                        dest='shuffle_opt',
                        help="""
                        Option for whether or not to shuffle the data before
                        splitting.
                        """,
                        type=_str2bool,
                        default=True)
    ## Option for Shuffling dataset when separing `training` and `testing` sets
    parser.add_argument('-dropna_opt',
                        dest='dropna_opt',
                        help="""
                        Option for whether or not to drop NaNs from the dataset
                        """,
                        type=_str2bool,
                        default=True)
    ## Option for removing file
    parser.add_argument('-pre_opt',
                        dest='pre_opt',
                        help="""
                        Option for which preprocessing of the data to use.
                        """,
                        type=str,
                        choices=['min_max', 'standard', 'normalize', 'no'],
                        default='standard')
    ## Option for which kind of separation of training/testing to use for the
    ## datasets.
    parser.add_argument('-test_train_opt',
                        dest='test_train_opt',
                        help="""
                        Option for which kind of separation of training/testing
                        to use for the datasets.
                        """,
                        type=str,
                        choices=['sample_frac', 'boxes_n', 'box_sample_frac'],
                        default='sample_frac')
    ## Initial and final indices of the simulation boxes to use for the
    ## testing and training datasets.
    parser.add_argument('-box_idx',
                        dest='box_idx',
                        help="""
                        Initial and final indices of the simulation boxes to
                        use for the `training` datasets.
                        And the index of the boxes used for `testing`.
                        Example: 0_4_5 >>> This will use from 0th to 4th box
                        for training, and the 5th box for testing.""",
                        type=str,
                        default='0_4_5')
    ## Index of the simulation box to use for the `training` and `testing
    parser.add_argument('-box_test',
                        dest='box_test',
                        help="""
                        Index of the simulation box to use for the
                        `training` and `testing` datasets.
                        This index represents the simulation box, from which
                        both the `training` and `testing` datasets will be
                        produced. It used the `test_size` variable to
                        determine the fraction of the sample used for the
                        `testing` dataset. This variable is used only when
                        ``test_train_opt == 'box_sample_frac'``. Default : `0`.
                        Example : 0 >> It used the 0th simulation box
                        for training and testing.""",
                        type=int,
                        default=0)
    ## Fraction of the sample to be used.
    ## Only if `test_train_opt == 'sample_frac'`
    parser.add_argument('-sample_frac',
                        dest='sample_frac',
                        help='fraction of the total dataset to use',
                        type=float,
                        default=0.1)
    ## Testing size for ML
    parser.add_argument('-test_size',
                        dest='test_size',
                        help="""
                        Percentage size of the catalogue used for testing""",
                        type=_check_pos_val,
                        default=0.25)
    ## Option for using all features or just a few
    parser.add_argument('-n_feat_use',
                        dest='n_feat_use',
                        help="""
                        Option for which features to use for the ML training
                        dataset.
                        """,
                        choices=['all', 'sub'],
                        default='sub')
    ## Option for calculating densities or not
    parser.add_argument('-dens_calc',
                        dest='dens_calc',
                        help='Option for calculating densities.',
                        type=_str2bool,
                        default=False)
    ## Total number of K-folds, i.e. 'kf_splits'
    parser.add_argument('-kf_splits',
                        dest='kf_splits',
                        help="""
                        Total number of K-folds to perform. Must be larger
                        than 2""",
                        type=int,
                        default=3)
    ## Number of hidden layers to use
    parser.add_argument('-hidden_layers',
                        dest='hidden_layers',
                        help="""
                        Number of hidden layers to use for neural network
                        """,
                        type=int,
                        default=3)
    ## Number of units per hidden layer for the neural network.
    parser.add_argument('-unit_layer',
                        dest='unit_layer',
                        help="""
                        Number of units per hidden layer for the neural
                        network. Default = `100`.
                        """,
                        type=int,
                        default=100)
    ## Option for determining scoring
    parser.add_argument('-score_method',
                        dest='score_method',
                        help="""
                        Option for determining which scoring method to use.
                        """,
                        type=str,
                        choices=['perc', 'threshold', 'model_score'],
                        default='threshold')
    ## Threshold value used for when `score_method == 'threshold'`
    parser.add_argument('-threshold',
                        dest='threshold',
                        help="""Threshold value used for when
                        `score_method == 'threshold'`""",
                        type=float,
                        default=0.1)
    ## Percentage value used for when `score_method == 'perc'`
    parser.add_argument('-perc_val',
                        dest='perc_val',
                        help="""Percentage value used for when
                        `score_method == 'perc'`""",
                        type=float,
                        default=0.68)
    ## Type of subsample/binning for the estimated group masses
    parser.add_argument('-sample_method',
                        dest='sample_method',
                        help="""
                        Method for binning or sumsample the array of the
                        estimated group mass.
                        """,
                        type=str,
                        choices=['binning', 'subsample', 'weights', 'normal'],
                        default='binning')
    ## Type of binning to use
    parser.add_argument('-bin_val',
                        dest='bin_val',
                        help='Type of binning to use for the mass',
                        type=str,
                        choices=['fixed', 'nbins'],
                        default='fixed')
    ## Type of analysis to perform.
    parser.add_argument('-ml_analysis',
                        dest='ml_analysis',
                        help='Type of analysis to perform.',
                        type=str,
                        choices=['hod_dv_fixed'],
                        default='hod_dv_fixed')
    ## Type of resampling to use if necessary
    parser.add_argument('-resample_opt',
                        dest='resample_opt',
                        help='Type of resampling to use if necessary',
                        type=str,
                        choices=['over', 'under'],
                        default='under')
    ## Algorithm used for the final estimation of mass
    parser.add_argument('-chosen_ml_alg',
                        dest='chosen_ml_alg',
                        help='Algorithm used for the final estimation of mass',
                        type=str,
                        choices=['xgboost', 'rf', 'nn'],
                        default='xgboost')
    ## CPU Counts
    parser.add_argument('-cpu',
                        dest='cpu_frac',
                        help='Fraction of total number of CPUs to use',
                        type=float,
                        default=0.75)
    ## Option for removing file
    parser.add_argument('-remove',
                        dest='remove_files',
                        help="""
                        Delete files from previous analyses with same
                        parameters
                        """,
                        type=_str2bool,
                        default=False)
    ## Verbose
    parser.add_argument('-v', '--verbose',
                        dest='verbose',
                        help='Option to print out project parameters',
                        type=_str2bool,
                        default=False)
    ## `Perfect Catalogue` Option
    parser.add_argument('-perf',
                        dest='perf_opt',
                        help='Option for using a `Perfect` catalogue',
                        type=_str2bool,
                        default=False)
    ## Random Seed
    parser.add_argument('-seed',
                        dest='seed',
                        help='Random seed to be used for the analysis',
                        type=int,
                        metavar='[0-4294967295]',
                        default=1)

    ## Program message
    parser.add_argument('-progmsg',
                        dest='Prog_msg',
                        help='Program message to use throught the script',
                        type=str,
                        default=cfutils.Program_Msg(__file__))
    ## Parsing Objects
    args = parser.parse_args()

    return args

def param_vals_test(param_dict):
    """
    Checks if values are consistent with each other.

    Parameters
    -----------
    param_dict : `dict`
        Dictionary with `project` variables

    Raises
    -----------
    ValueError : Error
        This function raises a `ValueError` error if one or more of the 
        required criteria are not met
    """
    ##    file_msg = param_dict['Prog_msg']
    ##
    ## Testing if `wget` exists in the system
    if is_tool('wget'):
        pass
    else:
        msg = '{0} You need to have `wget` installed in your system to run '
        msg += 'this script. You can download the entire dataset at {1}.\n\t\t'
        msg += 'Exiting....'
        msg = msg.format(file_msg, param_dict['url_catl'])
        raise ValueError(msg)
    ##
    ## Checking that Esmeralda is not ran when doing 'SO' halos
    if (param_dict['halotype'] == 'so') and (param_dict['sample'] == 20):
        msg = '{0} The `halotype`==`so` and `sample`==`20` are no compatible '
        msg += 'input parameters.\n\t\t'
        msg += 'Exiting...'
        msg = msg.format(file_msg)
        raise ValueError(msg)
    ##
    ## Checking that `hod_model_n` is set to zero for FoF-Halos
    if (param_dict['halotype'] == 'fof') and (param_dict['hod_n'] != 0):
        msg = '{0} The `halotype`==`{1}` and `hod_n`==`{2}` are no compatible '
        msg += 'input parameters.\n\t\t'
        msg += 'Exiting...'
        msg = msg.format(   file_msg,
                            param_dict['halotype'],
                            param_dict['hod_n'])
        raise ValueError(msg)
    ##
    ## Checking input different types of `test_train_opt`
    #
    # `sample_frac`
    if (param_dict['test_train_opt'] == 'sample_frac'):
        # `sample_frac`
        if not ((param_dict['sample_frac'] > 0) and
                (param_dict['sample_frac'] <= 1.)):
            msg = '{0} `sample_frac` ({1}) must be between (0,1]'.format(
                file_msg, param_dict['sample_frac'])
            raise ValueError(msg)
        # `test_size`
        if not ((param_dict['test_size'] > 0) and
                (param_dict['test_size'] < 1)):
            msg = '{0} `test_size` ({1}) must be between (0,1)'.format(
                file_msg, param_dict['test_size'])
            raise ValueError(msg)
    #
    # boxes_n
    if (param_dict['test_train_opt'] == 'boxes_n'):
        box_n_arr = num.array(param_dict['box_idx'].split('_')).astype(int)
        box_n_diff = num.diff(box_n_arr)
        # Larger than zero
        if not (all(box_n_arr >= 0)):
            msg = '{0} All values in `box_idx` ({1}) must be larger than 0!'
            msg = msg.format(file_msg, box_n_arr)
            raise ValueError(msg)
        # Difference between elements
        if not (all(box_n_diff > 0)):
            msg = '{0} The value of `box_idx` ({1}) is not valid!'.format(
                file_msg, param_dict['box_idx'])
            raise ValueError(msg)
    #
    # `box_test`
    if (param_dict['test_train_opt'] == 'box_sample_frac'):
        # Value of `box_test`
        if not (param_dict['box_test'] >= 0):
            msg = '{0} `box_test` ({1}) must be larger or equal to `0`.'
            msg = msg.format(file_msg, param_dict['box_test'])
            raise ValueError(msg)
        # Testing `test_size`
        # `test_size`
        if not ((param_dict['test_size'] > 0) and
                (param_dict['test_size'] < 1)):
            msg = '{0} `test_size` ({1}) must be between (0,1)'.format(
                file_msg, param_dict['test_size'])
            raise ValueError(msg)
    ##
    ## Checking that `kf_splits` is larger than `2`
    if (param_dict['kf_splits'] < 2):
        msg  = '{0} The value for `kf_splits` ({1}) must be LARGER than `2`'
        msg += 'Exiting...'
        msg  = msg.format(param_dict['Prog_msg'], param_dict['kf_splits'])
        raise ValueError(msg)
    ##
    ## Checking that `n_predict` is not smaller than `1`.
    if (param_dict['n_predict'] < 1):
        msg  = '{0} The value for `n_predict` ({1}) must be LARGER than `1`'
        msg += 'Exiting...'
        msg  = msg.format(param_dict['Prog_msg'], param_dict['n_predict'])
        raise ValueError(msg)

def is_tool(name):
    """Check whether `name` is on PATH and marked as executable."""

    # from whichcraft import which
    from shutil import which

    return which(name) is not None

def add_to_dict(param_dict):
    """
    Aggregates extra variables to dictionary

    Parameters
    ----------
    param_dict : `dict`
        Dictionary with input parameters and values

    Returns
    ----------
    param_dict : `dict`
        Dictionary with old and new values added
    """
    # Sample string
    sample_s = param_dict['ml_args'].sample_s
    ### Sample - Mr
    sample_Mr = param_dict['ml_args'].sample_Mr
    ## Sample volume
    # Units (Mpc/h)**3
    volume_sample = {   '18': 37820 / 0.01396,
                        '19': 6046016.60311  ,
                        '20': 2.40481e7      ,
                        '21': 8.79151e7      }
    vol_mr        = volume_sample[sample_s]
    ##
    ## Choice of Centrals and Satellites
    cens = int(1)
    sats = int(0)
    ## Other constants
    # Speed of light - In km/s
    speed_c = ac.c.to(u.km/u.s).value
    #
    # Catalogue prefix
    catl_str_fig = param_dict['ml_args'].catl_model_pred_prefix_str()
    #
    # Plotting constants
    plot_dict = {   'size_label':23,
                    'size_title':25,
                    'size_legend': 14
                }
    # Column names
    mass_cols = ['GG_M_group', 'GG_mdyn_rproj']
    mass_pred = 'M_h_pred'
    # Mass estimates - names dict
    mass_names_dict = { 'GG_M_group': 'Halo Abundance Matching',
                        'GG_mdyn_rproj': 'Dynamical Mass'}
    # Saving to dictionary
    param_dict['sample_s'       ] = sample_s
    param_dict['sample_Mr'      ] = sample_Mr
    param_dict['volume_sample'  ] = volume_sample
    param_dict['vol_mr'         ] = vol_mr
    param_dict['cens'           ] = cens
    param_dict['sats'           ] = sats
    param_dict['speed_c'        ] = speed_c
    param_dict['catl_str_fig'   ] = catl_str_fig
    param_dict['plot_dict'      ] = plot_dict
    param_dict['mass_cols'      ] = mass_cols
    param_dict['mass_pred'      ] = mass_pred
    param_dict['mass_names_dict'] = mass_names_dict

    return param_dict

def directory_skeleton(param_dict, proj_dict):
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
    # Main output figure directory
    figure_dir = param_dict['ml_args'].catl_model_pred_plot_dir(
                    check_exist=False, create_dir=True)
    # Paper Figure directory
    paper_fig_dir = os.path.join(   proj_dict['plot_dir'],
                                    'Paper_Figures')
    cfutils.Path_Folder(paper_fig_dir)
    # Saving to `proj_dict`
    proj_dict['figure_dir'   ] = figure_dir
    proj_dict['paper_fig_dir'] = paper_fig_dir

    return proj_dict

## ------------------------- Plotting Functions ------------------------------#

def mass_pred_comparison_plot(catl_pd, param_dict, proj_dict, arr_len=10,
    bin_statval='left', fig_fmt='pdf', figsize=(10,6), fig_number=1,
    plot_sigma_ax=False):
    """
    Plot for comparing the ML-predicted masses to those from more, traditional
    group mass estimations.

    Parameters
    ------------
    catl_pd : `pd.DataFrame`
        DataFrame containing information on the various features for each
        galaxy, as well as the `predicted` columns.

    param_dict : `dict`
        Dictionary with `project` variables.

    proj_dict : `dict`
        Dictionary with the `project` paths and directories.

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
        Size of the output figure. This variable is set to `(12,15.5)` by
        default.

    fig_number : `int`, optional
        Number of figure in the workflow. This variable is set to `1`
        by default.

    plot_sigma_ax : `bool`, optional
        If True, a second panel is created for error in predicted mass
        as function of traditional masses. This variable is set to
        ``False`` by default.
    """
    file_msg     = param_dict['Prog_msg']
    # Matplotlib option
    matplotlib.rcParams['axes.linewidth'] = 2.5
    matplotlib.rcParams['axes.edgecolor'] = 'black'
    # Constants
    cmap_opt  = 'Blues'
    err_color = 'Orange'
    vmax      = int(1) ## For the 2D Histograms
    one_arr   = num.linspace(6, 16.05, 1000)
    one_col   = 'k'
    plot_dict = param_dict['plot_dict']
    bin_width = param_dict['ml_args'].mass_bin_width
    ## Figure name
    fname = os.path.join(   proj_dict['figure_dir'],
                            'Fig_{0}_{1}_masses_comparison.{2}'.format(
                                fig_number,
                                param_dict['catl_str_fig'],
                                fig_fmt))
    ##
    ## Paper Figure
    fname_paper = os.path.join( proj_dict['paper_fig_dir'],
                                'Figure_12.{0}'.format(fig_fmt))
    # Labels
    labels_dict = { 'M_h_pred': r'\boldmath$\log M_{predicted}\left[h^{-1} M_{\odot}\right]$',
                    'GG_M_group': r'\boldmath$\log M_{\mathrm{HAM}}\left[h^{-1} M_{\odot}\right]$',
                    'GG_mdyn_rproj': r'\boldmath$\log M_{\mathrm{dyn}}\left[h^{-1} M_{\odot}\right]$'}
    # Selecting only these columns
    mass_cols = ['GG_M_group', 'GG_mdyn_rproj']
    mass_pred = 'M_h_pred'
    # Removing values 
    ##
    ## Figure details
    plt.clf()
    plt.close()
    fig    = plt.figure(figsize=figsize)
    gs     = gridspec.GridSpec(1, 2, hspace=0.1, wspace=0.0)
    ax_arr = [[[],[]] for xx in range(2)]
    # Figure contants
    xlim       = (10.9, 15.2)
    ylim       = (10.9, 15.2)
    sigma_lims = (0.0, 0.6)
    # Histogram bins
    dbin      = 0.1
    hist_bins = num.arange(xlim[0], xlim[1] + dbin*0.5, dbin)
    # Ticks
    xaxis_major_ticker = 1
    xaxis_minor_ticker = 0.2
    yaxis_major_ticker = 1
    yaxis_minor_ticker = 0.2
    ax_xaxis_major_loc = ticker.MultipleLocator(xaxis_major_ticker)
    ax_xaxis_minor_loc = ticker.MultipleLocator(xaxis_minor_ticker)
    ax_yaxis_major_loc = ticker.MultipleLocator(yaxis_major_ticker)
    ax_yaxis_minor_loc = ticker.MultipleLocator(yaxis_minor_ticker)
    # Sigma axis ticks
    ax_yaxis_major_loc_sig = ticker.MultipleLocator(0.5)
    ax_yaxis_minor_loc_sig = ticker.MultipleLocator(0.1)
    # Plotting sigmas if needed
    if plot_sigma_ax:
        # Looping over subplots
        for ii, mass_ii in enumerate(mass_cols):
            ## Labels
            x_label = labels_dict[mass_ii]
            y_label = labels_dict['M_h_pred']
            # Setting up axes
            gs_ax = gridspec.GridSpecFromSubplotSpec(2, 1, gs[ii],
                height_ratios=[3, 1], hspace=0)
            ax1 = plt.Subplot(fig, gs_ax[0,:], facecolor='white')
            ax2 = plt.Subplot(fig, gs_ax[1,:], facecolor='white', sharex=ax1)
            # Adding plots
            fig.add_subplot(ax1)
            fig.add_subplot(ax2)
            plt.setp(ax1.get_xticklabels(), visible=False)
            if ii != 0:
                plt.setp(ax1.get_yticklabels(), visible=False)
                plt.setp(ax2.get_yticklabels(), visible=False)
            else:
                ## Y-labels
                ax1.set_ylabel(y_label, fontsize=plot_dict['size_label'])
                ax2.set_ylabel(r'$\sigma$', fontsize=plot_dict['size_label'])
            ax2.set_xlabel(x_label, fontsize=plot_dict['size_label'])
            ## Ticks
            # ax1
            ax1.xaxis.set_major_locator(ax_xaxis_major_loc)
            ax1.xaxis.set_minor_locator(ax_xaxis_minor_loc)
            ax1.yaxis.set_major_locator(ax_yaxis_major_loc)
            ax1.yaxis.set_minor_locator(ax_yaxis_minor_loc)
            # ax2
            ax2.yaxis.set_major_locator(ax_yaxis_major_loc_sig)
            ax2.yaxis.set_minor_locator(ax_yaxis_minor_loc_sig)
            # Axes limits
            ax1.set_xlim(xlim)
            ax1.set_ylim(ylim)
            ax2.set_ylim(sigma_lims)
            # Setting up x- and y-arrays
            if (mass_ii  == 'GG_mdyn_rproj'):
                catl_pd_mod = catl_pd.loc[catl_pd[mass_ii] != 0]
                x_arr       = catl_pd_mod[mass_ii].values
                y_arr       = catl_pd_mod[mass_pred].values
            else:
                x_arr = catl_pd[mass_ii].values
                y_arr = catl_pd[mass_pred].values
            # Computing statistic
            (   mean_mass_trad_ii,
                mean_mass_pred   ,
                mass_pred_std    ,
                mass_pred_std_err) = cstats.Stats_one_arr(
                                                    x_arr,
                                                    y_arr,
                                                    base=bin_width,
                                                    arr_len=arr_len,
                                                    bin_statval=bin_statval)
            # Plotting 2D-histogram
            if (ii == (len(mass_cols) - 1)):
                im = ax1.hist2d(x_arr, y_arr, bins=hist_bins, norm=LogNorm(),
                    normed=True, cmap=cmap_opt, vmax=vmax)
            else:
                ax1.hist2d(x_arr, y_arr, bins=hist_bins, norm=LogNorm(),
                    normed=True, cmap=cmap_opt, vmax=vmax)
            # One-One-line
            ax1.plot(one_arr, one_arr, color=one_col, linestyle='--')
            # Errorbar
            ax1.errorbar(mean_mass_trad_ii, mean_mass_pred, yerr=mass_pred_std,
                ecolor=err_color, fmt='--', color=err_color)
            # Sigma plottting
            ax2.plot(mean_mass_trad_ii, mass_pred_std, color=err_color,
                linestyle='--')
            ## Axes limits
            ax1.set_xlim(xlim)
            ax1.set_ylim(ylim)
            ax2.set_xlim(xlim)
            ax2.set_ylim(sigma_lims)
            # Adding text
            if (ii == 1):
                ax1.text(0.05, 0.9, 'SDSS',
                    transform=ax1.transAxes, fontsize=plot_dict['size_legend'])
    else:
        # Looping over subplots
        for ii, mass_ii in enumerate(mass_cols):
            ## Labels
            x_label = labels_dict[mass_ii]
            y_label = labels_dict['M_h_pred']
            # Setting up axes
            gs_ax = gridspec.GridSpecFromSubplotSpec(1, 1, gs[ii],
                height_ratios=[3, 1], hspace=0)
            ax1 = plt.Subplot(fig, gs_ax[0,:], facecolor='white')
            # Adding plots
            fig.add_subplot(ax1)
            plt.setp(ax1.get_xticklabels(), visible=False)
            if ii != 0:
                plt.setp(ax1.get_yticklabels(), visible=False)
            else:
                ## Y-labels
                ax1.set_ylabel(y_label, fontsize=plot_dict['size_label'])
            ## Ticks
            # ax1
            ax1.xaxis.set_major_locator(ax_xaxis_major_loc)
            ax1.xaxis.set_minor_locator(ax_xaxis_minor_loc)
            ax1.yaxis.set_major_locator(ax_yaxis_major_loc)
            ax1.yaxis.set_minor_locator(ax_yaxis_minor_loc)
            # Axes limits
            ax1.set_xlim(xlim)
            ax1.set_ylim(ylim)
            # Setting up x- and y-arrays
            if (mass_ii  == 'GG_mdyn_rproj'):
                catl_pd_mod = catl_pd.loc[catl_pd[mass_ii] != 0]
                x_arr       = catl_pd_mod[mass_ii].values
                y_arr       = catl_pd_mod[mass_pred].values
            else:
                x_arr = catl_pd[mass_ii].values
                y_arr = catl_pd[mass_pred].values
            # Computing statistic
            (   mean_mass_trad_ii,
                mean_mass_pred   ,
                mass_pred_std    ,
                mass_pred_std_err) = cstats.Stats_one_arr(
                                                    x_arr,
                                                    y_arr,
                                                    base=bin_width,
                                                    arr_len=arr_len,
                                                    bin_statval=bin_statval)
            # Plotting 2D-histogram
            if (ii == (len(mass_cols) - 1)):
                im = ax1.hist2d(x_arr, y_arr, bins=hist_bins, norm=LogNorm(),
                    normed=True, cmap=cmap_opt, vmax=vmax)
            else:
                ax1.hist2d(x_arr, y_arr, bins=hist_bins, norm=LogNorm(),
                    normed=True, cmap=cmap_opt, vmax=vmax)
            # One-One-line
            ax1.plot(one_arr, one_arr, color=one_col, linestyle='--')
            # Errorbar
            ax1.errorbar(mean_mass_trad_ii, mean_mass_pred, yerr=mass_pred_std,
                ecolor=err_color, fmt='--', color=err_color)
            ## Axes limits
            ax1.set_xlim(xlim)
            ax1.set_ylim(ylim)
            # Adding text
            if (ii == 1):
                ax1.text(0.05, 0.9, 'SDSS',
                    transform=ax1.transAxes, fontsize=plot_dict['size_legend'])
    #
    # Colorbar
    cax  = fig.add_axes([0.92, 0.12, 0.03, 0.75])
    cbar = fig.colorbar(im[3], cax=cax)
    cbar.set_label('frequency', rotation=270, labelpad=15)
    ##
    ## Saving figure
    if fig_fmt=='pdf':
        plt.savefig(fname, bbox_inches='tight')
        plt.savefig(fname_paper, bbox_inches='tight')
    else:
        plt.savefig(fname, bbox_inches='tight', dpi=400)
        plt.savefig(fname_paper, bbox_inches='tight', dpi=400)
    ##
    ##
    print('{0} Figure saved as: {1}'.format(file_msg, fname))
    print('{0} Paper Figure saved as: {1}'.format(file_msg, fname_paper))
    plt.clf()
    plt.close()





def main(args):
    """
    Script that grabs the output catalogue of the *real* SDSS data, and
    produces sets of plots. The set of plots includes:
        - Comparison between HAM, DYN, and predicted group masses for galaxies.
        - Standard error as function of measured mass.
    """
    start_time = datetime.now()
    ## Reading all elements and converting to python dictionary
    param_dict = vars(args)
    ## Checking for correct input
    param_vals_test(param_dict)
    #
    # Creating instance of `ReadML` with the input parameters
    param_dict['ml_args'] = ReadML(**param_dict)
    ## Program message
    Prog_msg = param_dict['Prog_msg']
    ## Adding extra variables
    param_dict = add_to_dict(param_dict)
    ##
    ## Creating Folder Structure
    proj_dict = param_dict['ml_args'].proj_dict
    proj_dict  = directory_skeleton(param_dict, proj_dict)
    ##
    ## Printing out project variables
    print('\n'+50*'='+'\n')
    for key, key_val in sorted(param_dict.items()):
        if key !='Prog_msg':
            print('{0} `{1}`: {2}'.format(Prog_msg, key, key_val))
    print('\n'+50*'='+'\n')
    # Loading dataset with galaxy information
    catl_pd = param_dict['ml_args'].catl_model_pred_file_extract(
                    return_pd=True)
    # Cleaning sample
    catl_pd = catl_pd.loc[catl_pd['M_h_pred'] != 0]
    ##
    ## -- Creating Plots -- ##
    #
    # Mass comparison
    mass_pred_comparison_plot(catl_pd, param_dict, proj_dict, arr_len=0)




# Main function
if __name__=='__main__':
    ## Input arguments
    args = get_parser()
    # Main Function
    main(args)
