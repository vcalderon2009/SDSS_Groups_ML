#! /usr/bin/env python
# -*- coding: utf-8 -*-

# Victor Calderon
# Created      : 2018-06-03
# Last Modified: 2018-11-08
# Vanderbilt University
from __future__ import absolute_import, division, print_function
__author__     = ['Victor Calderon']
__copyright__  = ["Copyright 2018 Victor Calderon, "]
__email__      = ['victor.calderon@vanderbilt.edu']
__maintainer__ = ['Victor Calderon']
"""
Script that uses a set of ML algorithms to try to predict galaxy and
group properties by using a set of calculated `features` from synthetic
galaxy/group catalogues.

In here, we probe the algorithms:
    - Random Forest
    - XGBoost
    - Neural Network

We also employ different ways of subsampling/binning the data, in order
to improve the accuracy of the results. These include:
    - Binning the group mass and training models for each bin
    - Subsample the masses
    - Apply a weighting scheme, i.e. give more weight to high-mass systems
      than low-mass systems.
"""
# Importing Modules
from cosmo_utils.utils import file_utils      as cfutils
from cosmo_utils.utils import work_paths      as cwpaths
from cosmo_utils.utils import stats_funcs     as cstats
from cosmo_utils.ml    import ml_utils        as cmlu

from src.ml_tools import ReadML

from datetime import datetime
import numpy as num
import os
import pandas as pd
import pickle
import matplotlib
matplotlib.use( 'Agg' )
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from mpl_toolkits.axes_grid1.axes_divider import make_axes_locatable
from mpl_toolkits.axes_grid1.colorbar import colorbar
plt.rc('text', usetex=True)
from astropy.visualization import astropy_mpl_style
plt.style.use(astropy_mpl_style)
import seaborn as sns

# Extra-modules
import argparse
from argparse import ArgumentParser
from argparse import HelpFormatter
from operator import attrgetter
from tqdm import tqdm
from multiprocessing import cpu_count
import astropy.constants as ac
import astropy.units     as u

# ML modules
import sklearn
import sklearn.ensemble         as skem
import sklearn.neural_network   as skneuro
import xgboost

## Functions

## --------- General functions ------------##

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
    val: int or float
        value to be evaluated by `val_min`

    val_min: float or int, optional (default = 0)
        minimum value that `val` can be

    Returns
    -------
    ival: float
        value if `val` is larger than `val_min`

    Raises
    -------
    ArgumentTypeError: Raised if `val` is NOT larger than `val_min`
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
    Script that uses a set of ML algorithms to try to predict galaxy and
    group properties by using a set of calculated `features` from synthetic
    galaxy/group catalogues.
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
    # Value for the scatter in log(L) for central galaxies in the CLF
    parser.add_argument('-sigma_clf_c',
                        dest='sigma_clf_c',
                        help="""
                        Value for the scatter in log(L) for central galaxies
                        in the CLF
                        """,
                        type=_check_pos_val,
                        default=0.1417)
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
                        default='boxes_n')
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
                        default='0_3_4')
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
                        default='nbins')
    ## Type of analysis to perform.
    parser.add_argument('-ml_analysis',
                        dest='ml_analysis',
                        help='Type of analysis to perform.',
                        type=str,
                        choices=['hod_dv_fixed'],
                        default='hod_dv_fixed')
    ## Which axes to plot
    parser.add_argument('-plot_opt',
                        dest='plot_opt',
                        help='Option for which variable to plot on x-axis',
                        type=str,
                        choices=['mgroup', 'mhalo'],
                        default='mgroup')
    ## Which axes to plot
    parser.add_argument('-rank_opt',
                        dest='rank_opt',
                        help='Option for which type of ranking to plot',
                        type=str,
                        choices=['perc', 'idx'],
                        default='idx')
    ## CPU Counts
    parser.add_argument('-cpu',
                        dest='cpu_frac',
                        help='Fraction of total number of CPUs to use',
                        type=float,
                        default=0.75)
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
    param_dict: python dictionary
        dictionary with `project` variables

    Raises
    -----------
    ValueError: Error
        This function raises a `ValueError` error if one or more of the
        required criteria are not met
    """
    file_msg = param_dict['Prog_msg']
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
    param_dict: python dictionary
        dictionary with input parameters and values

    Returns
    ----------
    param_dict: python dictionary
        dictionary with old and new values added
    """
    ### Sample - Int
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
    ## Number of CPU's to use
    cpu_number = int(cpu_count() * param_dict['cpu_frac'])
    ##
    ## Plotting constants
    plot_dict = {   'size_label':23,
                    'size_title':25,
                    'color_ham' :'red',
                    'color_dyn' :'blue'}
    ##
    ## Catalogue Prefix string
    catl_str_fig = param_dict['ml_args'].catl_alg_comp_fig_str()
    ##
    ## Saving to `param_dict`
    param_dict['sample_s'    ] = sample_s
    param_dict['sample_Mr'   ] = sample_Mr
    param_dict['vol_mr'      ] = vol_mr
    param_dict['cens'        ] = cens
    param_dict['sats'        ] = sats
    param_dict['speed_c'     ] = speed_c
    param_dict['cpu_number'  ] = cpu_number
    param_dict['plot_dict'   ] = plot_dict
    param_dict['catl_str_fig'] = catl_str_fig

    return param_dict

def directory_skeleton(param_dict, proj_dict):
    """
    Creates the directory skeleton for the current project

    Parameters
    ----------
    param_dict: python dictionary
        dictionary with `project` variables

    proj_dict: python dictionary
        dictionary with info of the project that uses the
        `Data Science` Cookiecutter template.

    Returns
    ---------
    proj_dict: python dictionary
        Dictionary with current and new paths to project directories
    """
    # Catalogue prefix
    catl_prefix_path = param_dict['ml_args'].catl_prefix_path()
    # Figure directory
    figure_dir = os.path.join(proj_dict['plot_dir'],
                                'catl_props_exploration',
                                catl_prefix_path)
    # Paper Figure directory
    paper_fig_dir = os.path.join(   proj_dict['plot_dir'],
                                    'Paper_Figures')
    # Creating folder
    cfutils.Path_Folder(figure_dir)
    cfutils.Path_Folder(paper_fig_dir)
    #
    # Adding to `proj_dict`
    proj_dict['figure_dir'   ] = figure_dir
    proj_dict['paper_fig_dir'] = paper_fig_dir

    return proj_dict

## --------- Preparing data ------------##

# Different types of regressors
def array_insert(arr1, arr2, axis=1):
    """
    Joins the arrays into a signle multi-dimensional array

    Parameters
    ----------
    arr1: array_like
        first array to merge

    arr2: array_like
        second array to merge

    Return
    ---------
    arr3: array_like
        merged array from `arr1` and `arr2`
    """
    arr3 = num.insert(arr1, len(arr1.T), arr2, axis=axis)

    return arr3

# Clean version of the catalogue
def catl_file_read_clean(param_dict, proj_dict, dropna_opt=True):
    """
    Reads in the catalogue and cleans it for plotting.
    
    Parameters
    ------------
    param_dict : `dict`
        Dictionary with `project` variables

    proj_dict : `dict`
        Dictionary with info of the project that uses the
        `Data Science` Cookiecutter template.

    dropna_opt : `bool`
        If `True`, all the NaN's will be dropped from the main DataFrame.
        This variable is set to `True` by default.

    Returns
    ---------
    catl_pd: `pandas.DataFrame`
        DataFrame containing galaxy and group information
    """
    # Catalogue file - Total
    catl_pd_tot = param_dict['ml_args'].extract_merged_catl_info(opt='combined')
    # Fraction of total catalogue
    catl_pd = catl_pd_tot.sample(frac=param_dict['ml_args'].sample_frac,
                random_state=param_dict['ml_args'].seed)
    catl_pd_tot = None
    # Reind
    ## Reindexing
    catl_pd.reset_index(drop=True, inplace=True)
    ## Dropping NaN's
    if dropna_opt:
        catl_pd.dropna(how='any', inplace=True)
    ##  Temporarily fixing `GG_mdyn_rproj`
    catl_pd.loc[:,'GG_mdyn_rproj'] /= 0.96
    ##
    ## Dropping certain columns
    catl_drop_arr = ['groupid', 'halo_rvir', 'galtype', 'halo_ngal', 'box_n']
    catl_pd       = catl_pd.drop(catl_drop_arr, axis=1)

    return catl_pd

# Correctly converting the axes between fractional difference and normal
def stats_frac_diff(pred_arr, true_arr, base=0.4, arr_len=10,
    bin_statval='left', return_frac_diff=True):
    """
    Correctly converts the fractional difference of a given sample of 
    `predicted` and `true` values

    Parameters
    -----------
    pred_arr : array-like or `numpy.ndarray`
        Array of the `predicted` values

    true_arr : array-like or `numpy.ndarray`
        Array of the `true` values

    base : `float`, optional
        Bin width in units of `pred_arr`. This variable is set to 
        `0.4` by default.

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

    Returns
    ---------
    main_dict : `dict`
        Dictionary containing the different statistics to use for `pred_arr`
        and `true_arr`
    """
    # Computing normal statistics
    (   x_stat,
        y_stat,
        y_std,
        y_std_err) = cstats.Stats_one_arr(  pred_arr,
                                            true_arr,
                                            base=base,
                                            arr_len=arr_len,
                                            bin_statval=bin_statval)
    # Limits for 1 standard deviation
    y1 = y_stat - y_std
    y2 = y_stat + y_std
    # Computing fractional difference `correctly`
    frac_diff = 100 * (pred_arr - true_arr) / true_arr
    #
    # Computing the limits of the relation `correctly`
    y_stat_fd = 100 * (x_stat - y_stat) / y_stat
    y1_fd     = -100 * (x_stat - y1) / y1
    y2_fd     = -100 * (x_stat - y2) / y2
    #
    # Saving to dictionary
    main_dict = {}
    main_dict['x_stat'   ] = x_stat
    main_dict['y_stat'   ] = y_stat
    main_dict['y_std'    ] = y_std
    main_dict['y_std_err'] = y_std_err
    main_dict['y1'       ] = y1
    main_dict['y2'       ] = y2
    main_dict['y_stat_fd'] = y_stat_fd
    main_dict['y1_fd'    ] = y1_fd
    main_dict['y2_fd'    ] = y2_fd

    return main_dict




## --------- Plotting Functions ------------##

# Fractional difference
def frac_diff_model(param_dict, proj_dict, plot_opt='mhalo',
    arr_len=10, bin_statval='left', fig_fmt='pdf', figsize=(10, 8),
    fig_number=1):
    """
    Plots the fractional difference between `predicted` and `true`
    halo masses.

    Parameters
    -----------
    param_dict : `dict`
        Dictionary with input parameters and values related to this project.

    proj_dict: python dictionary
        Dictionary with current and new paths to project directories

    plot_opt : {'mgroup', 'mhalo'} `str`, optional  
        Option to which property to plot on the x-axis. This variable is set
        to `mhalo` by default.

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
    """
    file_msg = param_dict['Prog_msg']
    ## Matplotlib option
    matplotlib.rcParams['axes.linewidth'] = 2.5
    matplotlib.rcParams['axes.edgecolor'] = 'black'
    ##
    # Constants
    cm           = plt.cm.get_cmap('viridis')
    plot_dict    = param_dict['plot_dict']
    ham_color    = 'red'
    alpha        = 0.6
    alpha_mass   = 0.2
    zorder_mass  = 10
    zorder_shade = zorder_mass - 1
    zorder_ml    = zorder_mass + 1
    bin_width    = param_dict['ml_args'].mass_bin_width
    ##
    ## Figure name
    fname = os.path.join(   proj_dict['figure_dir'],
                            'Fig_{0}_{1}_group_mass_comparison.{2}'.format(
                                fig_number,
                                param_dict['catl_str_fig'],
                                fig_fmt))
    ##
    ## Paper Figure
    fname_paper = os.path.join( proj_dict['paper_fig_dir'],
                                'Figure_01.{0}'.format(fig_fmt))
    ## Abundance matched mass
    # HAM
    (   ham_pred,
        ham_true,
        ham_frac_diff) = param_dict['ml_args'].extract_trad_masses_alt(
                                                mass_opt='ham',
                                                return_frac_diff=True)
    # Dynamical
    (   dyn_pred,
        dyn_true,
        dyn_frac_diff) = param_dict['ml_args'].extract_trad_masses_alt(
                                                mass_opt='dyn',
                                                return_frac_diff=True,
                                                nlim_threshold=True,
                                                nlim_min=4)
    # Only choosing non-zero values
    dyn_pred_mask = dyn_pred >= 11.0
    dyn_pred      = dyn_pred[dyn_pred_mask]
    dyn_true      = dyn_true[dyn_pred_mask]
    dyn_frac_diff = dyn_frac_diff[dyn_pred_mask]
    ##
    ## Choosing which mass to plot
    if (plot_opt == 'mgroup'):
        ham_x = ham_pred
        dyn_x = dyn_pred
    elif (plot_opt == 'mhalo'):
        ham_x = ham_true
        dyn_x = dyn_true
    ##
    ## Binning data
    # # HAM
    # ham_dict = stats_frac_diff( ham_pred,
    #                             ham_true,
    #                             base=bin_width,
    #                             arr_len=arr_len,
    #                             bin_statval=bin_statval)
    # x_stat_ham = ham_dict['x_stat']
    # y_stat_ham = ham_dict['y_stat_fd']
    # y1_ham     = ham_dict['y1_fd']
    # y2_ham     = ham_dict['y2_fd']
    # # Dynamical
    # dyn_dict = stats_frac_diff( dyn_pred,
    #                             dyn_true,
    #                             base=bin_width,
    #                             arr_len=arr_len,
    #                             bin_statval=bin_statval)
    # x_stat_dyn = dyn_dict['x_stat']
    # y_stat_dyn = dyn_dict['y_stat_fd']
    # y1_dyn     = dyn_dict['y1_fd']
    # y2_dyn     = dyn_dict['y2_fd']
    (   x_stat_ham   ,
        y_stat_ham   ,
        y_std_ham    ,
        y_std_err_ham) = cstats.Stats_one_arr(  ham_x,
                                                ham_frac_diff,
                                                base=bin_width,
                                                arr_len=arr_len,
                                                bin_statval=bin_statval)
    y1_ham = y_stat_ham - y_std_ham
    y2_ham = y_stat_ham + y_std_ham
    # Dynamical
    (   x_stat_dyn   ,
        y_stat_dyn   ,
        y_std_dyn    ,
        y_std_err_dyn) = cstats.Stats_one_arr(  dyn_x,
                                                dyn_frac_diff,
                                                base=bin_width,
                                                arr_len=arr_len,
                                                bin_statval=bin_statval)
    y1_dyn = y_stat_dyn - y_std_dyn
    y2_dyn = y_stat_dyn + y_std_dyn
    ##
    ## Figure details
    # Labels
    # X-label
    if (plot_opt == 'mgroup'):
        xlabel = r'\boldmath$\log M_{predicted}\left[ h^{-1} M_{\odot}\right]$'
    elif (plot_opt == 'mhalo'):
        xlabel = r'\boldmath$\log M_{halo,\textrm{true}}\left[ h^{-1} M_{\odot}\right]$'
    # Y-label
    ylabel = r'Frac. Difference \boldmath$[\%]$'
    ##
    plt.clf()
    plt.close()
    fig = plt.figure(figsize=figsize)
    ax1 = fig.add_subplot(111, facecolor='white')
    ## Horizontal line
    ax1.axhline(y=0, color='black', linestyle='--', zorder=10)
    ##
    ## HAM Masses
    ax1.plot(   x_stat_ham,
                y_stat_ham,
                color=plot_dict['color_ham'],
                linestyle='-',
                marker='o',
                zorder=zorder_mass)
    ax1.fill_between(   x_stat_ham,
                        y1_ham,
                        y2_ham, 
                        color=plot_dict['color_ham'],
                        alpha=alpha_mass,
                        label='HAM',
                        zorder=zorder_shade)
    ## Dynamical Masses
    ax1.plot(   x_stat_dyn,
                y_stat_dyn,
                color=plot_dict['color_dyn'],
                linestyle='-',
                marker='o',
                zorder=zorder_mass)
    ax1.fill_between(   x_stat_dyn,
                        y1_dyn,
                        y2_dyn, 
                        color=plot_dict['color_dyn'],
                        alpha=alpha_mass,
                        label='Dynamical',
                        zorder=zorder_shade)
    ## Legend
    leg = ax1.legend(loc='upper left', numpoints=1, frameon=False,
        prop={'size':14})
    leg.get_frame().set_facecolor('none')
    ## Ticks
    # Y-axis
    xaxis_major_ticker = 1
    xaxis_minor_ticker = 0.2
    ax_xaxis_major_loc = ticker.MultipleLocator(xaxis_major_ticker)
    ax_xaxis_minor_loc = ticker.MultipleLocator(xaxis_minor_ticker)
    ax1.xaxis.set_major_locator(ax_xaxis_major_loc)
    ax1.xaxis.set_minor_locator(ax_xaxis_minor_loc)
    # Y-axis
    yaxis_major_ticker = 5
    yaxis_minor_ticker = 2
    ax_yaxis_major_loc = ticker.MultipleLocator(yaxis_major_ticker)
    ax_yaxis_minor_loc = ticker.MultipleLocator(yaxis_minor_ticker)
    ax1.yaxis.set_major_locator(ax_yaxis_major_loc)
    ax1.yaxis.set_minor_locator(ax_yaxis_minor_loc)
    ## Labels
    ax1.set_xlabel(xlabel, fontsize=plot_dict['size_label'])
    ax1.set_ylabel(ylabel, fontsize=plot_dict['size_label'])
    ##
    ## Limits
    ax1.set_ylim(-20, 25)
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

## Covariance matrix of the different properties/features
def covariance_plot(catl_pd, param_dict, proj_dict, plot_only_feat=False,
    fig_fmt='pdf', figsize=(10, 8), fig_number=2):
    """
    Covariance matrix of different features

    Parameters
    -----------
    catl_pd : `pandas.DataFrame`
        DataFrame containing galaxy and group information

    param_dict : `dict`
        Dictionary with `project` variables

    proj_dict : `dict`
        Dictionary with info of the project that uses the
        `Data Science` Cookiecutter template.

    plot_only_feat : `bool`, optional
        If `True`, only the features used in the project will be plotted.
        This variable is set to 'False' by default.

    fig_fmt : {'pdf'} `str`, optional
        File extension used to save the figure. This variable is set to
        'pdf' by default.

    figsize : `tuple`, optional
        Size of the output figure. This variable is set to `(10, 8)` by
        default.

    fig_number : `int`, optional
        Number of figure in the workflow. This variable is set to `2`
        by default.
    """
    file_msg  = param_dict['Prog_msg']
    plot_dict = param_dict['plot_dict']
    ## Filename
    fname    = os.path.join(    proj_dict['figure_dir'],
                                'Fig_{0}_{1}_feature_covariance.{2}'.format(
                                    fig_number,
                                    param_dict['catl_str_fig'],
                                    fig_fmt))
    ##
    ## Paper Figure
    fname_paper = os.path.join( proj_dict['paper_fig_dir'],
                                'Figure_02.{0}'.format(fig_fmt))
    ## Renaming properties
    catl_pd_copy = catl_pd.copy()
    ## Dropping columns
    cols_drop = ['GG_pointing', 'GG_haloid_point', 'GG_mhalo_point', 
                    'g_brightest', 'GG_r_rms']
    catl_pd_copy.drop(cols_drop, axis=1, inplace=True)
    ## Reordering columns
    mhalo_key = 'M_h'
    # Saving data from Halo mass
    gal_mhalo_arr = catl_pd_copy[mhalo_key].values
    # Removing Halo mass
    catl_pd_copy.drop(mhalo_key, axis=1, inplace=True)
    # Inserting it back again to the DataFrame
    catl_pd_copy.insert(0, mhalo_key, gal_mhalo_arr)
    ##
    ## Rearranging columns
    df_cols_new = [ 'M_h','dist_centre_group', 'M_r', 'logssfr', 'g_galtype',
                    'g_r', 'GG_mr_brightest', 'GG_mr_ratio',
                    'GG_M_r', 'GG_logssfr', 'GG_shape', 'GG_ngals',
                    'GG_rproj', 'GG_r_tot', 'GG_r_med', 'GG_sigma_v',
                    'GG_sigma_v_rmed', 'GG_M_group', 'GG_mdyn_rproj',
                    'GG_dist_cluster']
    catl_pd_copy = catl_pd_copy.loc[:, df_cols_new]
    # Plotting only features if applicable
    if plot_only_feat:
        feat_cols    = param_dict['ml_args']._feature_cols()
        catl_pd_copy = catl_pd_copy.loc[:,feat_cols]
    ## Renaming
    catl_pd_copy.rename(columns=param_dict['feat_cols_dict'], inplace=True)
    ## Selecting certain columns only
    ## Figure details
    plt.clf()
    plt.close()
    fig = plt.figure(figsize=figsize)
    ax1 = fig.add_subplot(111, facecolor='white')
    fig.subplots_adjust(bottom=0.1, top=0.9, left=0.1, right=0.8,
                        wspace=0.02, hspace=0.02)
    cax = fig.add_axes([0.30, 0.85, 0.30, 0.05])
    ## Correlation
    corr = catl_pd_copy.corr()
    # Generate a mask for the upper triangle
    mask = num.zeros_like(corr, dtype=num.bool)
    mask[num.triu_indices_from(mask)] = True
    # Generate a custom diverging colormap
    cmap = sns.diverging_palette(220, 10, as_cmap=True)
    # Draw the heatmat with the mask and correct aspect ratio
    g = sns.heatmap(corr, mask=mask, cmap=cmap, vmax=1.0, vmin=-1., center=0,
        square=True, linewidths=0.5, cbar=False, ax=ax1)
    cbar = fig.colorbar(g.get_children()[0], cax=cax, orientation='horizontal')
    cbar.set_label(r'$\Leftarrow$ Correlation $\Rightarrow$',
        fontsize=plot_dict['size_label'])
    cbar.ax.tick_params(labelsize=20)
    g.yaxis.set_tick_params(labelsize=25)
    g.xaxis.set_tick_params(labelsize=25)
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

## Plots showing the points of HAM and `Dynamical`
def pred_masses_halo_mass(param_dict, proj_dict,
    arr_len=10, bin_statval='left', fig_fmt='pdf', figsize=(15, 12),
    fig_number=3):
    """
    Plots the `predicted` vs `true` mass for each of the `traditional`
    mass estimates.

    Parameters
    ------------
    param_dict : `dict`
        Dictionary with input parameters and values related to this project.

    proj_dict: python dictionary
        Dictionary with current and new paths to project directories
    
    arr_len : `int`, optional
        Minimum number of elements in bins. This variable is set to `0`
        by default.

    bin_statval : `str`, optional
        Option for where to plot the bin values. This variable is set
        to `left` by default.

        Options:
        - 'average': Returns the x-points at the average x-value of the bin
        - 'left'   : Returns the x-points at the left-edge of the x-axis bin
        - 'right'  : Returns the x-points at the right-edge of the x-axis bin

    fig_fmt : `str`, optional (default = 'pdf')
        Extension used to save the figure

    figsize : `tuple`, optional
        Size of the output figure. This variable is set to `(12,15.5)` by
        default.

    fig_number : `int`, optional
        Number of figure in the workflow. This variable is set to `4`
        by default.
    """
    file_msg      = param_dict['Prog_msg']
    ## Matplotlib option
    matplotlib.rcParams['axes.linewidth'] = 2.5
    matplotlib.rcParams['axes.edgecolor'] = 'black'
    # Constants
    cm            = plt.cm.get_cmap('viridis')
    plot_dict     = param_dict['plot_dict']
    alpha         = 0.2
    zorder_points = 5
    zorder_shade  = 8
    zorder_line   = 10
    markersize    = 1
    bin_width     = param_dict['ml_args'].mass_bin_width
    #
    # Figure name
    fname    = os.path.join(proj_dict['figure_dir'],
                            'Fig_{0}_{1}_mass_ham_dyn.{2}'.format(
                                fig_number,
                                param_dict['catl_str_fig'],
                                fig_fmt))
    ##
    ## Paper Figure
    fname_paper = os.path.join( proj_dict['paper_fig_dir'],
                                'Figure_03.{0}'.format(fig_fmt))
    #
    # Obtaining HAM and Dynamical masses
    # Looping over HAM and Dynamical
    pred_mass_dict = {}
    for kk, mass_kk in enumerate(['ham', 'dyn']):
        # Initializing dictionary
        pred_mass_dict[mass_kk] = {}
        # Getting masses
        if (mass_kk == 'ham'):
            (   mass_pred,
                mass_true) = param_dict['ml_args'].extract_trad_masses_alt(
                                mass_opt=mass_kk,
                                return_frac_diff=False)
        else:
            (   mass_pred,
                mass_true) = param_dict['ml_args'].extract_trad_masses_alt(
                                mass_opt=mass_kk,
                                return_frac_diff=False,
                                nlim_threshold=True,
                                nlim_min=4)
            # Only choosing values larger than 11
            mass_pred_mask = mass_pred >= 11.
            mass_pred = mass_pred[mass_pred_mask]
            mass_true = mass_true[mass_pred_mask]
        #
        # Binning data for the different masses
        (   x_stat,
            y_stat,
            y_std,
            y_std_err) = cstats.Stats_one_arr(  mass_pred,
                                                mass_true,
                                                base=bin_width,
                                                arr_len=arr_len,
                                                bin_statval=bin_statval)
        #
        # Lower and upper limits for erryrs
        y1 = y_stat - y_std
        y2 = y_stat + y_std
        #
        # Saving to dictionary
        pred_mass_dict[mass_kk]['x_stat'   ] = x_stat
        pred_mass_dict[mass_kk]['y_stat'   ] = y_stat
        pred_mass_dict[mass_kk]['y_std'    ] = y_std
        pred_mass_dict[mass_kk]['y1'       ] = y1
        pred_mass_dict[mass_kk]['y2'       ] = y2
        pred_mass_dict[mass_kk]['mass_pred'] = mass_pred
        pred_mass_dict[mass_kk]['mass_true'] = mass_true
    #
    # One-one line
    one_arr = num.linspace(0, 15, num=100)
    #
    # Figure details
    xlabel = r'\boldmath$\log M_{predicted}\left[h^{-1} M_{\odot}\right]$'
    # Y-axis
    ylabel = r'\boldmath$\log M_{\textrm{halo}}\left[h^{-1} M_{\odot}\right]$'
    # Plotting constants
    label_dict = {'ham': 'HAM', 'dyn': 'Dynamical Mass'}
    # Initializing figure
    plt.clf()
    plt.close()
    fig, axes = plt.subplots(nrows=1, ncols=2, sharex=True,
                    sharey=True, figsize=figsize)
    # Flattening out the axes
    axes = axes.flatten()
    # Plotting HAM and Dynamical
    for kk, mass_kk in enumerate(pred_mass_dict.keys()):
        # Axis
        ax = axes[kk]
        # Axis background color
        ax.set_facecolor('white')
        # Aspect ratio
        ax.set_aspect(aspect=1)
        # Mass dictionary
        mass_dict = pred_mass_dict[mass_kk]
        # Points
        ax.plot(mass_dict['mass_pred'],
                mass_dict['mass_true'],
                marker='o',
                linestyle='',
                markersize=markersize,
                color=plot_dict['color_{0}'.format(mass_kk)],
                alpha=alpha,
                zorder=zorder_points,
                rasterized=True)
        # Relation
        ax.plot(mass_dict['x_stat'],
                mass_dict['y_stat'],
                linestyle='-',
                color=plot_dict['color_{0}'.format(mass_kk)],
                marker='o',
                zorder=zorder_line,
                linewidth=5)
        # Shade
        ax.fill_between(mass_dict['x_stat'],
                        mass_dict['y1'],
                        mass_dict['y2'],
                        color=plot_dict['color_{0}'.format(mass_kk)],
                        alpha=alpha,
                        label=label_dict[mass_kk],
                        zorder=zorder_shade)
    # --- One-One Array and other settings
    # Constants
    # Setting limits
    xlim = (10.8, 15)
    ylim = (10.8, 15)
    # Mayor and minor locators
    xaxis_major = 1
    xaxis_minor = 0.2
    yaxis_major = 1
    yaxis_minor = 0.2
    xaxis_major_loc = ticker.MultipleLocator(xaxis_major)
    xaxis_minor_loc = ticker.MultipleLocator(xaxis_minor)
    yaxis_major_loc = ticker.MultipleLocator(yaxis_major)
    yaxis_minor_loc = ticker.MultipleLocator(yaxis_minor)
    # Looping over axes
    for kk, ax in enumerate(axes):
        ax.plot(one_arr, one_arr, linestyle='--', color='black', zorder=10)
        # Axis labels
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)
        # Labels
        ax.set_xlabel(xlabel, fontsize=plot_dict['size_label'])
        if (kk == 0):
            ax.set_ylabel(ylabel, fontsize=plot_dict['size_label'])
        # Mayor and minor locators
        ax.xaxis.set_major_locator(xaxis_major_loc)
        ax.xaxis.set_minor_locator(xaxis_minor_loc)
        ax.yaxis.set_major_locator(yaxis_major_loc)
        ax.yaxis.set_minor_locator(yaxis_minor_loc)
        # Legend
        ax.legend(loc='upper left', numpoints=1, frameon=False,
            prop={'size': 14})
    # Spacing
    plt.subplots_adjust(wspace=0.05)
    #
    # Saving figure
    ##
    ## Saving figure
    if (fig_fmt == 'pdf'):
        plt.savefig(fname, bbox_inches='tight', rasterize=True)
        plt.savefig(fname_paper, bbox_inches='tight', rasterize=True)
    else:
        plt.savefig(fname, bbox_inches='tight', dpi=400)
        plt.savefig(fname_paper, bbox_inches='tight', dpi=400)
    ##
    ##
    print('{0} Figure saved as: {1}'.format(file_msg, fname))
    print('{0} Paper Figure saved as: {1}'.format(file_msg, fname_paper))
    plt.clf()
    plt.close()

## Error in conventional methods - Galaxy Groups
def frac_diff_groups_model(param_dict, proj_dict, plot_opt='mhalo',
    nlim_min=5, nlim_threshold=False, arr_len=10, bin_statval='left',
    fig_fmt='pdf', figsize=(10, 8), fig_number=1):
    """
    Plots the fractional difference between `predicted` and `true`
    halo masses for galaxy GROUPS.

    Parameters
    -----------
    param_dict : `dict`
        Dictionary with input parameters and values related to this project.

    proj_dict: python dictionary
        Dictionary with current and new paths to project directories

    plot_opt : {'mgroup', 'mhalo'} `str`, optional  
        Option to which property to plot on the x-axis. This variable is set
        to `mhalo` by default.

    nlim_min : `int`, optional
        Minimum number of elements in a group, to show as part of the
        catalogue. This variable is set to `5` by default.

    nlim_threshold: `bool`, optional
        If `True`, only groups with number of members larger than
        `nlim_min` are included as part of the catalogue.

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
    """
    file_msg = param_dict['Prog_msg']
    ## Matplotlib option
    matplotlib.rcParams['axes.linewidth'] = 2.5
    matplotlib.rcParams['axes.edgecolor'] = 'black'
    ##
    # Constants
    cm           = plt.cm.get_cmap('viridis')
    plot_dict    = param_dict['plot_dict']
    ham_color    = 'red'
    alpha        = 0.6
    alpha_mass   = 0.2
    zorder_mass  = 10
    zorder_shade = zorder_mass - 1
    zorder_ml    = zorder_mass + 1
    bin_width    = param_dict['ml_args'].mass_bin_width
    ##
    ## Figure name
    fname = os.path.join(   proj_dict['figure_dir'],
                            'Fig_{0}a_{1}_group_mass_comparison.{2}'.format(
                                fig_number,
                                param_dict['catl_str_fig'],
                                fig_fmt))
    ##
    ## Paper Figure
    fname_paper = os.path.join( proj_dict['paper_fig_dir'],
                                'Figure_01a.{0}'.format(fig_fmt))
    ##
    ## Reading in 'master' combined catalogue
    catl_pd_tot = param_dict['ml_args'].extract_merged_catl_info(
        opt='combined')
    ##
    ## Only selecting groups with `nlim_min` galaxies or larger
    if nlim_threshold:
        catl_pd_tot = catl_pd_tot.loc[(catl_pd_tot['GG_ngals'] >= nlim_min)]
    ##
    ## Temporarily fixing `GG_mdyn_rproj`
    # catl_pd_tot.loc[:, 'GG_mdyn_rproj'] /= 0.96
    ##
    ## Dropping NaN's
    catl_pd_tot.dropna(how='any', inplace=True)
    ##
    ## Selecting only 'Good' groups
    good_p_opt    = int(1)
    catl_cols_arr = ['GG_pointing', 'GG_mhalo_point', 'GG_M_group',
                        'GG_mdyn_rproj', 'groupid']
    catl_pd_tot_mod = catl_pd_tot.loc[catl_pd_tot['GG_pointing'] == good_p_opt,
                        catl_cols_arr].drop_duplicates().reset_index(drop=True)
    ##
    ## Selecting Masses
    # - HAM Mass -
    ham_cols = ['GG_M_group', 'GG_mhalo_point']
    (   ham_pred,
        ham_true) = catl_pd_tot_mod.loc[:, ham_cols].values.T
    ham_frac_diff = 100. * (ham_pred - ham_true) / ham_true
    # - DYN Mass -
    dyn_cols = ['GG_mdyn_rproj', 'GG_mhalo_point']
    (   dyn_pred,
        dyn_true) = catl_pd_tot_mod.loc[
                        catl_pd_tot_mod['GG_mdyn_rproj'] >= 11.0,
                            dyn_cols].values.T
    dyn_frac_diff = 100. * (dyn_pred - dyn_true) / dyn_true
    ##
    ## Choosing which mass to plot
    if (plot_opt == 'mgroup'):
        ham_x = ham_pred
        dyn_x = dyn_pred
    elif (plot_opt == 'mhalo'):
        ham_x = ham_true
        dyn_x = dyn_true
    ##
    ## Binning the data
    ## -- HAM --
    (   x_stat_ham   ,
        y_stat_ham   ,
        y_std_ham    ,
        y_std_err_ham) = cstats.Stats_one_arr(  ham_x,
                                                ham_frac_diff,
                                                base=bin_width,
                                                arr_len=arr_len,
                                                bin_statval=bin_statval)
    y1_ham = y_stat_ham - y_std_ham
    y2_ham = y_stat_ham + y_std_ham
    ## -- DYN --
    (   x_stat_dyn   ,
        y_stat_dyn   ,
        y_std_dyn    ,
        y_std_err_dyn) = cstats.Stats_one_arr(  dyn_x,
                                                dyn_frac_diff,
                                                base=bin_width,
                                                arr_len=arr_len,
                                                bin_statval=bin_statval)
    y1_dyn = y_stat_dyn - y_std_dyn
    y2_dyn = y_stat_dyn + y_std_dyn
    ## Figure details
    # Labels
    # X-label
    if (plot_opt == 'mgroup'):
        xlabel = r'\boldmath$\log M_{predicted}  \textrm{ - Groups}\left[ h^{-1} M_{\odot}\right]$'
    elif (plot_opt == 'mhalo'):
        xlabel = r'\boldmath$\log M_{halo,\textrm{true}} \textrm{ - Groups}\left[ h^{-1} M_{\odot}\right]$'
    # Y-label
    ylabel = r'Frac. Difference - Groups \boldmath$[\%]$'
    ##
    plt.clf()
    plt.close()
    fig = plt.figure(figsize=figsize)
    ax1 = fig.add_subplot(111, facecolor='white')
    ## Horizontal line
    ax1.axhline(y=0, color='black', linestyle='--', zorder=10)
    ##
    ## HAM Masses
    ax1.plot(   x_stat_ham,
                y_stat_ham,
                color=plot_dict['color_ham'],
                linestyle='-',
                marker='o',
                zorder=zorder_mass)
    ax1.fill_between(   x_stat_ham,
                        y1_ham,
                        y2_ham, 
                        color=plot_dict['color_ham'],
                        alpha=alpha_mass,
                        label='HAM',
                        zorder=zorder_shade)
    ## Dynamical Masses
    ax1.plot(   x_stat_dyn,
                y_stat_dyn,
                color=plot_dict['color_dyn'],
                linestyle='-',
                marker='o',
                zorder=zorder_mass)
    ax1.fill_between(   x_stat_dyn,
                        y1_dyn,
                        y2_dyn, 
                        color=plot_dict['color_dyn'],
                        alpha=alpha_mass,
                        label='Dynamical',
                        zorder=zorder_shade)
    ## Legend
    leg = ax1.legend(loc='upper left', numpoints=1, frameon=False,
        prop={'size':14})
    leg.get_frame().set_facecolor('none')
    ## Ticks
    # Y-axis
    xaxis_major_ticker = 1
    xaxis_minor_ticker = 0.2
    ax_xaxis_major_loc = ticker.MultipleLocator(xaxis_major_ticker)
    ax_xaxis_minor_loc = ticker.MultipleLocator(xaxis_minor_ticker)
    ax1.xaxis.set_major_locator(ax_xaxis_major_loc)
    ax1.xaxis.set_minor_locator(ax_xaxis_minor_loc)
    # Y-axis
    yaxis_major_ticker = 5
    yaxis_minor_ticker = 2
    ax_yaxis_major_loc = ticker.MultipleLocator(yaxis_major_ticker)
    ax_yaxis_minor_loc = ticker.MultipleLocator(yaxis_minor_ticker)
    ax1.yaxis.set_major_locator(ax_yaxis_major_loc)
    ax1.yaxis.set_minor_locator(ax_yaxis_minor_loc)
    ## Labels
    ax1.set_xlabel(xlabel, fontsize=plot_dict['size_label'])
    ax1.set_ylabel(ylabel, fontsize=plot_dict['size_label'])
    ##
    ## Limits
    ax1.set_ylim(-10, 10)
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


## --------- Main Function ------------##

def main(args):
    """
    Script that uses a set of ML algorithms to try to predict galaxy and
    group properties by using a set of calculated `features` from synthetic
    galaxy/group catalogues .
    """
    ## Starting time
    start_time = datetime.now()
    ## Reading all elements and converting to python dictionary
    param_dict = vars(args)
    ## Checking for correct input
    param_vals_test(param_dict)
    #
    # Creating instance of `ReadML` with the input parameters
    param_dict['ml_args'] = ReadML(**param_dict)
    ## Program message
    prog_msg = param_dict['Prog_msg']
    # Adding additional parameters
    param_dict = add_to_dict(param_dict)
    ##
    ## Creating Folder Structure
    # proj_dict = cwpaths.cookiecutter_paths(__file__)
    proj_dict = param_dict['ml_args'].proj_dict
    proj_dict = directory_skeleton(param_dict, proj_dict)
    ##
    ## Printing out project variables
    print('\n'+50*'='+'\n')
    for key, key_val in sorted(param_dict.items()):
        if key != 'Prog_msg':
            print('{0} `{1}`: {2}'.format(prog_msg, key, key_val))
    print('\n'+50*'='+'\n')
    ##
    ## Feature keys
    param_dict['feat_cols_dict'] = param_dict['ml_args'].feat_cols_names_dict(
                                        return_all=True)
    ##
    ## Reading in the main catalogue
    catl_pd = catl_file_read_clean(param_dict, proj_dict)
    ###
    ### ------ Figures ------ ###
    ##
    ## Comparison of estimated group masses via HAM and Dynamical Masses
    frac_diff_model(param_dict, proj_dict, plot_opt=param_dict['plot_opt'])
    #
    # Covariance Matrix
    covariance_plot(catl_pd, param_dict, proj_dict)
    #
    # Traditional methods for estimating masses
    # pred_masses_halo_mass(param_dict, proj_dict)
    #
    # Fractional Difference plots vs True mass of galaxy GROUPS
    # frac_diff_groups_model(param_dict, proj_dict,
    #     plot_opt=param_dict['plot_opt'])
    ##
    ## End time for running the catalogues
    end_time = datetime.now()
    total_time = end_time - start_time
    print('{0} Total Time taken (Create): {1}'.format(prog_msg, total_time))

# Main function
if __name__ == '__main__':
    ## Input arguments
    args = get_parser()
    # Main Function
    main(args)
