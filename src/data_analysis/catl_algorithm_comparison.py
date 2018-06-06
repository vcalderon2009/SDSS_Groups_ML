#! /usr/bin/env python
# -*- coding: utf-8 -*-

# Victor Calderon
# Created      : 2018-06-03
# Last Modified: 2018-06-03
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
import sklearn.metrics           as skmetrics
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
    ## Dictionary of ML Regressors
    skem_dict = sklearns_models(param_dict, cpu_number)
    ##
    ## Saving to `param_dict`
    param_dict['sample_s'  ] = sample_s
    param_dict['sample_Mr' ] = sample_Mr
    param_dict['vol_mr'    ] = vol_mr
    param_dict['cens'      ] = cens
    param_dict['sats'      ] = sats
    param_dict['speed_c'   ] = speed_c
    param_dict['cpu_number'] = cpu_number
    param_dict['skem_dict' ] = skem_dict

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
    #
    # Main output file for this script
    out_dir = param_dict['ml_args'].catl_train_alg_comp_dir(check_exist=True,
                create_dir=True)
    #
    # Adding to `proj_dict`
    proj_dict['out_dir'] = out_dir

    return proj_dict

## --------- Preparing data ------------##

# Different types of regressors
def sklearns_models(param_dict, cpu_number):
    """
    Defines the set of Regressors used by Scikit-Learn

    Parameters
    -----------
    param_dict : `dict`
        Dictionary with input parameters and values related to this project.

    cpu_number : `int`
        Number of CPU's to use.

    Returns
    ----------
    skem_dict : `dict`
        Dictionary with a set of regressors uninitialized
    """
    # Dictionary with regressors
    skem_dict = {}
    # Random Forest
    skem_dict['random_forest'] = skem.RandomForestRegressor(
                                    n_jobs=cpu_number,
                                    random_state=param_dict['seed'])
    # XGBoost
    skem_dict['XGBoost'] = xgboost.XGBRegressor(
                            n_jobs=cpu_number,
                            random_state=param_dict['seed'])
    # Neural Network
    if (param_dict['hidden_layers'] == 1):
        hidden_layer_obj = (param_dict['unit_layer'],)
    elif (param_dict['hidden_layers'] > 1):
        hidden_layers = param_dict['hidden_layers']
        unit_layer    = param_dict['unit_layer']
        hidden_layer_obj = tuple([unit_layer for x in range(hidden_layers)])
    skem_dict['neural_network'] = skneuro.MLPRegressor(
                                    random_state=param_dict['seed'],
                                    solver='adam',
                                    hidden_layer_sizes=hidden_layer_obj,
                                    warm_start=True)

    return skem_dict

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

## --------- Main Analysis of the data ------------##

# Finding indices of bins
def binning_idx(train_dict, test_dict, param_dict, mass_opt='group'):
    """
    Finds the indices for the different sets of dictionaries, based on
    the parameter `bin_val`

    Parameters
    ------------
    test_dict_ii : `dict`
        Dictionary with the `training` data.

    train_dict_ii : `dict`
        Dictionary with the `testing` data.

    param_dict : `dict`
        Dictionary with input parameters and values related to this project.

    mass_opt : {'group', 'halo'} `bool`, optional
        Option for which mass to use when binning.

    Returns
    ---------
    train_idx_bins : `numpy.ndarray`, shape [N, n_bins]
        Indices from the `train_dict` for each bin in halo mass.

    test_idx_bins : `numpy.ndarray`, shape [N, n_bins]
        Indices from the `test_dict` for each bin in halo mass.

    """
    # Predicted columns
    pred_cols = num.array(param_dict['ml_args']._predicted_cols())
    # Feature columns
    feat_cols = num.array(param_dict['ml_args']._feature_cols())
    # Unpacking dictionaries
    X_train_ns = train_dict['X_train_ns']
    Y_train    = train_dict['Y_train']
    X_test_ns  = test_dict ['X_test_ns']
    Y_test     = test_dict ['Y_test']
    ### --- Group mass --- ###
    # Unpacking `estimated` group mass
    if ((len(feat_cols) == 1) and ('GG_M_group' in feat_cols)):
        mgroup_train = X_train_ns
        mgroup_test  = X_test_ns
    elif ((len(feat_cols) > 1) and ('GG_M_group' in feat_cols)):
        # Group mass index
        mgroup_idx = num.where(feat_cols == 'GG_M_group')[0]
        # Training and testing arrays
        mgroup_train = X_train_ns.T[mgroup_idx].flatten()
        mgroup_test  = X_test_ns.T[mgroup_idx].flatten()
    ### --- Halo mass --- ###
    # Unpacking `true` halo mass array
    if ((param_dict['n_predict'] == 1) and ('M_h' in pred_cols)):
        mhalo_train = Y_train
        mhalo_test  = Y_test
    elif ((param_dict['n_predict'] > 1) and ('M_h' in pred_cols)):
        # Halo mass index
        mhalo_idx = num.where(pred_cols == 'M_h')[0]
        # Training and testing arrays
        mhalo_train = Y_train.T[mhalo_idx].flatten()
        mhalo_test  = Y_test.t[mhalo_idx].flatten()
    #
    # Indices for `mass`
    if (mass_opt == 'group'):
        mass_train = mgroup_train
        mass_test  = mgroup_test
    elif (mass_opt == 'halo'):
        mass_train = mhalo_train
        mass_test  = mhalo_test
    ##
    ## Indices for `training` and `testing`
    train_idx = num.arange(len(mass_train))
    test_idx  = num.arange(len(mass_test))
    #
    ## Binning data
    # Evenly-spaced bins
    if (param_dict['bin_val'] == 'fixed'):
        # Bin width for `mhalo`
        bin_width = param_dict['ml_args'].mass_bin_width
        # Selecting boundaries
        mass_min  = num.max([mass_train.min(), mass_test.min()])
        mass_max  = num.min([mass_train.max(), mass_test.max()])
        mass_bins = num.array([mass_min, mass_max])
        ## - Training
        # Creating bins
        train_bins = cstats.Bins_array_create(mass_bins, base=bin_width)
        # Digitizing array
        train_digits = num.digitize(mass_train, train_bins)
        # Total number of bins indices
        train_digits_idx = num.arange(1, len(train_bins))
        # Indices in each bin
        train_idx_bins = num.array([train_idx[train_digits == ii]
                            for ii in train_digits_idx])
        ##
        ## - Testing
        # Creating bins
        test_bins = cstats.Bins_array_create(mass_bins, base=bin_width)
        # Digitizing array
        test_digits = num.digitize(mass_test, test_bins)
        # Total number of bins indices
        test_digits_idx = num.arange(1, len(test_bins))
        # Indices in each bin
        test_idx_bins = num.array([test_idx[test_digits == ii]
                            for ii in test_digits_idx])
    #
    # Fixed number of bins
    if (param_dict['bin_val'] == 'nbins'):
        # Selecting boundary
        nbins = param_dict['ml_args'].nbins
        mass_min  = num.min([mass_train.min(), mass_test.min()])
        mass_max  = num.max([mass_train.max(), mass_test.max()])
        mass_bins = num.linspace(mass_min, mass_max, nbins + 1)
        ## -- Training
        # Digitizing array
        train_digits = num.digitize(mass_train, mass_bins)
        # Total number of bins indices
        train_digits_idx = num.arange(1, len(mass_bins))
        # Indices in each bin
        train_idx_bins = num.array([train_idx[train_digits == ii]
                            for ii in train_digits_idx])
        ## -- Testing
        # Digitizing array
        test_digits = num.digitize(mass_test, mass_bins)
        # Total number of bins indices
        test_digits_idx = num.arange(1, len(mass_bins))
        # Indices in each bin
        test_idx_bins = num.array([test_idx[test_digits == ii]
                            for ii in test_digits_idx])

    return train_idx_bins, test_idx_bins

# General Model metrics
def model_metrics(skem_ii, test_dict_ii, train_dict_ii, param_dict):
    """
    Determines the general metrics of a model.

    Parameters
    -----------
    skem_ii : `str`
        Key of the Regressor being used. Taken from `skem_dict` dictionary.

    test_dict_ii : `dict`
        Dictionary with the `training` data.

    train_dict_ii : `dict`
        Dictionary with the `testing` data.

    param_dict : `dict`
        Dictionary with input parameters and values related to this project.

    Returns
    --------
    model_gen_dict : `dict`
        Dictionary with the set of general model metrics, i.e.
            - Model instance
            - Model score
            - Array of predicted values
            - Array of `true` halo masses
            - Fractional difference between `true` and `predicted`
            - Feature importance when applicable
    """
    ## Constants
    # Feature columns
    feat_cols = num.array(param_dict['ml_args']._feature_cols())
    # Predicted columns
    pred_cols = num.array(param_dict['ml_args']._predicted_cols())
    ##
    ## Unpacking dictionariesx
    # Training
    X_train_ii    = train_dict_ii['X_train']
    X_train_ns_ii = train_dict_ii['X_train_ns']
    Y_train_ii    = train_dict_ii['Y_train']
    # Testing
    X_test_ii     = test_dict_ii['X_test']
    X_test_ns_ii  = test_dict_ii['X_test_ns']
    Y_test_ii     = test_dict_ii['Y_test']
    ##
    ## Training model
    #
    # Initializing model
    model_ii = sklearn.base.clone(param_dict['skem_dict'][skem_ii])
    #
    # Training model
    model_ii.fit(X_train_ii, Y_train_ii)
    #
    # Overall Score
    model_ii_score = cmlu.scoring_methods(Y_test_ii,
                            feat_arr=X_test_ii,
                            model=model_ii,
                            score_method=param_dict['score_method'],
                            threshold=param_dict['threshold'],
                            perc=param_dict['perc_val'])
    #
    # Predicting outputs
    pred_ii_arr = model_ii.predict(X_test_ii)
    #
    ## Fractional difference - Mass
    # Array of `estimated` mass
    if ((param_dict['n_predict'] == 1) and ('M_h' in pred_cols)):
        # Array of `true` halo mass
        mhalo_ii_arr      = Y_test_ii
        mhalo_ii_pred_arr = pred_ii_arr
    elif ((param_dict['n_predict'] > 1) and ('M_h' in pred_cols)):
        # Halo mass index
        mhalo_idx = num.where(pred_cols == 'M_h')[0]
        # Array of `true` halo mass
        mhalo_ii_arr = Y_test_ii.T[mhalo_idx].flatten()
        # Array of `predicted` halo mass
        mhalo_ii_pred_arr = pred_ii_arr.T[mhalo_idx].flatten()
    # Calculation
    frac_diff_ii = 100. * (mhalo_ii_pred_arr - mhalo_ii_arr) / mhalo_ii_arr
    #
    # Feature importance
    if not (skem_ii == 'neural_network'):
        # Importance
        feat_importance_ii = model_ii.feature_importances_
        # Rank
    else:
        # Importance
        feat_importance_ii = num.ones(feat_cols.shape) * num.nan
        # Rank
    #
    # Estimated Group mass
    if ((len(feat_cols) == 1) and ('GG_M_group' in feat_cols)):
        mgroup_arr = X_test_ns_ii
    elif ((len(feat_cols) > 1) and ('GG_M_group' in feat_cols)):
        # Group mass idx
        mgroup_idx = num.where(feat_cols == 'GG_M_group')[0]
        # Group mass array
        mgroup_arr = X_test_ns_ii.T[mgroup_idx].flatten()
    else:
        mgroup_arr = num.ones(len(X_test_ns_ii)) * num.nan
    #
    # Estimated Group mass
    if ((len(feat_cols) == 1) and ('GG_mdyn_rproj' in feat_cols)):
        mdyn_arr = X_test_ns_ii
    elif ((len(feat_cols) > 1) and ('GG_mdyn_rproj' in feat_cols)):
        # Group mass idx
        mdyn_idx = num.where(feat_cols == 'GG_mdyn_rproj')[0]
        # Group mass array
        mdyn_arr = X_test_ns_ii.T[mdyn_idx].flatten()
    else:
        mdyn_arr = num.ones(len(X_test_ns_ii)) * num.nan
    #
    # Reshaping array
    # if (len(pred_cols) == 1):
    #     pred_ii_arr_sh = pred_ii_arr.reshape((1, len(pred_ii_arr)))
    #     true_vals_sh   = Y_test_ii.reshape((1, len(Y_test_ii)))
    # else:
    pred_ii_arr_sh = pred_ii_arr
    true_vals_sh   = Y_test_ii
    #
    # Saving to dictionary
    model_gen_dict = {  'model_ii'  : model_ii,
                        'score'     : model_ii_score,
                        'mhalo_pred': mhalo_ii_pred_arr,
                        'mhalo_true': mhalo_ii_arr,
                        'frac_diff' : frac_diff_ii,
                        'feat_imp'  : feat_importance_ii,
                        'mgroup_arr': mgroup_arr,
                        'mdyn_arr'  : mdyn_arr,
                        'pred_vals' : pred_ii_arr_sh,
                        'true_vals' : true_vals_sh}

    return model_gen_dict

# ML Models training wrapper
def ml_models_training(models_dict, param_dict, proj_dict):
    """
    Trains different ML models to determine metrics of accuracy, feature
    importance, and more for a given ML algorithm.

    Parameters
    ------------
    models_dict : `dict`
        Dictionary for storing the outputs of the different ML algorithms.

    param_dict : `dict`
        Dictionary with input parameters and values related to this project.

    proj_dict: python dictionary
        Dictionary with current and new paths to project directories

    Returns
    ------------
    models_dict : `dict`
        Dictionary for storing the outputs of the different ML algorithms.
    """
    file_msg = param_dict['Prog_msg']
    ##
    ## Preparing the data
    train_dict, test_dict = param_dict['ml_args'].extract_feat_file_info()
    # List of the different ML models
    skem_keys_arr = num.sort(list(param_dict['skem_dict'].keys()))
    # Looping over each ML algorithm
    for zz, skem_ii in tqdm(enumerate(skem_keys_arr)):
        print('{0} Analyzing: `{1}`'.format(file_msg, skem_ii))
        # Training datset
        models_dict[skem_ii] = ml_analysis(skem_ii, train_dict, test_dict,
                                    param_dict, proj_dict)

    return models_dict

# Main Analysis for fixed HOD and DV
def ml_analysis(skem_ii, train_dict, test_dict, param_dict, proj_dict):
    """
    Main analysis for fixed `HOD` and velocity bias.
    This functions determines:
        - Fractional difference between `predicted` and `true` values.
        - Feature importance for each algorithm
        - 1-sigma error in log(M_halo)

    Parameters
    ------------
    skem_ii : `str`
        Key of the Regressor being used. Taken from `skem_dict` dictionary.

    train_dict : `dict`
        Dictionary with the `training` data.

    test_dict : `dict`
        Dictionary with the `testing` data.

    param_dict : `dict`
        Dictionary with input parameters and values related to this project.

    proj_dict: python dictionary
        Dictionary with current and new paths to project directories

    Returns
    --------
    ml_model_dict : `dict`
        Dictionary containing metrics for the ML model.

        Keys :
            - ''
    """
    ## Constants
    # Feature columns
    feat_cols     = num.array(param_dict['ml_args']._feature_cols())
    feat_cols_idx = num.arange(len(feat_cols)) + 1
    #
    ## Determining type of `sample_method`
    # `Normal` sample Method
    if (param_dict['sample_method'] == 'normal'):
        ml_model_dict = model_metrics(skem_ii, test_dict, train_dict,
                            param_dict)
    # `Binning` sample method
    if (param_dict['sample_method'] == 'binning'):
        # Dictionaries with the indices from the `training` and `testing`
        # datasets, at each halo mass bin.
        train_idx_bins, test_idx_bins = binning_idx(train_dict, test_dict,
                                            param_dict)
        ##
        ## Looping over bins, and training each independently
        ml_model_dict = {}
        # Looping over bins
        for ii in tqdm(range(len(train_idx_bins))):
            ## -- Constructing new dictionaries
            # Training
            train_idx_ii   = train_idx_bins[ii]
            train_dict_bin = {k: train_dict[k][train_idx_ii]
                                for k in train_dict.keys()}
            # Testing
            test_idx_ii   = test_idx_bins[ii]
            test_dict_bin = {k: test_dict[k][test_idx_ii]
                                for k in test_dict.keys()}
            ##
            ## -- Metrics
            model_metrics_ii = model_metrics(   skem_ii,
                                                test_dict_bin,
                                                train_dict_bin,
                                                param_dict)
            ##
            ## -- Concatenatng Arrays and expanding arrays
            if (ii == 0):
                # `mhalo_pred`
                mhalo_pred_main = model_metrics_ii['mhalo_pred']
                # `mhalo_true`
                mhalo_true_main = model_metrics_ii['mhalo_true']
                # `frac_diff`
                frac_diff_main  = model_metrics_ii['frac_diff']
                # `mgroup_arr`
                mgroup_main     = model_metrics_ii['mgroup_arr']
                # Dynamical mass
                mdyn_main       = model_metrics_ii['mdyn_arr']
                # `feat_imp`
                feat_imp_main   = model_metrics_ii['feat_imp']
                # `score`
                score_main      = [model_metrics_ii['score']]
                # `models
                models_main     = [model_metrics_ii['model_ii']]
                # `pred_vals`
                pred_vals_main  = model_metrics_ii['pred_vals']
                # `Expected values`
                exp_vals_main   = model_metrics_ii['true_vals']
            else:
                # `mhalo_pred`
                mhalo_pred_main = array_insert(mhalo_pred_main,
                                    model_metrics_ii['mhalo_pred'],
                                    axis=0)
                # `mhalo_true`
                mhalo_true_main = array_insert(mhalo_true_main,
                                    model_metrics_ii['mhalo_true'],
                                    axis=0)
                # `frac_diff`
                frac_diff_main  = array_insert(frac_diff_main,
                                    model_metrics_ii['frac_diff'],
                                    axis=0)
                # `mgroup_arr`
                mgroup_main     = array_insert(mgroup_main,
                                    model_metrics_ii['mgroup_arr'],
                                    axis=0)
                # Dynamical mass
                mdyn_main       = array_insert(mdyn_main,
                                    model_metrics_ii['mdyn_arr'],
                                    axis=0)
                # `feat_imp`
                feat_imp_temp   = model_metrics_ii['feat_imp']
                feat_imp_main   = num.column_stack((feat_imp_main,
                                    feat_imp_temp))
                # `score`
                score_main.append(model_metrics_ii['score'])
                # `models`
                models_main.append(model_metrics_ii['model_ii'])
                # `pred_vals`
                
                pred_vals_main = num.concatenate((
                                        pred_vals_main,
                                        model_metrics_ii['pred_vals']))
                # `Expected values`
                exp_vals_main = num.concatenate((
                                        exp_vals_main,
                                        model_metrics_ii['true_vals']))

        ###
        ### Calculating TOTAL score for `predicted` and `expected`
        if (param_dict['score_method'] == 'model_score'):
            score_type = 'r2'
        else:
            score_type = param_dict['score_method']
        # Total score
        total_score = cmlu.scoring_methods(exp_vals_main,
                        pred_arr=pred_vals_main,
                        score_method=score_type,
                        threshold=param_dict['threshold'],
                        perc=param_dict['perc_val'])
        ##
        ## -- Feature Importance - Mean
        # feat_imp_mean = num.mean(feat_imp_main.T, axis=1)
        feat_imp_mean = num.average(feat_imp_main, axis=1,
                            weights=[len(x) for x in train_idx_bins])
        ##
        ## -- Adding values to main dictionary
        ml_model_dict['model_ii'  ] = models_main
        ml_model_dict['score'     ] = total_score
        ml_model_dict['score_all' ] = score_main
        ml_model_dict['mhalo_pred'] = mhalo_pred_main
        ml_model_dict['mhalo_true'] = mhalo_true_main
        ml_model_dict['frac_diff' ] = frac_diff_main
        ml_model_dict['mgroup_arr'] = mgroup_main
        ml_model_dict['mdyn_arr'  ] = mdyn_main
        ml_model_dict['feat_imp'  ] = feat_imp_mean
    ##
    ## -- Feature importance ranking --
    feat_imp_comb      = num.vstack(zip(feat_cols, ml_model_dict['feat_imp']))
    feat_imp_comb_idx  = num.argsort(feat_imp_comb[:, 1])[::-1]
    feat_imp_comb_sort = feat_imp_comb[feat_imp_comb_idx]
    # Adding rank indices
    feat_imp_comb_sort = num.column_stack((feat_imp_comb_sort,
                            num.arange(len(feat_cols)) + 1))
    #
    # Adding to dictionary
    ml_model_dict['feat_imp_sort'] = feat_imp_comb_sort

    return ml_model_dict

## --------- Saving Data ------------##

def test_alg_comp_file(param_dict, proj_dict):
    """
    Determines whether or not to run the calculations.

    Parameters
    ----------
    param_dict : `dict`
        Dictionary with `project` variables

    proj_dict : `dict`
        Dictionary with info of the project that uses the
        `Data Science` Cookiecutter template.

    Returns
    ----------
    run_opt : `bool`
        If True, the whole analysis is run.

    param_dict : `dict`
        Dictionary with `project` variables
    """
    ## Filename, under which to save all of the information
    ##
    ## Path to output file
    filepath = param_dict['ml_args'].catl_train_alg_comp_file(
                    check_exist=False)
    ## Saving
    ##
    ## Checking if to run or not
    if os.path.exists(filepath):
        if param_dict['remove_files']:
            os.remove(filepath)
            run_opt = True
        else:
            run_opt = False
    else:
        run_opt = True
    ##
    ## Saving name to dictionary
    param_dict['filepath'] = filepath

    return run_opt, param_dict

def saving_data(models_dict, param_dict, proj_dict, ext='p'):
    """
    Saves the final data file to directory.

    Parameters
    ------------
    models_dict : `dict`
        Dictionary containing the results from the ML analysis.

    param_dict : `dict`
        Dictionary with input parameters and values related to this project.

    proj_dict: python dictionary
        Dictionary with current and new paths to project directories

    ext : `str`, optional
        Extension of the file. This variable is set to `p` by default, i.e.
        for `pickle file`.
    """
    file_msg = param_dict['Prog_msg']
    # Filename
    filepath = param_dict['ml_args'].catl_train_alg_comp_file(
                    check_exist=False)
    # Elements to be saved
    obj_arr = [models_dict]
    # Saving pickle file
    with open(filepath, 'wb') as file_p:
        pickle.dump(obj_arr, file_p)
    # Checking the file exists
    if not (os.path.exists(filepath)):
        msg = '{0} `filepath` ({1}) was not found!!'.format(file_msg, filepath)
        raise FileNotFoundError(msg)
    #
    # Output message
    msg = '{0} Output file: `{1}`'.format(file_msg, filepath)
    print(msg)

## --------- Main Function ------------##

def main(args):
    """
    Script that uses a set of ML algorithms to try to predict galaxy and
    group properties by using a set of calculated `features` from synthetic
    galaxy/group catalogues.
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
    ## -------- ML main analysis -------- ##
    ##
    ## Testing of whether or not to run the analysis
    (   run_opt   ,
        param_dict) = test_alg_comp_file(param_dict, proj_dict)
    # Analysis
    if run_opt:
        # Dictionary for storing outputs
        models_dict = {}
        #
        # Analyzing data
        models_dict = ml_models_training(models_dict, param_dict, proj_dict)
        #
        ## -------- Saving final results -------- ##
        # Saving `models_dict`
        saving_data(models_dict, param_dict, proj_dict)
    else:
        ##
        ## Output message
        assert(os.path.exists(param_dict['filepath']))
        msg = '{0} Output file: {1}'.format(prog_msg, param_dict['filepath'])
        print(msg)
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
