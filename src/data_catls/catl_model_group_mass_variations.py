#! /usr/bin/env python
# -*- coding: utf-8 -*-

# Victor Calderon
# Created      : 2018-10-09
# Last Modified: 2018-10-09
# Vanderbilt University
from __future__ import absolute_import, division, print_function
__author__     = ['Victor Calderon']
__copyright__  = ["Copyright 2018 Victor Calderon, "]
__email__      = ['victor.calderon@vanderbilt.edu']
__maintainer__ = ['Victor Calderon']
"""
Script that uses the final version of the galaxy catalogue of the
SDSS data and cross-references it to the original dataset. It then
looks at how different the predicted masses for individual galaxies
are at fixed mass and on a group-per-group basis.
"""
# Importing Modules
from cosmo_utils       import mock_catalogues as cm
from cosmo_utils       import utils           as cu
from cosmo_utils.utils import file_utils      as cfutils
from cosmo_utils.utils import file_readers    as cfreaders
from cosmo_utils.utils import work_paths      as cwpaths
from cosmo_utils.utils import stats_funcs     as cstats
from cosmo_utils.utils import geometry        as cgeom
from cosmo_utils.ml    import ml_utils        as cmlu
from cosmo_utils.mock_catalogues import catls_utils as cmcu

from src.ml_tools import ReadML

import numpy as num
import math
import os
import sys
import pandas as pd
import pickle
import matplotlib
matplotlib.use( 'Agg' )
import matplotlib.pyplot as plt
plt.rc('text', usetex=True)
import seaborn as sns


import astropy.constants as ac
import astropy.units     as u

from datetime import datetime
from collections import Counter

# Extra-modules
import argparse
from argparse import ArgumentParser
from argparse import HelpFormatter
from operator import attrgetter
from tqdm import tqdm, trange
import itertools

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
    val : `int` or float`
        Value to be evaluated by `val_min`

    val_min: `float` or `int`, optional
        Minimum value that `val` can be. This variable is set to `0` by default

    Returns
    -------
    ival : `float`
        Value if `val` is larger than `val_min`

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
    Script that uses the already-trained ML models to estimate the
    masses of groups, and adds them to the already-defined
    masses from HAM and Dynamical Mass methods.
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
    """
    Check whether `name` is on PATH and marked as executable.
    """
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
    # Catalogue prefix
    catl_str_fig = param_dict['ml_args'].catl_model_pred_plots_prefix_str()
    ##
    ## Choice of Centrals and Satellites
    cens = int(1)
    sats = int(0)
    ## Other constants
    # Speed of light - In km/s
    speed_c = ac.c.to(u.km/u.s).value
    ##
    ## Save to dictionary
    param_dict['sample_s'     ] = sample_s
    param_dict['sample_Mr'    ] = sample_Mr
    param_dict['volume_sample'] = volume_sample
    param_dict['vol_mr'       ] = vol_mr
    param_dict['cens'         ] = cens
    param_dict['sats'         ] = sats
    param_dict['speed_c'      ] = speed_c
    param_dict['catl_str_fig' ] = catl_str_fig

    return param_dict

def directory_skeleton(param_dict, proj_dict):
    """
    Creates the directory skeleton for the current project

    Parameters
    ----------
    param_dict : `dict`
        dictionary with `project` variables

    proj_dict : `dict`
        dictionary with info of the project that uses the
        `Data Science` Cookiecutter template.

    Returns
    ---------
    proj_dict : `dict`
        Dictionary with current and new paths to project directories
    """
    ## In here, you define the directories of your project
    #
    # Main output file for this script
    catl_output_dirpath = param_dict['ml_args'].catl_model_pred_plot_dir(
        check_exist=True)
    #
    # Saving to `proj_dict`
    proj_dict['catl_output_dirpath'] = catl_output_dirpath

    return proj_dict

## ----------------------------- Data Extraction ----------------------------#

def catl_extract_and_merge(param_dict, proj_dict, complete_groups=False,
    min_group_ngal=1):
    """
    Extracts the set of catalogues and merges them into a single catalogue.

    Parameters
    ----------
    param_dict : `dict`
        dictionary with `project` variables

    proj_dict : `dict`
        dictionary with info of the project that uses the
        `Data Science` Cookiecutter template.

    complete_groups : `bool`, optional  
        If True, it only looks at 'complete' galaxy groups, i.e. only
        when all of the galaxies of the group are present. This variable
        is set ot `False` by default.

    min_group_ngal : `int`, optional    
        Minimum number of galaxies in a galaxy group. This variable is set
        to ``1`` by default.

    Returns
    ---------
    catl_final_pd : `pandas.DataFrame`
        DataFrame containing the `merged` catalogue with info about the
        galaxy groups and their corresponding galaxies.
    """
    ## Catalogue with `predicted` masses
    catl_pred_pd = param_dict['ml_args'].catl_model_pred_file_extract(
                        return_pd=True,
                        return_arr=False,
                        remove_file=param_dict['remove_files'],
                        return_path=False)
    pred_cols_arr = ['index', 'M_h_pred', 'GG_M_group', 'GG_mdyn_rproj',
                        'GG_ngals']
    # Selecting only desired columns
    catl_pred_pd_mod = catl_pred_pd.loc[:, pred_cols_arr]
    # Renaming the 'index' column
    catl_pred_pd_mod.rename(columns={'index': 'gal_index'}, inplace=True)
    ## Catalogue with galaxy and group information
    catl_data_pd = cmcu.catl_sdss_merge(    0,
                                            catl_kind='data',
                                            catl_type=param_dict['catl_type'],
                                            sample_s=param_dict['sample_s'],
                                            halotype=param_dict['halotype'],
                                            clf_method=param_dict['clf_method'],
                                            dv=param_dict['dv'],
                                            hod_n=param_dict['hod_n'],
                                            clf_seed=param_dict['clf_seed'],
                                            perf_opt=param_dict['perf_opt'],
                                            print_filedir=False,
                                            return_memb_group=False)
    catl_data_cols_arr = ['groupid']
    # Selecting only desired columns
    catl_data_pd_mod = catl_data_pd.loc[:, catl_data_cols_arr]
    # Join both Datasets
    catl_main_pd = pd.merge(catl_pred_pd_mod,
                            catl_data_pd_mod,
                            how='inner',
                            left_on='gal_index',
                            right_index=True)
    # Removing galaxies smaller than `min_group_ngal`
    catl_main_pd = catl_main_pd.loc[catl_main_pd['GG_ngals'] >= min_group_ngal]
    # Checking groups that are complete
    if complete_groups:
        group_id_arr     = catl_main_pd['groupid'].values
        group_id_unq_arr = num.unique(group_id_arr)
        ngroups          = len(group_id_unq_arr)
        # Counting galaxies per group
        groupid_counts   = Counter(group_id_arr)
        group_ngals_catl = num.array([groupid_counts[xx] for xx in group_id_arr])
        catl_main_pd.loc[:, 'ngals_in_catl'] = group_ngals_catl
        # Checking if group is complete
        catl_main_pd.loc[:, 'group_complete'] = False
        catl_main_pd.loc[(catl_main_pd['ngals_in_catl'] == \
                            catl_main_pd['GG_ngals']), 'group_complete'] = True
        # Creating new DataFrame with only `complete groups`
        catl_main_pd_mod = catl_main_pd.loc[catl_main_pd['group_complete'] == True]
        # Dropping columns
        drop_cols = ['group_complete', 'ngals_in_catl']
        catl_main_pd_final = catl_main_pd_mod.drop(drop_cols, axis=1)
        catl_main_pd_final.reset_index(drop=True, inplace=True)

    if complete_groups:
        catl_final_pd = catl_main_pd_final
    else:
        catl_final_pd = catl_main_pd

    return catl_final_pd

## ----------------------------- Analysis -----------------------------------#

## Determining the bin of mass
def mass_bin_calculation(catl_final_pd, param_dict):
    """
    Determines the bin of mass for each galaxy.

    Parameters
    ------------
    catl_final_pd : `pandas.DataFrame`
        DataFrame containing the `merged` catalogue with info about the
        galaxy groups and their corresponding galaxies.

    param_dict : `dict`
        dictionary with `project` variables

    Returns
    ----------
    catl_final_pd : `pandas.DataFrame`
        DataFrame containing the `merged` catalogue with info about the
        galaxy groups and their corresponding galaxies + info on the group
        mass bin for each galaxy.

    param_dict : `dict`
        dictionary with `project` variables + added info on mass bins.
    """
    ## Mass Bins
    mass_cols       = ['GG_M_group', 'GG_mdyn_rproj']
    mass_min        = catl_final_pd.loc[:, mass_cols].min().values.min()
    mass_max        = catl_final_pd.loc[:, mass_cols].max().values.max()
    mass_bin_width  = param_dict['ml_args'].mass_bin_width
    mass_bins_edges = cstats.Bins_array_create( [mass_min, mass_max],
                                                base=mass_bin_width)
    mass_bins       = num.array([[mass_bins_edges[xx], mass_bins_edges[xx+1]] \
                                    for xx in range(len(mass_bins_edges) - 1)])
    # Mass bins labels
    mass_bins_labels = num.array(['[{0:.1f}, {1:.1f})'.format(xx[0], xx[1])
                            for xx in mass_bins])
    # Total number of bins
    n_mass_bins     = len(mass_bins)
    # Centers of the bins
    mass_bins_cent  = num.mean(mass_bins, axis=1)
    ## Determining the bin_value
    # HAM
    mass_bins_ham = num.digitize(   catl_final_pd['GG_M_group'],
                                    mass_bins_edges) - 1
    # Dynamical
    mass_bins_dyn = num.digitize(   catl_final_pd['GG_mdyn_rproj'],
                                    mass_bins_edges) - 1
    # Bin value label
    mass_bins_labels_ham = [mass_bins_labels[xx] for xx in mass_bins_ham]
    mass_bins_labels_dyn = [mass_bins_labels[xx] for xx in mass_bins_dyn]
    # Saving to DataFrame
    catl_final_pd.loc[:, 'HAM_bin_idx'] = mass_bins_ham
    catl_final_pd.loc[:, 'DYN_bin_idx'] = mass_bins_dyn
    catl_final_pd.loc[:, 'HAM_bin_lab'] = mass_bins_labels_ham
    catl_final_pd.loc[:, 'DYN_bin_lab'] = mass_bins_labels_dyn 
    # Saving group mass bins to dictionary `param_dict`
    param_dict['mass_bins_edges' ] = mass_bins_edges
    param_dict['mass_bins'       ] = mass_bins
    param_dict['mass_bins_cent'  ] = mass_bins_cent
    param_dict['n_mass_bins'     ] = n_mass_bins
    param_dict['mass_bins_labels'] = mass_bins_labels

    return catl_final_pd, param_dict

# Scatter of masses within group
def group_mass_scatter(catl_final_pd, param_dict):
    """
    Determines the scatter of predicted mass for distinct galaxy groups.

    Parameters
    ------------
    catl_final_pd : `pandas.DataFrame`
        DataFrame containing the `merged` catalogue with info about the
        galaxy groups and their corresponding galaxies.

    param_dict : `dict`
        dictionary with `project` variables

    Returns
    ----------
    catl_final_pd : `pandas.DataFrame`
        DataFrame containing the `merged` catalogue with info about the
        galaxy groups and their corresponding galaxies + info on the group
        mass bin for each galaxy + Info on the `normalized` predicted
        masses.
    """
    # Group array
    groupid_gals_arr = catl_final_pd['groupid'].values
    groupid_unq_arr  = num.unique(groupid_gals_arr)
    ngroups_unq      = len(groupid_unq_arr)
    ngals            = len(catl_final_pd)
    # Creating empty list of new values for the *normalized* masses
    mpred_norm_gals_arr = num.zeros(ngals) * num.nan
    # Looping over groups
    tqdm_msg = 'Normalizing predicted masses: '
    for kk, group_kk in enumerate(tqdm(groupid_unq_arr, desc=tqdm_msg)):
        # Galaxy indices
        gals_kk_pd = catl_final_pd.loc[catl_final_pd['groupid'] == group_kk]
        # ML-Predicted masses
        gals_pred_mass_kk = 10**gals_kk_pd['M_h_pred'].values
        # Galaxy indices
        gals_kk_idx = gals_kk_pd.index.values
        # Normalized HAM masses
        gals_mpred_norm_kk = gals_pred_mass_kk / num.mean(gals_pred_mass_kk)
        # Assigning to galaxies
        mpred_norm_gals_arr[gals_kk_idx] = gals_mpred_norm_kk
    #
    # Assigning it to DataFrame
    catl_final_pd.loc[:, 'mpred_norm'] = mpred_norm_gals_arr

    return catl_final_pd

## ----------------------------- Plotting -----------------------------------#

## Plotting the distributions of `predicted` masses

def group_mass_scatter_plot(catl_final_pd, param_dict,
    proj_dict, fig_fmt='pdf', figsize=(10,6), fig_number=1):
    """
    Plotting of the violinplot for the scatter in the 

    Parameters
    ------------
    catl_final_pd : `pandas.DataFrame`
        DataFrame containing the `merged` catalogue with info about the
        galaxy groups and their corresponding galaxies + info on the group
        mass bin for each galaxy + Info on the `normalized` predicted
        masses.

    fig_fmt : `str`, optional (default = 'pdf')
        extension used to save the figure

    figsize : `tuple`, optional
        Size of the output figure. This variable is set to `(12,15.5)` by
        default.

    fig_number : `int`, optional
        Number of figure in the workflow. This variable is set to `1`
        by default.
    """
    file_msg     = param_dict['Prog_msg']
    # Matplotlib option
    matplotlib.rcParams['axes.linewidth'] = 2.5
    matplotlib.rcParams['axes.edgecolor'] = 'black'
    # Constants
    label_size = 23
    ## Figure name
    fname = os.path.join(   proj_dict['catl_output_dirpath'],
                            'Fig_{0}_{1}_masses_comparison.{2}'.format(
                                fig_number,
                                param_dict['catl_str_fig'],
                                fig_fmt))
    # Figure contants
    ##
    ## Figure details
    plt.clf()
    plt.close()
    fig = plt.figure(figsize=figsize)
    ax1 = fig.add_subplot(121, facecolor='white')
    ax2 = fig.add_subplot(122, facecolor='white', sharey=ax1)
    # Deleting Y-label for 2nd axis
    plt.setp(ax2.get_yticklabels(), visible=False)
    # Labels
    ylabel    = r'\boldmath Normalized $\log \left[M_{pred}/ \right]$'
    ham_label = r'\boldmath$\log M_{\mathrm{HAM}}\left[h^{-1} M_{\odot}\right]$'
    dyn_label = r'\boldmath$\log M_{\mathrm{dyn}}\left[h^{-1} M_{\odot}\right]$'
    ## Violin plots
    # HAM
    sns.violinplot( x='HAM_bin_lab', y='mpred_norm',
                    inner='quart', data=catl_final_pd, ax=ax1)
    # DYN
    sns.violinplot( x='HAM_bin_lab', y='mpred_norm',
                    inner='quart', data=catl_final_pd, ax=ax2)
    # Adjusting spacing
    plt.subplots_adjust(hspace=0)
    # Axis labels
    ax1.set_xlabel(ham_label, fontsize=label_size)
    ax2.set_xlabel(dyn_label, fontsize=label_size)
    ax1.set_ylabel(ylabel   , fontsize=label_size)
    ##
    ## Saving figure
    if fig_fmt=='pdf':
        plt.savefig(fname, bbox_inches='tight')
    else:
        plt.savefig(fname, bbox_inches='tight', dpi=400)
    ##
    ##
    print('{0} Figure saved as: {1}'.format(file_msg, fname))
    plt.clf()
    plt.close()







## --------------------------- Main Function --------------------------------#
def main(args):
    """
    Script to produce catalogues with features and predicted masses for
    SDSS DR7. It uses the sets of already-trained ML algorithms and
    applies them to real data.
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
    ##
    ## -------- Main analysis -------- ##
    ##
    # Predicting masses
    catl_final_pd = catl_extract_and_merge(param_dict, proj_dict,
        complete_groups=True)
    # Group mass bins
    (   catl_final_pd,
        param_dict   ) = mass_bin_calculation(catl_final_pd, param_dict)
    # Scatter within groups at fixed mass
    catl_final_pd = group_mass_scatter(catl_final_pd, param_dict)
    # Plotting of the group mass scatter plot
    group_mass_scatter_plot(catl_final_pd, param_dict, proj_dict)



# Main function
if __name__=='__main__':
    ## Input arguments
    args = get_parser()
    # Main Function
    main(args)
