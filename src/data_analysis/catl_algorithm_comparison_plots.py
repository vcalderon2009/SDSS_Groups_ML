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
import matplotlib
matplotlib.use( 'Agg' )
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
plt.rc('text', usetex=True)
from astropy.visualization import astropy_mpl_style
plt.style.use(astropy_mpl_style)
import seaborn as sns
#sns.set()

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
from sklearn.utils import resample
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
    ## Which axes to plot
    parser.add_argument('-plot_opt',
                        dest='plot_opt',
                        help='Option for which variable to plot on x-axis',
                        type=str,
                        choices=['mgroup', 'mhalo'],
                        default='mhalo')
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
                                'ml_alg_comparison',
                                catl_prefix_path,
                                'sf_{0}'.format(param_dict['sample_frac']),
                                param_dict['pre_opt'])
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

## --------- Plotting Functions ------------##



# Fractional difference
def frac_diff_model(models_dict, param_dict, proj_dict, plot_opt='mhalo',
    arr_len=10, bin_statval='left', fig_fmt='pdf', figsize=(10, 8),
    fig_number=1):
    """
    Plots the fractional difference between `predicted` and `true`
    halo masses.

    Parameters
    -----------
    models_dict : `dict`
        Dictionary containing the results from the ML analysis.

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
    #
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
                            'Fig_{0}_{1}_{2}_frac_diff_predicted.pdf'.format(
                                fig_number,
                                param_dict['catl_str_fig'],
                                plot_opt))
    ##
    ## Paper Figure
    fname_paper = os.path.join( proj_dict['paper_fig_dir'],
                                'Figure_04.{0}'.format(fig_fmt))
    ## Algorithm names - Thought as indices for the plot
    ml_algs_names = num.sort(list(models_dict.keys()))
    n_ml_algs     = len(ml_algs_names)
    # Initializing dictionary that will contain the necessary information
    # on each model
    frac_diff_dict = {}
    ## Reading in arrays for different models
    for kk, model_kk in enumerate(ml_algs_names):
        # X and Y coordinates
        model_kk_data = models_dict[model_kk]
        ## X- and Y-axis
        # X-axis
        # if (plot_opt == 'mhalo'):
        #     model_kk_x    = model_kk_data['mhalo_true']
        # elif (plot_opt == 'mgroup'):
        #     model_kk_x    = model_kk_data['mgroup_arr']
        #
        # New X-axis
        model_kk_x = model_kk_data['mhalo_pred']
        # Y-axis
        model_kk_y = model_kk_data['frac_diff']
        # Calculating error in bins
        (   x_stat_arr,
            y_stat_arr,
            y_std_arr,
            y_std_err) = cstats.Stats_one_arr(  model_kk_x,
                                                model_kk_y,
                                                base=bin_width,
                                                arr_len=arr_len,
                                                bin_statval=bin_statval)
        # Saving to dictionary
        frac_diff_dict[model_kk]      = {}
        frac_diff_dict[model_kk]['x_val' ] = model_kk_x
        frac_diff_dict[model_kk]['y_val' ] = model_kk_y
        frac_diff_dict[model_kk]['x_stat'] = x_stat_arr
        frac_diff_dict[model_kk]['y_stat'] = y_stat_arr
        frac_diff_dict[model_kk]['y_err' ] = y_std_arr
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
    # if (plot_opt == 'mgroup'):
    #     ham_x = ham_pred
    #     dyn_x = dyn_pred
    # elif (plot_opt == 'mhalo'):
    #     ham_x = ham_true
    #     dyn_x = dyn_true
    ham_x = ham_pred
    dyn_x = dyn_pred
    ##
    ## Binning data
    # HAM
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
    # ML algorithms - names
    ml_algs_names_mod  = [xx.replace('_',' ').title() for xx in ml_algs_names]
    ml_algs_names_dict = dict(zip(ml_algs_names, ml_algs_names_mod))
    # Labels
    # X-label
    # if (plot_opt == 'mgroup'):
    #     xlabel = r'\boldmath$\log M_{group}\left[ h^{-1} M_{\odot}\right]$'
    # elif (plot_opt == 'mhalo'):
    #     xlabel = r'\boldmath$\log M_{halo,\textrm{true}}\left[ h^{-1} M_{\odot}\right]$'
    xlabel = r'\boldmath$\log M_{predicted}\left[ h^{-1} M_{\odot}\right]$'
    # Y-label
    ylabel = r'Frac. Difference \boldmath$[\%]$'
    ##
    plt.clf()
    plt.close()
    fig = plt.figure(figsize=figsize)
    ax1 = fig.add_subplot(111, facecolor='white')
    ## Color
    cm  = plt.cm.get_cmap('viridis')
    cm_arr = [cm(kk/float(n_ml_algs)) for kk in range(n_ml_algs)]
    ## Horizontal line
    ax1.axhline(y=0, color='black', linestyle='--', zorder=10)
    ##
    ## Plotttin ML relations
    for kk, model_kk in enumerate(ml_algs_names):
        ## ML algorithm name
        ml_alg_kk_name = model_kk.replace('_',' ').title()
        ## Stats
        x_stat = frac_diff_dict[model_kk]['x_stat']
        y_stat = frac_diff_dict[model_kk]['y_stat']
        y_err  = frac_diff_dict[model_kk]['y_err' ]
        ## Fill-between variables
        y1 = y_stat - y_err
        y2 = y_stat + y_err

        ## Plotting relation
        ax1.plot(   x_stat,
                    y_stat,
                    color=cm_arr[kk],
                    linestyle='-',
                    marker='o',
                    zorder=zorder_ml)
        ax1.fill_between(x_stat, y1, y2, color=cm_arr[kk], alpha=alpha,
                        label=ml_alg_kk_name, zorder=zorder_ml)
    ## HAM Masses
    ax1.plot(   x_stat_ham,
                y_stat_ham,
                color=plot_dict['color_ham'],
                linestyle='--',
                marker='o',
                zorder=zorder_ml)
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
                linestyle='--',
                marker='o',
                zorder=zorder_ml)
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
    xlim = (10, 15)
    ylim = (-20, 20)
    ax1.set_xlim(xlim)
    ax1.set_ylim(ylim)
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

# Ranking of each galaxy property for each different algorithm
def feature_ranking_ml_algs(models_dict, param_dict, proj_dict,
    fig_fmt='pdf', figsize=(15, 12), fig_number=2, stacked_opt=True,
    rank_opt='idx'):
    """
    Plots the `ranking` of each galaxy proeperty based on the different ML
    algorithms used.

    Parameters
    ------------
    models_dict : `dict`
        Dictionary containing the results from the ML analysis.

    param_dict : `dict`
        Dictionary with input parameters and values related to this project.

    proj_dict: python dictionary
        Dictionary with current and new paths to project directories

    fig_fmt : `str`, optional (default = 'pdf')
        extension used to save the figure

    figsize : `tuple`, optional
        Size of the output figure. This variable is set to `(12,15.5)` by
        default.

    fig_number : `int`, optional
        Number of figure in the workflow. This variable is set to `1`
        by default.

    stacked_opt : `bool`, optional
        If True, it stacks the bar plots. This variable is set to `True`
        by default.

    rank_opt : {'idx', 'perc'} `str`, optional
        Option for which type of `ranking` to show.

        Options:
            - 'idx' : Shows the ranking indices for each feature.
            - ' perc' : Shows the ranking percentage for each feature.
    """
    file_msg = param_dict['Prog_msg']
    ## Matplotlib option
    matplotlib.rcParams['axes.linewidth'] = 2.5
    matplotlib.rcParams['axes.edgecolor'] = 'black'
    # Figure name
    ##
    ## Figure name
    fname = os.path.join(   proj_dict['figure_dir'],
                            'Fig_{0}_{1}_feature_ranking_{2}.{3}'.format(
                                fig_number,
                                param_dict['catl_str_fig'],
                                rank_opt,
                                fig_fmt))
    ##
    ## Paper Figure
    fname_paper = os.path.join( proj_dict['paper_fig_dir'],
                                'Figure_05.{0}'.format(fig_fmt))
    #
    # Constants
    feat_cols_dict = param_dict['ml_args'].feat_cols_names_dict()
    plot_dict      = param_dict['plot_dict']
    # List of algorithms used
    skem_key_arr = num.sort(list(models_dict.keys()))
    # Removing `neural network` from list if needed
    if ('neural_network' in skem_key_arr):
        skem_key_arr = num.array([i for i in skem_key_arr if i != 'neural_network'])
    # Number of ML algorithms
    n_ml_algs = len(skem_key_arr)
    # Features used
    feat_arr = num.array(param_dict['ml_args']._feature_cols())
    # Numbre of features
    n_feat = len(feat_arr)
    # Initializing array
    feat_rank_arr = num.zeros((n_feat, n_ml_algs))
    ## Choosing which type of `ranking` to show
    if (rank_opt == 'perc'):
        rank_col  = 1
        rank_type = float
    elif (rank_opt == 'idx'):
        rank_col  = 2
        rank_type = int
    # Looping over ML algorithms
    for kk, skem_key in tqdm(enumerate(skem_key_arr)):
        # Reading in Data
        feat_imp_sort = models_dict[skem_key]['feat_imp_sort']
        # Converting to DataFrame
        feat_imp_sort_pd = pd.DataFrame(feat_imp_sort[:, rank_col].astype(rank_type),
                                index=feat_imp_sort[:, 0],
                                columns=[skem_key])
        if (kk == 0):
            feat_rank_pd = feat_imp_sort_pd
        else:
            feat_rank_pd = pd.merge(feat_rank_pd,
                                    feat_imp_sort_pd,
                                    left_index=True,
                                    right_index=True)
    #
    # Calculating total ranking
    feat_rank_pd.loc[:,'rank_sum'] = feat_rank_pd.sum(axis=1)
    ##
    ## Ordering by rank
    feat_rank_pd.sort_values('rank_sum', ascending=True, inplace=True)
    ##
    ## Renaming columns
    feat_rank_pd.rename(index=feat_cols_dict, inplace=True)
    ##
    ## Excluding `rank_sum` column
    feat_rank_col_exclude = feat_rank_pd.columns.difference(['rank_sum'])
    feat_rank_pd_mod      = feat_rank_pd.loc[:,feat_rank_col_exclude].copy()
    ## Renaming columns
    feat_rank_pd_mod_cols     = feat_rank_pd_mod.columns.values
    feat_rank_pd_mod_cols_mod = [xx.replace('_',' ').replace('_rank','').title() for xx in 
                                feat_rank_pd_mod_cols]
    feat_rank_pd_mod.rename(columns=dict(zip(   feat_rank_pd_mod_cols,
                                                feat_rank_pd_mod_cols_mod)),
                            inplace=True)
    ##
    ## Plotting details
    plt.clf()
    plt.close()
    fig = plt.figure(figsize=figsize)
    ax1 = fig.add_subplot(111, facecolor='white')
    # Axis labels
    if (rank_opt == 'idx'):
        xlabel = r'$\leftarrow \textrm{Importance ranking}$'
    elif (rank_opt == 'perc'):
        xlabel = r'$\rightarrow \textrm{Importance ranking}$'
    ax1.set_xlabel(xlabel, fontsize=plot_dict['size_label'])
    # Plotting
    # Width
    if stacked_opt:
        b = feat_rank_pd_mod.plot(  kind='barh',
                                stacked=stacked_opt,
                                ax=ax1,
                                legend=True)
    else:
        b = feat_rank_pd_mod.plot(  kind='barh',
                                stacked=stacked_opt,
                                ax=ax1,
                                legend=True,
                                width=0.5)
    b.tick_params(labelsize=25)
    ## Legend
    leg = ax1.legend(loc='upper right', numpoints=1, frameon=False,
        prop={'size':20})
    # leg.get_frame().set_facecolor('none')
    ## Ticks
    if (rank_opt == 'idx'):
        major_loc_val = 10
        minor_loc_val = 5
    elif (rank_opt == 'perc'):
        major_loc_val = 0.2
        minor_loc_val = 0.1
    ax_data_major_loc  = ticker.MultipleLocator(major_loc_val)
    ax_data_minor_loc  = ticker.MultipleLocator(minor_loc_val)
    ax1.xaxis.set_major_locator(ax_data_major_loc)
    ax1.xaxis.set_minor_locator(ax_data_minor_loc)
    # Inverting axis
    ax1.invert_yaxis()
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
    plt.clf()
    plt.close()

# Model Score - Different algorithms - Bar Chart
def model_score_chart_1d(models_dict, param_dict, proj_dict,
    fig_fmt='pdf', figsize=(10,8), fig_number=3, score_type='perc'):
    """
    Plots the overall `score` for each algorithm, and compares them to
    the more traditional group mass estimation techniques, i.e. HAM and 
    dynamical mass.

    Parameters
    ------------
    models_dict : `dict`
        Dictionary containing the results from the ML analysis.

    param_dict : `dict`
        Dictionary with input parameters and values related to this project.

    proj_dict: python dictionary
        Dictionary with current and new paths to project directories

    fig_fmt : `str`, optional (default = 'pdf')
        extension used to save the figure

    figsize : `tuple`, optional
        Size of the output figure. This variable is set to `(12,15.5)` by
        default.

    fig_number : `int`, optional
        Number of figure in the workflow. This variable is set to `1`
        by default.

    score_type : {'perc', 'threshold', 'r2'}, `str`, optional
        Type of scoring to plot. This variable is set to `perc` by default.

        Options:
            - 'perc' : Use percentage and rank-ordering of the values
            - 'threshold' : Score based on diffs of `threshold` or less from true value.
            - 'r2': R-squared statistic for error calcuation.
    """
    file_msg  = param_dict['Prog_msg']
    plot_dict = param_dict['plot_dict']
    ## Matplotlib option
    matplotlib.rcParams['axes.linewidth'] = 2.5
    matplotlib.rcParams['axes.edgecolor'] = 'black'
    # Score
    if (param_dict['score_method'] == 'model_score'):
        score_type = 'r2'
    #
    # Figure name
    fname = os.path.join(   proj_dict['figure_dir'],
                            'Fig_{0}_{1}_ml_algorithms_scores_{2}.{3}'.format(
                                fig_number,
                                param_dict['catl_str_fig'],
                                score_type,
                                fig_fmt))
    ##
    ## Paper Figure
    fname_paper = os.path.join( proj_dict['paper_fig_dir'],
                                'Figure_06.{0}'.format(fig_fmt))
    #
    # Algorithms names - Thought as indices for this plot
    ml_algs_names = num.sort(list(models_dict))
    #
    # Initializing DataFrame
    zero_arr = num.zeros(len(ml_algs_names))
    col_names = ['ML']
    ml_algs_pd = pd.DataFrame(dict(zip(col_names, [zero_arr.copy()
                        for x in range(len(col_names))])))
    #
    # Reading in data
    for kk, ml_kk in enumerate(ml_algs_names):
        # Reading data
        # Model dictionary
        ml_model_kk_dict = models_dict[ml_kk]
        # Score
        model_score = ml_model_kk_dict['score_{0}'.format(score_type)]
        # Assigning to DataFrame
        ml_algs_pd.loc[kk, 'ML'] = model_score
    #
    # -- Traditonal mehods
    # HAM
    (   ham_pred,
        ham_true,
        ham_score) = param_dict['ml_args'].extract_trad_masses(
                                                    mass_opt='ham',
                                                    return_score=True,
                                                    score_method=score_type)
    # DYNAMICAL
    (   dyn_pred,
        dyn_true,
        dyn_score) = param_dict['ml_args'].extract_trad_masses(
                                                    mass_opt='dyn',
                                                    return_score=True,
                                                    score_method=score_type)
    #
    # rename indices
    ml_algs_indices = [xx.replace('_', ' ').title() for xx in ml_algs_names]
    ml_algs_pd.rename(index=dict(zip(range(len(ml_algs_names)),
                            ml_algs_indices)),
                            inplace=True)
    ## Plotting
    plt.clf()
    plt.close()
    fig = plt.figure(figsize=figsize)
    ax1 = fig.add_subplot(111, facecolor='white')
    # Constants
    ml_algs_pd.plot(kind='barh',
                    stacked=False,
                    ax=ax1,
                    legend=True)
    # HAM and Dynamical masses - Lines
    ax1.axvline(    x=ham_score,
                    color=plot_dict['color_ham'],
                    label='HAM Mass')
    ax1.axvline(    x=dyn_score,
                    color=plot_dict['color_dyn'],
                    label='Dynamical mass')
    ## Ticks
    ax_data_minor_loc  = ticker.MultipleLocator(0.05)
    ax_data_major_loc  = ticker.MultipleLocator(0.1)
    ax1.xaxis.set_minor_locator(ax_data_minor_loc)
    ax1.xaxis.set_major_locator(ax_data_major_loc)
    ##
    ## Axis label
    if (score_type == 'perc'):
        xlabel = r'$1\sigma$ error in $\Delta \log M_{halo} [\mathrm{dex}]$'
    else:
        xlabel = 'Score'
    ax1.set_xlabel(xlabel)
    ## Legend
    leg = ax1.legend(loc='upper right', numpoints=1, frameon=False,
        prop={'size': 14})
    leg.get_frame().set_facecolor('none')
    ##
    ## Saving figure
    if fig_fmt == 'pdf':
        plt.savefig(fname, bbox_inches='tight')
        plt.savefig(fname_paper, bbox_inches='tight')
    else:
        plt.savefig(fname, bbox_inches='tight', dpi=400)
        plt.savefig(fname_paper, bbox_inches='tight', dpi=400)
    ##
    ##
    print('{0} Figure saved as: {1}'.format(file_msg, fname))
    plt.clf()
    plt.close()

# HAM, Dynamical, and ML masses vs `True` halo mass
def pred_masses_halo_mass(models_dict, param_dict, proj_dict,
    arr_len=10, bin_statval='left', fig_fmt='pdf', figsize=(25,5),
    fig_number=4):
    """
    Plots the `predicted` vs the `true` mass for each of the different
    algorithms.

    Parameters
    -------------
    models_dict : `dict`
        Dictionary containing the results from the ML analysis.

    param_dict : `dict`
        Dictionary with input parameters and values related to this project.

    proj_dict: python dictionary
        Dictionary with current and new paths to project directories
    
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
    file_msg     = param_dict['Prog_msg']
    ## Matplotlib option
    matplotlib.rcParams['axes.linewidth'] = 2.5
    matplotlib.rcParams['axes.edgecolor'] = 'black'
    # Constants
    cm           = plt.cm.get_cmap('viridis')
    plot_dict    = param_dict['plot_dict']
    alpha        = 0.2
    alpha_mass   = 0.2
    zorder_points= 1
    zorder_shade = zorder_points + 2
    zorder_line  = zorder_points + 3
    # zorder_mass  = 10
    markersize   = 1
    # zorder_shade = zorder_mass - 1
    # zorder_ml    = zorder_mass + 1
    bin_width    = param_dict['ml_args'].mass_bin_width
    #
    # Figure name
    fname = os.path.join(proj_dict['figure_dir'],
                'Fig_{0}_{1}_pred_true_masses.{2}'.format(
                    fig_number, param_dict['catl_str_fig'], fig_fmt))
    ##
    ## Paper Figure
    fname_paper = os.path.join( proj_dict['paper_fig_dir'],
                                'Figure_07.{0}'.format(fig_fmt))
    #
    # Algorithm names - thought as indices for the plot
    ml_algs_names = num.sort(list(models_dict.keys()))
    n_ml_algs     = len(ml_algs_names)
    # Initializing dictionary that will hold the necessary information
    # of each model
    pred_true_dict = {}
    # Looping over different models
    for zz, model_zz in tqdm(enumerate(ml_algs_names)):
        # ML dictionary
        model_zz_dict = models_dict[model_zz]
        # Coordinates for the x- and y-axes.
        model_zz_x = model_zz_dict['mhalo_pred']
        model_zz_y = model_zz_dict['mhalo_true']
        # Calculating statistics
        x_arr, y_arr, y_std_arr, y_std_err = cstats.Stats_one_arr(
                                                model_zz_x,
                                                model_zz_y,
                                                base=bin_width,
                                                arr_len=arr_len,
                                                bin_statval=bin_statval)
        # Saving to dictionary
        pred_true_dict[model_zz] = {}
        pred_true_dict[model_zz]['x_stat'] = x_arr
        pred_true_dict[model_zz]['y_stat'] = y_arr
        pred_true_dict[model_zz]['y_err' ] = y_std_arr
        pred_true_dict[model_zz]['pred'  ] = model_zz_x
        pred_true_dict[model_zz]['true'  ] = model_zz_y
    ##
    ## Obtaining HAM and Dynamical masses
    #
    # - HAM
    (   ham_pred,
        ham_true,
        ham_frac_diff) = param_dict['ml_args'].extract_trad_masses_alt(
                            mass_opt='ham', return_frac_diff=True)
    #
    # - Dynamical
    (   dyn_pred,
        dyn_true,
        dyn_frac_diff) = param_dict['ml_args'].extract_trad_masses_alt(
                                                    mass_opt='dyn',
                                                    return_frac_diff=True,
                                                    nlim_threshold=True,
                                                    nlim_min=4)
    #
    # Only choosing non-zero values and those larger than `10`
    dyn_pred_mask = dyn_pred >= 11.0
    dyn_pred      = dyn_pred[dyn_pred_mask]
    dyn_true      = dyn_true[dyn_pred_mask]
    dyn_frac_diff = dyn_frac_diff[dyn_pred_mask]
    #
    # Binning data for the different masses
    # - HAM mass
    (   x_stat_ham,
        y_stat_ham,
        y_std_ham ,
        y_std_err_ham) = cstats.Stats_one_arr(  ham_pred,
                                                ham_true,
                                                base=bin_width,
                                                arr_len=arr_len,
                                                bin_statval=bin_statval)
    # - Dynamical mass
    (   x_stat_dyn,
        y_stat_dyn,
        y_std_dyn,
        y_std_err_dyn) = cstats.Stats_one_arr(  dyn_pred,
                                                dyn_true,
                                                base=bin_width,
                                                arr_len=arr_len,
                                                bin_statval=bin_statval)
    ##
    ## Lower- and upper-limits for errors
    # - HAM
    y1_ham = y_stat_ham - y_std_ham
    y2_ham = y_stat_ham + y_std_ham
    # - Dynamical
    y1_dyn = y_stat_dyn - y_std_dyn
    y2_dyn = y_stat_dyn + y_std_dyn
    #
    # One-one line
    one_arr = num.linspace(0, 15, num=100)
    #
    # -- Figure details
    # ML algorithms - Names
    ml_algs_names_dict = {x: x.replace('_',' ').title() for x in ml_algs_names}
    # Axes labels
    # X-axis
    xlabel = r'\boldmath$\log M_{predicted}\left[h^{-1} M_{\odot}\right]$'
    # Y-axis
    ylabel = r'\boldmath$\log M_{\textrm{halo}}\left[h^{-1} M_{\odot}\right]$'
    # Initializing color schemes
    cm     = plt.cm.get_cmap('viridis')
    cm_arr = [cm(kk/float(n_ml_algs)) for kk in range(n_ml_algs)]
    # Initializing figure
    plt.clf()
    plt.close()
    fig, axes = plt.subplots(nrows=1, ncols=2 + n_ml_algs, sharex=True,
                    sharey=True, figsize=figsize)
    # Flattening out the axes
    axes = axes.flatten()
    # Looping over ML algorithms
    for kk, ml_kk in enumerate(ml_algs_names):
        # Axes being used
        ax = axes[kk]
        # Background
        ax.set_facecolor('white')
        # Model name
        ml_kk_name = ml_algs_names_dict[ml_kk]
        # Plotting x- and y-arrays
        x_stat = pred_true_dict[ml_kk]['x_stat']
        y_stat = pred_true_dict[ml_kk]['y_stat']
        y_err  = pred_true_dict[ml_kk]['y_err' ]
        pred   = pred_true_dict[ml_kk]['pred'  ]
        true   = pred_true_dict[ml_kk]['true'  ]
        # Fill-between variables
        y1 = y_stat - y_err
        y2 = y_stat + y_err
        #
        # - Plotting
        # Points
        ax.scatter( pred,
                    true,
                    c=cm_arr[kk],
                    marker='o',
                    s=1,
                    alpha=alpha,
                    zorder=3,
                    rasterized=True)
        # Fill-between
        ax.fill_between(x_stat,
                        y1, y2,
                        color=cm_arr[kk],
                        alpha=alpha,
                        label=ml_kk_name,
                        zorder=4)
        # Relation
        ax.plot(x_stat,
                y_stat,
                color=cm_arr[kk],
                linestyle='-',
                marker='o',
                markersize=markersize,
                zorder=5)
        ax.legend(loc='upper left', numpoints=1, frameon=False,
            prop={'size': 14})
    # HAM and Dynamical masses
    # -- HAM Mass
    # Points
    # Relation
    ax_ham = axes[n_ml_algs]
    ax_ham.set_facecolor('white')
    ax_ham.plot(x_stat_ham,
                y_stat_ham,
                color=plot_dict['color_ham'],
                linestyle='-',
                marker='o',
                zorder=zorder_line)
    # Fill-between
    ax_ham.fill_between(x_stat_ham,
                        y1_ham,
                        y2_ham,
                        color=plot_dict['color_ham'],
                        alpha=alpha,
                        label='HAM',
                        zorder=zorder_points)
    # -- DYNAMICAL Mass
    # Points
    # Relation
    ax_dyn = axes[n_ml_algs + 1]
    ax_dyn.set_facecolor('white')
    ax_dyn.plot(x_stat_dyn,
                y_stat_dyn,
                color=plot_dict['color_dyn'],
                linestyle='-',
                marker='o',
                zorder=zorder_line)
    # Fill-between
    ax_dyn.fill_between(x_stat_dyn,
                        y1_dyn,
                        y2_dyn,
                        color=plot_dict['color_dyn'],
                        alpha=alpha,
                        label='Dynamical',
                        zorder=zorder_shade)
    # --- One-One Array and other settings
    # Constants
    # Setting limits
    xlim = (10, 15)
    ylim = (10, 15)
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
    ## Reading in catalogue
    models_dict = param_dict['ml_args'].extract_catl_alg_comp_info()
    ##
    ## Feature keys
    param_dict['feat_cols_dict'] = param_dict['ml_args'].feat_cols_names_dict()
    ##
    ## Fractional difference of `predicted` and `truth`
    frac_diff_model(models_dict, param_dict, proj_dict,
            plot_opt=param_dict['plot_opt'])
    #
    # Feature ranking
    feature_ranking_ml_algs(models_dict, param_dict, proj_dict,
        rank_opt='perc')
    feature_ranking_ml_algs(models_dict, param_dict, proj_dict,
        rank_opt='idx')
    #
    # Model Score - Different algorithms - Bar Chart
    model_score_chart_1d(models_dict, param_dict, proj_dict)
    #
    # HAM, Dynamical, and ML masses vs `True` halo mass
    pred_masses_halo_mass(models_dict, param_dict, proj_dict)
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
