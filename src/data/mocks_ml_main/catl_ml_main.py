#! /usr/bin/env python
# -*- coding: utf-8 -*-

# Victor Calderon
# Created      : 03/21/2018
# Last Modified: 03/22/2018
# Vanderbilt University
from __future__ import print_function, division, absolute_import
__author__     =['Victor Calderon']
__copyright__  =["Copyright 2017 Victor Calderon, "]
__email__      =['victor.calderon@vanderbilt.edu']
__maintainer__ =['Victor Calderon']
"""
Trains a ML algorithm to predict certain group/galaxy properties
based on a training and validation set of galaxies.
"""
# Path to Custom Utilities folder
import os
import sys
import git

# Importing Modules
from cosmo_utils.utils import file_utils      as cfutils
from cosmo_utils.utils import file_readers    as cfreaders
from cosmo_utils.utils import work_paths      as cwpaths
from cosmo_utils.utils import stats_funcs     as cstats

import numpy as num
import math
import os
import sys
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
from tqdm import tqdm

# Extra-modules
from argparse import ArgumentParser
from argparse import HelpFormatter
from operator import attrgetter
from datetime import datetime
import subprocess
import requests
from multiprocessing import Pool, Process, cpu_count
import copy
from collections import Counter
import astropy.cosmology as astrocosmo
import astropy.constants as ac
import astropy.units     as u
import astropy.table     as astro_table
from   astropy.coordinates import SkyCoord
from glob import glob
# ML modules
import sklearn
import sklearn.metrics          as skmetrics
import sklearn.model_selection  as skms
import sklearn.ensemble         as skem
import sklearn.neural_network   as skneuro
import sklearn.preprocessing    as skpre
import xgboost
import scipy

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
                    Trains ML algorithms on the dataset consisting of 
                    group and galaxy properties.
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
                        choices=range(0,10),
                        metavar='[0-10]',
                        default=0)
    ## Type of dark matter halo to use in the simulation
    parser.add_argument('-halotype',
                        dest='halotype',
                        help='Type of the DM halo.',
                        type=str,
                        choices=['so','fof'],
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
                        choices=[1,2,3],
                        default=3)
    ## Random Seed for CLF
    parser.add_argument('-clf_seed',
                        dest='clf_seed',
                        help='Random seed to be used for CLF',
                        type=int,
                        metavar='[0-4294967295]',
                        default=0)
    ## Luminosity sample to analyze
    parser.add_argument('-sample',
                        dest='sample',
                        help='SDSS Luminosity sample to analyze',
                        type=str,
                        choices=['all', '19','20','21'],
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
                        choices=range(2,1000),
                        metavar='[1-1000]',
                        default=2)
    ## Fraction of the sample to use
    parser.add_argument('-sample_frac',
                        dest='sample_frac',
                        help='fraction of the total dataset to use',
                        type=float,
                        default=0.01)
    ## `Perfect Catalogue` Option
    parser.add_argument('-perf',
                        dest='perf_opt',
                        help='Option for using a `Perfect` catalogue',
                        type=_str2bool,
                        default=False)
    ## Testing size for ML
    parser.add_argument('-test_size',
                        dest='test_size',
                        help='Percentage size of the catalogue used for testing',
                        type=_check_pos_val,
                        default=0.25)
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
                        help='Number of hidden layers to use for neural network',
                        type=int,
                        default=1000)
    ## Total number of properties to predict. Default = 1
    parser.add_argument('-n_predict',
                        dest='n_predict',
                        help="""
                        Number of properties to predict. Default = 1""",
                        type=int,
                        choices=range(1,4),
                        default=1)
    ## Option for Shuffling dataset when separing 
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
                        choices=['min_max','standard','normalize', 'no', 'all'],
                        default='standard')
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
    ## Preprocessing Option
    parser.add_argument('-remove',
                        dest='remove_files',
                        help='Delete HMF ',
                        type=_str2bool,
                        default=False)
    ## Verbose
    parser.add_argument('-v','--verbose',
                        dest='verbose',
                        help='Option to print out project parameters',
                        type=_str2bool,
                        default=False)
    ## CPU Counts
    parser.add_argument('-cpu',
                        dest='cpu_frac',
                        help='Fraction of total number of CPUs to use',
                        type=float,
                        default=0.75)
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
    ##
    ## Testing if `wget` exists in the system
    if is_tool('wget'):
        pass
    else:
        msg = '{0} You need to have `wget` installed in your system to run '
        msg += 'this script. You can download the entire dataset at {1}.\n\t\t'
        msg += 'Exiting....'
        msg = msg.format(param_dict['Prog_msg'], param_dict['url_catl'])
        raise ValueError(msg)
    ##
    ## Checking that Esmeralda is not ran when doing 'SO' halos
    if (param_dict['halotype'] == 'so') and (param_dict['sample'] == 20):
        msg = '{0} The `halotype`==`so` and `sample`==`20` are no compatible '
        msg += 'input parameters.\n\t\t'
        msg += 'Exiting...'
        msg = msg.format(param_dict['Prog_msg'])
        raise ValueError(msg)
    ##
    ## Checking that `hod_model_n` is set to zero for FoF-Halos
    if (param_dict['halotype'] == 'fof') and (param_dict['hod_n'] != 0):
        msg = '{0} The `halotype`==`{1}` and `hod_n`==`{2}` are no compatible '
        msg += 'input parameters.\n\t\t'
        msg += 'Exiting...'
        msg = msg.format(   param_dict['Prog_msg'],
                            param_dict['halotype'],
                            param_dict['hod_n'])
        raise ValueError(msg)
    ##
    ## Checking that `kf_splits` is larger than `2`
    if (param_dict['kf_splits'] < 2):
        msg  = '{0} The value for `kf_splits` ({1}) must be LARGER than `2`'
        msg += 'Exiting...'
        msg  = msg.format(  param_dict['Prog_msg' ],
                            param_dict['kf_splits'])
        raise ValueError(msg)
    ##
    ## Checking that `kf_splits` is larger than `2`
    if (param_dict['n_predict'] < 1):
        msg  = '{0} The value for `n_predict` ({1}) must be LARGER than `1`'
        msg += 'Exiting...'
        msg  = msg.format(  param_dict['Prog_msg' ],
                            param_dict['n_predict'])
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
    sample_s = str(param_dict['sample'])
    ### Sample - Mr
    sample_Mr = 'Mr{0}'.format(param_dict['sample'])
    ##
    ## Choice of Centrals and Satellites
    cens = int(1)
    sats = int(0)
    ## Other constants
    # Speed of light - In km/s
    speed_c = ac.c.to(u.km/u.s).value
    ##
    ## Catalogue string
    catl_str_arr = [sample_Mr,
                    param_dict['hod_n'], 
                    param_dict['clf_method'],
                    param_dict['cosmo_choice'],
                    param_dict['nmin'],
                    param_dict['halotype'], 
                    param_dict['perf_opt']]
    catl_str     = '{0}_hodn_{1}_clf_{2}_cosmo_{3}_nmin_{4}_halotype_{5}_perf_'
    catl_str    += '{6}'
    catl_str     = catl_str.format(*catl_str_arr)
    ## Number of CPU's to use
    cpu_number = int(cpu_count() * param_dict['cpu_frac'])
    ##
    ## Dictionary of ML Regressors
    skem_dict = sklearns_models(param_dict, cpu_number)
    ##
    ## Saving to `param_dict`
    param_dict['sample_s'    ] = sample_s
    param_dict['sample_Mr'   ] = sample_Mr
    param_dict['cens'        ] = cens
    param_dict['sats'        ] = sats
    param_dict['speed_c'     ] = speed_c
    param_dict['catl_str'    ] = catl_str
    param_dict['skem_dict'   ] = skem_dict
    param_dict['cpu_number'  ] = cpu_number

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
    ## Main Directories
    proj_str = os.path.join('halos_{0}'.format(param_dict['halotype']),
                            'hod_model_{0}'.format(param_dict['hod_n']),
                            'clf_seed_{0}'.format(param_dict['clf_seed']),
                            'clf_method_{0}'.format(param_dict['clf_method']),
                            param_dict['catl_type'],
                            param_dict['sample_Mr'])
    # External Directory
    ext_dir        = os.path.join( proj_dict['data_dir'], 'external')
    # Processes Directory
    processed_dir  = os.path.join( proj_dict['data_dir'], 'processed')
    # Interim Directory
    int_dir        = os.path.join( proj_dict['data_dir'], 'interim')
    # Raw Directory
    raw_dir        = os.path.join( proj_dict['data_dir'], 'raw')
    ## Training and Testing directories
    # Training and testing
    test_train_dir   = os.path.join(  int_dir,
                                    'training_testing',
                                    proj_str)
    ## Output file for all catalogues
    catl_outdir    = os.path.join(  proj_dict['data_dir'],
                                    'processed',
                                    'SDSS',
                                    'mocks',
                                    proj_str,
                                    'merged_vac')
    ## Creating output folders for the catalogues
    merged_gal_dir          = os.path.join(catl_outdir, 'merged_vac'         )
    merged_gal_perf_dir     = os.path.join(catl_outdir, 'merged_vac_perf'    )
    merged_gal_all_dir      = os.path.join(catl_outdir, 'merged_vac_all'     )
    merged_gal_perf_all_dir = os.path.join(catl_outdir, 'merged_vac_perf_all')
    ##
    ## Creating Directories
    catl_dir_arr = [catl_outdir, merged_gal_dir, merged_gal_perf_dir,
                    merged_gal_all_dir, merged_gal_perf_all_dir,
                    ext_dir, processed_dir, int_dir, raw_dir]
    for catl_ii in catl_dir_arr:
        try:
            assert(os.path.exists(catl_ii))
        except:
            msg = '{0} `{1}` does not exist! Exiting'.format(
                param_dict['Prog_msg'], catl_ii)
            raise ValueError(msg)
    ## Creating directories
    cfutils.Path_Folder(test_train_dir)
    ##
    ## Adding to `proj_dict`
    proj_dict['ext_dir'                ] = ext_dir
    proj_dict['processed_dir'          ] = processed_dir
    proj_dict['int_dir'                ] = int_dir
    proj_dict['raw_dir'                ] = raw_dir
    proj_dict['catl_outdir'            ] = catl_outdir
    proj_dict['merged_gal_dir'         ] = merged_gal_dir
    proj_dict['merged_gal_perf_dir'    ] = merged_gal_perf_dir
    proj_dict['merged_gal_all_dir'     ] = merged_gal_all_dir
    proj_dict['merged_gal_perf_all_dir'] = merged_gal_perf_all_dir
    proj_dict['test_train_dir'         ] = test_train_dir

    return proj_dict

## --------- Scoring tools ------------##

# Scoring methods
def scoring_methods(feat_arr, truth_arr, model_used,
    score_method='perc', threshold=0.1, perc_val=0.9):
    """
    Determines the overall score for given arrays `pred_arr` and `truth_arr`.

    Parameters
    ------------
    feat_arr: numpy.ndarray
        array consisting of the `features` values

    truth: numpy.ndarray
        array consisting of the `truth` values

    model_used: sklearn object
        model used to estimate the score if `score_method == 'model_score'`
    
    score_method: string, optional (default = 'perc')
        Type of scoring to use when determining how well an algorithm 
        performs.
        Options:
            - 'perc': Use percentage and rank-ordering of the values
            - 'threshold': Score based on difference of `threshold` or less 
                            from the true value
            - `model_score`: Out-of-the-box method from sklearn to determine 
                    the success of a ML model.

    threshold: float, optional (default = 0.1)
        value to use when calculating the error within `threshold` value 
        from the truth

    perc_val: float, optional (default = 0.9)
        value used when determining score within some `perc_val` percentile
        value from [0,1]
    
    Returns
    ------------
    method_score: float
        overall score from `pred_arr` to predict `truth_arr`
    """
    ## Converting method
    ## Choosing method
    # `Percentile` method
    if score_method == 'perc':
        pred_arr     = model_used.predict(feat_arr)
        pred_err     = num.abs(pred_arr - truth_arr)
        method_score = scipy.stats.scoreatpercentile(pred_err, 100.*perc_val)
    ## Threshold method
    if score_method == 'threshold':
        pred_arr     = model_used.predict(feat_arr)
        pred_err     = num.abs(pred_arr - truth_arr)
        pred_thresh  = len(pred_err[pred_err <= threshold])
        method_score = pred_thresh / len(pred_arr)
    ## Model Score
    if score_method == 'model_score':
        method_score = model_used.score(feat_arr, truth_arr)

    return method_score



## --------- Preparing Data ------------##

# Separating data into `training` and `testing` dataset
def training_testing_data(param_dict, proj_dict, test_size=0.25,
    random_state=0, shuffle_opt=True, dropna_opt=True, sample_frac=0.1,
    ext='hdf5'):
    """
    Reads in the catalogue, and separates data into a `training` and 
    `testing` datasets.

    Parameters
    -------------
    param_dict: python dictionary
        dictionary with `project` variables

    proj_dict: python dictionary
        dictionary with info of the project that uses the
        `Data Science` Cookiecutter template.

    test_size: float, optional (default = 0.25)
        percentage of the catalogue that represents the `test` size of the 
        'testing' dataset

    random_state: int, optional (default = 0)
        random state used for when selecting `training` and `testing`
        dataset. If set, it will always have the same seed `random_state`.

    shuffle_opt: boolean, optional (default = True)
        Whether or not to shuffle the data before splitting. 
        If shuffle=False then stratify must be None.

    dropna_opt: boolean, optional (default = True)
        Option for dropping 'NaN' from the catalogue

    sample_frac: float, optional (default = 0.10)
        fraction of the total dataset to use

    ext: string, optional (default = 'hdf5')
        file extension of the `merged` catalogue

    Returns
    -------------
    train_dict: python dictionary
        dictionary containing the 'training' data from the catalogue

    test_dict: python dictionary
        dictionary containing the 'testing' data from the catalogue

    param_dict: python dictionary
        dictionary with `project` variables + list of `predicted` and `features`

    """
    ## Read in catalogue
    catl_arr = glob('{0}/*{1}*.{2}'.format( proj_dict['merged_gal_all_dir'],
                                            param_dict['catl_str'],
                                            ext))
    ## Checking if file exists
    try:
        assert(len(catl_arr) > 0)
    except:
        msg = '{0} The length of `catl_arr` ({1}) must be larger than `0`. '
        msg += 'Exiting ...'
        msg = msg.format(param_dict['Prog_msg'], len(catl_arr))
        raise ValueError(msg)
    ## Reading in catalogue
    catl_pd_tot  = cfreaders.read_hdf5_file_to_pandas_DF(catl_arr[0])
    ## Selecting only a fraction of the dataset
    catl_pd     = catl_pd_tot.sample(frac=sample_frac, random_state=random_state)
    catl_pd_tot = None
    ## Dropping `groupid`
    catl_drop_arr = ['groupid', 'halo_rvir', 'galtype', 'halo_ngal']
    catl_pd       = catl_pd.drop(catl_drop_arr, axis=1)
    ## Dropping NaN's
    if dropna_opt:
        catl_pd.dropna(how='any', inplace=True)
    ##
    ##  Temporarily fixing `GG_mdyn_rproj`
    catl_pd.loc[:,'GG_mdyn_rproj'] /= 0.96
    ##
    ## Separing `features` and `predicted values`
    catl_cols      = catl_pd.columns.values
    # predicted_cols = ['g_brightest', 'M_r', 'dist_centre', 'galtype']
    if param_dict['n_predict'] == 1:
        predicted_cols = ['M_h']
    elif param_dict['n_predict'] == 3:
        predicted_cols = ['M_h', 'galtype']
    # features_cols  = [s for s in catl_cols if s not in predicted_cols]
    features_cols = [   'M_r','GG_mr_brightest','g_r','GG_rproj','GG_sigma_v',
                        'GG_M_r','GG_ngals','GG_M_group','GG_mdyn_rproj']
    # features_cols = [   'GG_mdyn_rproj','GG_M_group']
    # Creating new DataFrames
    pred_arr = catl_pd.loc[:,predicted_cols].values
    feat_arr = catl_pd.loc[:,features_cols ].values
    ## Reshaping array
    if len(predicted_cols) == 1:
        pred_arr = pred_arr.reshape(len(pred_arr),)
    if len(features_cols) == 1:
        feat_arr = feat_arr.reshape(len(feat_arr),)
    ##
    ## Rescaling training and testing dataset
    feat_arr_scaled = data_preprocessing(feat_arr, param_dict, 
                            pre_opt=param_dict['pre_opt'])
    ## Training and Testing Dataset
    #  Scaled
    (   X_train, X_test,
        Y_train, Y_test) = skms.train_test_split(   feat_arr_scaled,
                                                    pred_arr,
                                                    test_size=test_size,
                                                    shuffle=shuffle_opt,
                                                    random_state=random_state)
    # Not Scaled
    (   X_train_ns, X_test_ns,
        Y_train_ns, Y_test_ns) = skms.train_test_split( feat_arr,
                                                        pred_arr,
                                                        test_size=test_size,
                                                        shuffle=shuffle_opt,
                                                        random_state=random_state)
    ##
    ## Assigning `training` and `testing` datasets to dictionary
    train_dict = {  'X_train': X_train, 'Y_train': Y_train,
                    'X_train_ns':X_train_ns, 'Y_train_ns':Y_train_ns}
    test_dict  = {'X_test' : X_test , 'Y_test' : Y_test,
                    'X_test_ns':X_test_ns, 'Y_test_ns':Y_test_ns}
    ## Adding lists to `param_dict`
    param_dict['predicted_cols' ] = predicted_cols
    param_dict['features_cols'  ] = features_cols
    param_dict['feat_arr_scaled'] = feat_arr_scaled
    param_dict['feat_arr'       ] = feat_arr

    return train_dict, test_dict, param_dict

# Different types of Regressors
def sklearns_models(param_dict, cpu_number):
    """
    Returns a set of Regressors used by Scikit-Learn

    Parameters
    -----------
    param_dict: python dictionary
        dictionary with input parameters and values

    cpu_number: int
        number of CPUs to use for the training
    
    Returns
    ---------
    skem_dict: python dicitonary
        Dictioanry with a set of regressors uninitialized
    """
    skem_dict = {}
    # Random Forest
    skem_dict['random_forest'    ] = skem.RandomForestRegressor(
                                        n_jobs=cpu_number,
                                        random_state=param_dict['seed'])
    # Gradient Boosting Method
    # skem_dict['gradient_boosting'] = skem.GradientBoostingRegressor(
    #                                     random_state=param_dict['seed'])
    # XGBoost Regressor
    skem_dict['XGBoost'          ] = xgboost.XGBRegressor(
                                        n_jobs=cpu_number,
                                        random_state=param_dict['seed'])
    # AdaBoost Regressor
    # skem_dict['AdaBoost'         ] = skem.AdaBoostRegressor(
    #                                     random_state=param_dict['seed'])
    # Neural network
    skem_dict['neural_network'   ] = skneuro.MLPRegressor(
                                        random_state=param_dict['seed'],
                                        hidden_layer_sizes=(
                                            param_dict['hidden_layers'],
                                            param_dict['hidden_layers'],
                                            param_dict['hidden_layers']),
                                        solver='adam')

    return skem_dict

## Data Preprocessing
def data_preprocessing(feat_arr, param_dict, pre_opt='min_max'):
    """
    Preprocess the data used, in order to clean and make the data more 
    suitable for the machine learning algorithms

    Parameters
    -------------
    feat_arr: numpy.ndarray
        array of feature values. Used for training the ML algorithm

    param_dict: python dictionary
        dictionary with input parameters and values

    pre_opt: string, optional (default = 'min_max')
        Option for which preprocessing of the data to use
        Options:
            - 'min_max':
            - 'standard':
            - 'normalize':
            - 'no':

    Returns
    -------------
    feat_arr_scaled: numpy.ndarray
        rescaled version of array of feature values
    """
    ## Min-Max scales (Rescale) Data
    if pre_opt == 'min_max':
        # Scaler
        scaler = skpre.MinMaxScaler(feature_range=(0,1))
        # Rescaling
        feat_arr_scaled = scaler.fit_transform(feat_arr)
    ## Standardize Data
    if pre_opt == 'standard':
        # Scaler
        scaler = skpre.StandardScaler().fit(feat_arr)
        # Rescaling
        feat_arr_scaled = scaler.transform(feat_arr)
    ## Normalize Data
    if pre_opt == 'normalize':
        # Scaler
        scaler = skpre.Normalizer().fit(feat_arr)
        # Rescaling
        feat_arr_scaled = scaler.transform(feat_arr)
    ## No Preprocessing
    if pre_opt == 'no':
        feat_arr_scaled = feat_arr

    return feat_arr_scaled

## --------- Training and Testing Function ------------##

## Algorithm Score, Testing, and other metrics
def model_score_general(train_dict, test_dict, skem_key, param_dict):
    """
    Computes different statistics for determining if `model` is good.
    It calculates:
        - General Score
        - K-fold scores
        - Feature importance for `General` and `K-fold`

    Parameters
    -----------
    train_dict: python dictionary
        dictionary containing the 'training' data from the catalogue

    test_dict: python dictionary
        dictionary containing the 'testing' data from the catalogue

    skem_key: string
        key of the Regressor being used. Taken from 'skem_dict'

    Returns
    -----------
    model_dict: python dictionary
        dictionary containing metrics that evaluate the model
        Keys:
            - 'model_score_tot'  : total score of the model after 1 run
            - 'kf_scores'        : score for each of the  K-folds
            - 'feat_imp_gen_sort': feature importance for 1 run
            - 'feat_imp_kf_sort' : mean feature importance for K-folds
    
    """
    ## Constants
    kf_splits  = param_dict['kf_splits' ]
    cpu_number = param_dict['cpu_number']
    ## Training and Testing sets
    X_train    = train_dict['X_train']
    Y_train    = train_dict['Y_train']
    X_test     = test_dict ['X_test' ]
    Y_test     = test_dict ['Y_test' ]
    X_train_ns = train_dict['X_train_ns']
    Y_train_ns = train_dict['Y_train_ns']
    X_test_ns  = test_dict ['X_test_ns' ]
    Y_test_ns  = test_dict ['Y_test_ns' ]
    feat_cols  = num.array(param_dict['features_cols'])
    n_feat     = len(feat_cols)
    # Scoring defaults
    threshold = param_dict['threshold']
    perc_val  = param_dict['perc_val' ]
    ## ------- General Model ------- ##
    ##
    ## -- General Model
    model_gen = sklearn.base.clone(param_dict['skem_dict'][skem_key])
    ##
    ## -- Training
    model_gen.fit(X_train, Y_train)
    ###
    ### -- Score - Total and K-fold -- ###
    # Normal score
    model_score_tot = scoring_methods(  X_test,
                                        Y_test,
                                        model_gen,
                                        score_method=param_dict['score_method'],
                                        threshold=threshold,
                                        perc_val=perc_val)
    # K-fold
    # Choosing which method to use
    kf_models_arr      = [[] for kk in range(kf_splits)]
    kf_scores          = num.zeros(kf_splits)
    kdf_features_imp   = num.zeros((kf_splits, n_feat))
    kf_obj             = skms.KFold(n_splits=kf_splits,
                                    shuffle=param_dict['shuffle_opt'],
                                    random_state=param_dict['seed'])
    tqdm_desc   = 'General and K-Fold Score: '
    tqdm_kf_obj = tqdm(enumerate(kf_obj.split(X_train)), desc=tqdm_desc)
    for kk, (train_idx_kk, test_idx_kk) in tqdm_kf_obj:
        ## Determining Training and Testing
        X_train_kk, X_test_kk = X_train[train_idx_kk], X_train[test_idx_kk]
        Y_train_kk, Y_test_kk = Y_train[train_idx_kk], Y_train[test_idx_kk]
        ## Fitting Model
        model_kf = sklearn.base.clone(param_dict['skem_dict'][skem_key])
        model_kf.fit(X_train_kk, Y_train_kk)
        ## Calculating Score
        kf_kk_score          = scoring_methods( X_test_kk,
                                                Y_test_kk,
                                                model_kf,
                                                score_method=param_dict['score_method'],
                                                threshold=threshold,
                                                perc_val=perc_val)
        ## Saving to array
        kf_scores[kk]        = kf_kk_score
        ##
        if skem_key != 'neural_network':
            ## Feature importances
            kdf_features_imp[kk] = model_kf.feature_importances_.astype(float)
        ## Saving Model to array
        kf_models_arr   [kk] = model_kf
    ##
    ## Score for HAM and Dynamical mass
    ## Masses
    # HAM
    ham_idx        = num.where(feat_cols == 'GG_M_group')[0]
    pred_ham       = X_train_ns.T[ham_idx].flatten()
    true_halo_ham  = Y_train_ns.flatten()
    ## Dynamical
    dyn_idx        = num.where(feat_cols == 'GG_mdyn_rproj')[0]
    pred_dyn       = X_train_ns.T[dyn_idx].flatten()
    true_halo_dyn  = Y_train_ns.flatten()
    # Avoid zeros
    pred_dyn_idx      = num.where(pred_dyn != 0.)[0]
    pred_dyn_mod      = pred_dyn[pred_dyn_idx]
    true_halo_dyn_mod = true_halo_dyn[pred_dyn_idx]
    # `Score`
    if param_dict['score_method'] == 'perc':
        # HAM
        pred_ham_error = num.abs(pred_ham - true_halo_ham)
        score_ham      = scipy.stats.scoreatpercentile( pred_ham_error,
                                                        100*perc_val)
        # Dynamical
        pred_dyn_error = num.abs(pred_dyn_mod - true_halo_dyn_mod)
        score_dyn      = scipy.stats.scoreatpercentile( pred_dyn_error,
                                                        100*perc_val)
    # Threshold
    if param_dict['score_method'] == 'threshold':
        # HAM
        pred_ham_error = num.abs(pred_ham - true_halo_ham)
        thresh_ham     = len(pred_ham_error[pred_ham_error <= threshold])
        score_ham      = thresh_ham / len(pred_ham)
        # Dynamical
        pred_dyn_error = num.abs(pred_dyn_mod - true_halo_dyn_mod)
        thresh_dyn     = len(pred_dyn_error[pred_dyn_error <= threshold])
        score_dyn      = thresh_dyn / len(pred_dyn_mod)
    # Model Score
    if param_dict['score_method'] == 'model_score':
        # HAM
        score_ham = skmetrics.r2_score(true_halo_ham, pred_ham)
        # Dynamical
        score_dyn = skmetrics.r2_score(true_halo_dyn_mod, pred_dyn_mod)
    ##
    ##
    ## ------- True and Predicted values ------- ##
    #  `General` case
    model_gen_prediction_arr = model_gen.predict(X_test)
    if param_dict['n_predict'] == 1:
        model_gen_frac_diff_arr  = 100*(model_gen_prediction_arr - Y_test)/Y_test
    else:
        model_gen_frac_diff_arr  = 100*(model_gen_prediction_arr - Y_test)/Y_test
    model_gen_truth_arr      = Y_test.copy()
    #  `K-fold` case
    model_kf_prediction_arr  = [[] for kk in range(kf_splits)]
    model_kf_frac_diff_arr   = [[] for kk in range(kf_splits)]
    for kk in range(kf_splits):
        kf_prediction_kk = kf_models_arr[kk].predict(X_test)
        kf_frac_diff_kk  = 100.*(kf_prediction_kk - Y_test)/Y_test
        ## Saving to Arrays
        model_kf_prediction_arr[kk] = kf_prediction_kk
        model_kf_frac_diff_arr [kk] = kf_frac_diff_kk
    #
    # Fractional difference for HAM and dynamical
    frac_diff_ham = 100*(pred_ham - true_halo_ham)/true_halo_ham
    frac_diff_dyn = 100*(pred_dyn_mod - true_halo_dyn_mod)/true_halo_dyn_mod
    ##
    ## ------- Feature Importance ------- ##
    ##
    if skem_key != 'neural_network':
        ## Feature Importance - Sorted from highest to lowest
        #  -- General
        feat_imp_gen          = num.vstack(zip( feat_cols,
                                                model_gen.feature_importances_))
        feat_imp_gen_sort_idx = num.argsort(feat_imp_gen[:,1])[::-1]
        feat_imp_gen_sort     = feat_imp_gen[feat_imp_gen_sort_idx]
        #  -- K-folds
        feat_imp_kf_mean     = num.mean(kdf_features_imp.T, axis=1)
        feat_imp_kf          = num.vstack(zip(feat_cols, feat_imp_kf_mean))
        feat_imp_kf_sort_idx = num.argsort(feat_imp_kf[:,1])[::-1]
        feat_imp_kf_sort     = feat_imp_kf[feat_imp_kf_sort_idx]
        ##
        ## ------- Scores after evaluating with different features ------- ##
        ##
        # General and K-folds
        feat_score_gen_arr = num.zeros(n_feat)
        feat_score_kf_arr  = num.zeros(n_feat)
        # Looping over features
        tqdm_desc   = 'Scores for different Feature Importances: '
        tqdm_kf_obj = tqdm(range(n_feat), desc=tqdm_desc)
        for kk in tqdm_kf_obj:
            ## ---- General ---- ##
            # Initializing model
            model_gen_kk = sklearn.base.clone(param_dict['skem_dict'][skem_key])
            # Training model
            X_train_gen_kk = X_train.T[feat_imp_gen_sort_idx[0:kk+1]].T
            X_test_gen_kk  = X_test.T [feat_imp_gen_sort_idx[0:kk+1]].T
            model_gen_kk.fit(X_train_gen_kk, Y_train)
            # Getting Score
            model_gen_kk_score = scoring_methods(   X_test_gen_kk,
                                                    Y_test,
                                                    model_gen_kk,
                                                    score_method=param_dict['score_method'],
                                                    threshold=threshold,
                                                    perc_val=perc_val)
            ##
            ## ---- K-fold ---- ##
            model_kf_kk = sklearn.base.clone(param_dict['skem_dict'][skem_key])
            # Training model
            X_train_kf_kk = X_train.T[feat_imp_kf_sort_idx[0:kk+1]].T
            X_test_kf_kk  = X_test.T [feat_imp_kf_sort_idx[0:kk+1]].T
            model_kf_kk.fit(X_train_kf_kk, Y_train)
            # Getting Score
            model_kf_kk_score = scoring_methods(    X_test_kf_kk,
                                                    Y_test,
                                                    model_kf_kk,
                                                    score_method=param_dict['score_method'],
                                                    threshold=threshold,
                                                    perc_val=perc_val)
            ##
            ## ---- Saving to array ---- ##
            feat_score_gen_arr[kk] = model_gen_kk_score
            feat_score_kf_arr [kk] = model_kf_kk_score
        # Joining Keys
        feat_score_gen_cumu = num.vstack(zip(   feat_imp_gen_sort[:,0],
                                                feat_score_gen_arr))
        feat_score_kf_cumu  = num.vstack(zip(   feat_imp_kf_sort[:,0],
                                                feat_score_kf_arr))
    ##
    ## Saving to dicitoanry
    model_dict = {}
    model_dict['model_score_tot'         ] = model_score_tot
    model_dict['kf_scores'               ] = kf_scores
    model_dict['model_gen'               ] = model_gen
    model_dict['kf_models_arr'           ] = kf_models_arr
    model_dict['model_gen_prediction_arr'] = model_gen_prediction_arr
    model_dict['model_gen_frac_diff_arr' ] = model_gen_frac_diff_arr
    model_dict['model_gen_truth_arr'     ] = model_gen_truth_arr
    model_dict['model_kf_prediction_arr' ] = model_kf_prediction_arr
    model_dict['model_kf_frac_diff_arr'  ] = model_kf_frac_diff_arr
    model_dict['score_dyn'               ] = score_dyn
    model_dict['score_ham'               ] = score_ham
    model_dict['pred_ham'                ] = pred_ham
    model_dict['true_halo_ham'           ] = true_halo_ham
    model_dict['pred_dyn_mod'            ] = pred_dyn_mod
    model_dict['true_halo_dyn_mod'       ] = true_halo_dyn_mod
    model_dict['frac_diff_ham'           ] = frac_diff_ham
    model_dict['frac_diff_dyn'           ] = frac_diff_dyn
    ##
    ## Only for algorithms that are not `neural_network`
    if skem_key != 'neural_network':
        model_dict['feat_imp_gen_sort'       ] = feat_imp_gen_sort
        model_dict['feat_imp_kf_sort'        ] = feat_imp_kf_sort
        model_dict['feat_score_gen_cumu'     ] = feat_score_gen_cumu
        model_dict['feat_score_kf_cumu'      ] = feat_score_kf_cumu

    return model_dict

## Training different models
def ml_models_training(train_dict, test_dict, param_dict, proj_dict,
    model_fits_dict):
    """
    Trains different ML models to determine metrics of accuracy and more for
    the given algorithm.

    Parameters
    ------------
    train_dict: python dictionary
        dictionary containing the 'training' data from the catalogue

    test_dict: python dictionary
        dictionary containing the 'testing' data from the catalogue

    param_dict: python dictionary
        dictionary with `project` variables

    proj_dict: python dictionary
        dictionary with info of the project that uses the
        `Data Science` Cookiecutter template.

    model_fits_dict: python dictionary
        Dictionary for storing 'fit' and 'score' data for different algorithms

    Returns
    ------------
    model_fits_dict: python dictionary
        Dictionary for storing 'fit' and 'score' data for different algorithms
    """
    ## List of keys for the different ML models
    skem_keys_arr = num.sort(list(param_dict['skem_dict'].keys()))
    ## Looping over each ML model
    for zz, skem_key in tqdm(enumerate(skem_keys_arr)):
        print('{0} Analyzing: `{1}`'.format(param_dict['Prog_msg'],skem_key))
        # Training Dataset
        model_fits_dict[skem_key] = model_score_general(train_dict,
                                                        test_dict,
                                                        skem_key,
                                                        param_dict)

    return model_fits_dict

## --------- Saving Data ------------##

## Saving results from algorithms
def saving_data(param_dict, proj_dict, model_fits_dict, train_dict, test_dict):
    """
    Saves the final data file

    Parameters
    ------------
    param_dict: python dictionary
        dictionary with `project` variables

    proj_dict: python dictionary
        dictionary with info of the project that uses the
        `Data Science` Cookiecutter template.

    model_fits_dict: python dictionary
        Dictionary for storing 'fit' and 'score' data for different algorithms

    train_dict: python dictionary
        dictionary containing the 'training' data from the catalogue

    test_dict: python dictionary
        dictionary containing the 'testing' data from the catalogue
    """
    ## Filename
    filepath_str  = '{0}_n_predict_{1}_pre_opt_{2}_sample_frac_{3}'
    filepath_str += '_score_method_{4}'
    filepath_str += '_model_fits_dict.p'
    filepath_str  = filepath_str.format(param_dict['catl_str' ],
                                        param_dict['n_predict'],
                                        param_dict['pre_opt'  ],
                                        param_dict['sample_frac'],
                                        param_dict['score_method'])
    filepath = os.path.join(    proj_dict['test_train_dir'],
                                filepath_str)
    ## Elements to be saved
    obj_arr = [model_fits_dict, train_dict, test_dict, param_dict]
    ## Saving pickle file
    with open(filepath, 'wb') as file_p:
        pickle.dump(obj_arr, file_p)
    ## Checking that file exists
    try:
        assert(os.path.exists(filepath))
    except:
        msg = '{0} File `{1}` was not found... Exiting'.format(
            param_dict['Prog_msg'], filepath)
    ##
    ##
    msg = '{0} Output file: {1}'.format(param_dict['Prog_msg'], filepath)
    print(msg)

## --------- Main Function ------------##

def main(args):
    """
    Trains a ML algorithm to predict galaxy and group properties.
    """
    ## Starting time
    start_time = datetime.now()
    ## Reading all elements and converting to python dictionary
    param_dict = vars(args)
    ## Initializing random seed
    num.random.seed(param_dict['seed'])
    ## Checking for correct input
    param_vals_test(param_dict)
    ## ---- Adding to `param_dict` ---- 
    param_dict = add_to_dict(param_dict)
    ## Program message
    Prog_msg = param_dict['Prog_msg']
    ##
    ## Creating Folder Structure
    # proj_dict  = directory_skeleton(param_dict, cwpaths.cookiecutter_paths(__file__))
    proj_dict  = directory_skeleton(param_dict, cwpaths.cookiecutter_paths('./'))
    ##
    ## Printing out project variables
    print('\n'+50*'='+'\n')
    for key, key_val in sorted(param_dict.items()):
        if key !='Prog_msg':
            if key == 'skem_dict':
                print('{0} `{1}`: {2}'.format(Prog_msg, key, list(key_val.keys())))
            else:
                print('{0} `{1}`: {2}'.format(Prog_msg, key, key_val))
    print('\n'+50*'='+'\n')
    ##
    ## Reading in merged catalogue and separating training and testing 
    ## dataset
    (   train_dict,
        test_dict ,
        param_dict) = training_testing_data(    
                                        param_dict,
                                        proj_dict, 
                                        test_size=param_dict['test_size'],
                                        random_state=param_dict['seed'],
                                        shuffle_opt=param_dict['shuffle_opt'],
                                        dropna_opt=param_dict['dropna_opt'],
                                        sample_frac=param_dict['sample_frac'])
    ##
    ## ----- Training Dataset - Algorithms -----
    # Dictionary for storing Fit data and scores
    model_fits_dict = {}
    ##
    ## Random Forest
    model_fits_dict = ml_models_training(train_dict, test_dict, param_dict,
        proj_dict, model_fits_dict)
    ##
    ## ----- Saving final resulst -----
    # Saving final result
    saving_data(param_dict, proj_dict, model_fits_dict, train_dict, test_dict)
    ##
    ## End time for running the catalogues
    end_time   = datetime.now()
    total_time = end_time - start_time
    print('{0} Total Time taken (Create): {1}'.format(Prog_msg, total_time))
    ##
    ## Making the `param_dict` None
    param_dict = None


# Main function
if __name__=='__main__':
    ## Input arguments
    args = get_parser()
    ## Checking galaxy sample
    # Main Function
    if args.pre_opt == 'all':
        for pre_opt_ii in ['min_max','standard','normalize']:
            if args.sample == 'all':
                for sample_ii in [19, 20, 21]:
                    print('\n'+50*'='+'\n')
                    print('\n Sample: {0}\n'.format(sample_ii))
                    print('\n'+50*'='+'\n')
                    ## Copy of `args`
                    args_c = copy.deepcopy(args)
                    ## Changing galaxy sample int
                    args_c.sample  = sample_ii
                    args_c.pre_opt = pre_opt_ii
                    main(args_c)
            else:
                args.sample  = int(args.sample)
                args.pre_opt = pre_opt_ii
                main(args)
    else:
        if args.sample == 'all':
            for sample_ii in [19, 20, 21]:
                print('\n'+50*'='+'\n')
                print('\n Sample: {0}\n'.format(sample_ii))
                print('\n'+50*'='+'\n')
                ## Copy of `args`
                args_c = copy.deepcopy(args)
                ## Changing galaxy sample int
                args_c.sample  = sample_ii
                main(args_c)
        else:
            args.sample  = int(args.sample)
            main(args)

