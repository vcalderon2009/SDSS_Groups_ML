#! /usr/bin/env python
# -*- coding: utf-8 -*-

# Victor Calderon
# Created      : 2018-06-04
# Last Modified: 2018-06-04
# Vanderbilt University
from __future__ import print_function, division, absolute_import
__author__     = ['Victor Calderon']
__copyright__  = ["Copyright 2017 Victor Calderon, SDSS Mocks Create - Make"]
__email__      = ['victor.calderon@vanderbilt.edu']
__maintainer__ = ['Victor Calderon']
"""
Main script to run the data preprocessing pipeline.
The pipeline includes:
    - Calculates the features for each galaxy/group mock catalogues
    - Transforms the features into proper inputs for the ML algoriths
    - Saves the features arrays as pickle files
"""

# Path to Custom Utilities folder
import os

# Importing Modules
from cosmo_utils.utils import file_utils as cfutils
from cosmo_utils.utils import work_paths as cwpaths

import numpy as num
import pandas as pd

# Extra-modules
import argparse
from argparse import ArgumentParser
from argparse import HelpFormatter
from operator import attrgetter
import datetime

# Get configuration information from setup.cfg
try:
    from ConfigParser import ConfigParser
except ImportError:
    from configparser import ConfigParser

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
    ## Number of HOD's to create. Dictates how many different types of
    ##      mock catalogues to create
    parser.add_argument('-hod_models_n',
                        dest='hod_models_n',
                        help="""
                        HOD models to use for this analysis. The values in the
                        string consist of the `first` HOD model, which will be
                        used for the `training` dataset, while, the next
                        set of HOD numbers will be use to validate the
                        `training` of the ML algorithm.
                        """,
                        type=str,
                        default='1_2_3_4_5_6_7_8')
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
    ## Number of distinct velocity bias measurements to create.
    ## Dictates how many different types of mock catalogues to create
    parser.add_argument('-dv_models_n',
                        dest='dv_models_n',
                        help="""
                        Velocity dispersion models to use for this analysis.
                        The values in the string consist of the `first`
                        models with a different velocity dispersion, which
                        will be used for the `testing` of the ML algorithm.
                        """,
                        type=str,
                        default='0.9_0.925_0.95_0.975_1.025_1.05_1.10')
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
                        choices=['hod_dv_fixed', 'dv_fixed', 'hod_fixed'],
                        default='hod_dv_fixed')
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
    ## Algorithm used for the final estimation of mass
    parser.add_argument('-chosen_ml_alg',
                        dest='chosen_ml_alg',
                        help='Algorithm used for the final estimation of mass',
                        type=str,
                        choices=['xgboost', 'rf', 'nn'],
                        default='xgboost')
    ## Type of resampling to use if necessary
    parser.add_argument('-resample_opt',
                        dest='resample_opt',
                        help='Type of resampling to use if necessary',
                        type=str,
                        choices=['over', 'under'],
                        default='under')
    ## Option to include results from Neural Network or Not include_nn
    parser.add_argument('-include_nn',
                        dest='include_nn',
                        help="""
                        Option to include results from Neural network or not.
                        """,
                        type=_str2bool,
                        default=False)
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
    ## Metadata from project
    # proj_dict = cwpaths.cookiecutter_paths(__file__)
    proj_dict = cwpaths.cookiecutter_paths('./')
    conf = ConfigParser()
    conf.read([os.path.join(proj_dict['base_dir'], 'setup.cfg')])
    setup_cfg = dict(conf.items('metadata'))
    param_dict['setup_cfg'] = setup_cfg
    ## Project Constants
    param_dict = project_const(param_dict)

    return param_dict

def df_value_modifier(df, name, param_dict):
    """
    Modifies the values for in a given DataFrame based on user input.

    Parameters
    -----------
    df : `pandas.DataFrame`
        DataFrame which stores the data about the `name` field.

    name : `str`
        Identifier that will be updated with the values in `param_dict`.

    param_dict : `dict`
        dictionary with project variables
    """
    ## Modifying name
    df.loc[df['Name'] == name, 'Value'] = param_dict[name]

    return df

### --------------- Analysis Parameters --------------- ###

def get_analysis_alg_comp_params(param_dict):
    """
    Parameters for the data analysis step, right after training and
    testing ML algorithms.

    Parameters
    -----------
    param_dict : `dict`
        dictionary with project variables

    Returns
    --------
    catl_feat_df : `pd.DataFrame`
        DataFrame with necessary parameters to run `catl_feature_calculation`
        script.
    """
    ##
    ## Array of values used for the analysis.
    ## Format: (name of variable, flag, value)
    #
    ## --------------------------------------------------------------------- ##
    ## Calculation for the ML analysis predictions - Parameters
    ## --------------------------------------------------------------------- ##
    catl_params_main_arr = num.array([
                            ('hod_n'         , '-hod_model_n'   , 0          ),
                            ('halotype'      , '-halotype'      , 'so'       ),
                            ('clf_method'    , '-clf_method'    , 1          ),
                            ('dv'            , '-dv'            , 1.0        ),
                            ('clf_seed'      , '-clf_seed'      , 1235       ),
                            ('sample'        , '-sample'        , '19'       ),
                            ('catl_type'     , '-abopt'         , 'mr'       ),
                            ('cosmo_choice'  , '-cosmo'         , 'LasDamas' ),
                            ('nmin'          , '-nmin'          , 2          ),
                            ('mass_factor'   , '-mass_factor'   , 10         ),
                            ('n_predict'     , '-n_predict'     , 1          ),
                            ('shuffle_opt'   , '-shuffle_opt'   , True       ),
                            ('dropna_opt'    , '-dropna_opt'    , True       ),
                            ('pre_opt'       , '-pre_opt'       , 'standard' ),
                            ('test_train_opt', '-test_train_opt', 'boxes_n'  ),
                            ('box_idx'       , '-box_idx'       , '0_4_5'    ),
                            ('box_test'      , '-box_test'      , 0          ),
                            ('sample_frac'   , '-sample_frac'   , 0.01       ),
                            ('test_size'     , '-test_size'     , 0.25       ),
                            ('n_feat_use'    , '-n_feat_use'    , 'sub'      ),
                            ('dens_calc'     , '-dens_calc'     , True       ),
                            ('kf_splits'     , '-kf_splits'     , 3          ),
                            ('hidden_layers' , '-hidden_layers' , 3          ),
                            ('unit_layer'    , '-unit_layer'    , 100        ),
                            ('score_method'  , '-score_method'  , 'threshold'),
                            ('threshold'     , '-threshold'     , 0.1        ),
                            ('perc_val'      , '-perc_val'      , 0.68       ),
                            ('sample_method' , '-sample_method' , 'binning'  ),
                            ('bin_val'       , '-bin_val'       , 'fixed'    ),
                            ('ml_analysis'   , '-ml_analysis'   , 'hod_dv_fixed'),
                            ('resample_opt'  , '-resample_opt'  , 'under'    ),
                            ('cpu_frac'      , '-cpu'           , 0.75       ),
                            ('remove_files'  , '-remove'        , False      ),
                            ('verbose'       , '-v'             , False      ),
                            ('perf_opt'      , '-perf'          , False      ),
                            ('seed'          , '-seed'          , 1         )])
    ##
    ## Converting to pandas DataFrame
    colnames = ['Name', 'Flag', 'Value']
    main_df = pd.DataFrame(catl_params_main_arr, columns=colnames)
    ##
    ## Sorting out DataFrame by `name`
    main_df = main_df.sort_values(by='Name')
    main_df.reset_index(inplace=True, drop=True)
    ##
    ## HOD Model to use
    main_df = df_value_modifier(main_df, 'hod_n', param_dict)
    ##
    ## Type of dark matter halo to use in the simulation
    main_df = df_value_modifier(main_df, 'halotype', param_dict)
    ##
    ## CLF Method for assigning galaxy properties
    main_df = df_value_modifier(main_df, 'clf_method', param_dict)
    ##
    ## Random seed used during the CLF assignment
    main_df = df_value_modifier(main_df, 'clf_seed', param_dict)
    ##
    ## Difference between galaxy and mass velocity profiles
    main_df = df_value_modifier(main_df, 'dv', param_dict)
    ##
    ## SDSS luminosity sample to analyze
    main_df = df_value_modifier(main_df, 'sample', param_dict)
    ##
    ## Type of Abundance matching
    main_df = df_value_modifier(main_df, 'catl_type', param_dict)
    ##
    ## Cosmology choice
    main_df = df_value_modifier(main_df, 'cosmo_choice', param_dict)
    ##
    ## Minimum number of galaxies in a group
    main_df = df_value_modifier(main_df, 'nmin', param_dict)
    ##
    ## Total number of properties to predict. Default = 1
    main_df = df_value_modifier(main_df, 'n_predict', param_dict)
    ##
    ## Option for shuffling dataset when creating `testing` and `training`
    ## datasets
    main_df = df_value_modifier(main_df, 'shuffle_opt', param_dict)
    ##
    ## Option for Shuffling dataset when separing `training` and `testing` sets
    main_df = df_value_modifier(main_df, 'dropna_opt', param_dict)
    ##
    ## Option for which preprocessing of the data to use.
    main_df = df_value_modifier(main_df, 'pre_opt', param_dict)
    ##
    ## Option for which kind of separation of training/testing to use for the 
    ## datasets.
    main_df = df_value_modifier(main_df, 'test_train_opt', param_dict)
    ##
    ## Initial and final indices of the simulation boxes to use for the 
    ## testing and training datasets.
    main_df = df_value_modifier(main_df, 'box_idx', param_dict)
    ##
    ## Index of the simulation box to use for the `training` and `testing
    main_df = df_value_modifier(main_df, 'box_test', param_dict)
    ##
    ## Fraction of the sample to be used.
    ## Only if `test_train_opt == 'sample_frac'`
    main_df = df_value_modifier(main_df, 'sample_frac', param_dict)
    ##
    ## Testing size for ML
    main_df = df_value_modifier(main_df, 'test_size', param_dict)
    ##
    ## Option for using all features or just a few
    main_df = df_value_modifier(main_df, 'n_feat_use', param_dict)
    ##
    ## Option for calculating densities or not
    main_df = df_value_modifier(main_df, 'dens_calc', param_dict)
    ##
    ## Number of hidden layers to use for neural network
    main_df = df_value_modifier(main_df, 'hidden_layers', param_dict)
    ##
    ## Number of units per hidden layer for the neural. network.
    ## Default = `100`.
    main_df = df_value_modifier(main_df, 'unit_layer', param_dict)
    ##
    ## Option for determining which scoring method to use.
    main_df = df_value_modifier(main_df, 'score_method', param_dict)
    ##
    ## Threshold value used for when ``score_method == 'threshold'``
    main_df = df_value_modifier(main_df, 'threshold', param_dict)
    ##
    ## Percentage value used for when ``score_method == 'perc'``
    main_df = df_value_modifier(main_df, 'perc_val', param_dict)
    ##
    ## Method for binning or sumsample the array of the estimated group mass.
    main_df = df_value_modifier(main_df, 'sample_method', param_dict)
    ##
    ## Type of binning to use for the mass
    main_df = df_value_modifier(main_df, 'bin_val', param_dict)
    ##
    ## Type of analysis to perform.
    main_df = df_value_modifier(main_df, 'ml_analysis', param_dict)
    ##
    ## Type of resampling to use if necessary
    main_df = df_value_modifier(main_df, 'resample_opt', param_dict)
    ##
    ## Percentage of CPU to use
    main_df = df_value_modifier(main_df, 'cpu_frac', param_dict)
    ##
    ## Option for removing files or not
    main_df = df_value_modifier(main_df, 'remove_files', param_dict)
    ##
    ## Option for displaying outputs or not
    main_df = df_value_modifier(main_df, 'verbose', param_dict)
    ##
    ## Option for looking at `perfect` mock catalogues
    main_df = df_value_modifier(main_df, 'perf_opt', param_dict)
    ##
    ## Random seed for the analysis
    main_df = df_value_modifier(main_df, 'seed', param_dict)
    ##
    ## --------------------------------------------------------------------- ##
    ## Calculation for the ML analysis predictions - Parameters - Plot
    ## --------------------------------------------------------------------- ##
    catl_params_main_plot_arr = num.array([
                            ('hod_n'         , '-hod_model_n'   , 0          ),
                            ('halotype'      , '-halotype'      , 'so'       ),
                            ('clf_method'    , '-clf_method'    , 1          ),
                            ('dv'            , '-dv'            , 1.0        ),
                            ('clf_seed'      , '-clf_seed'      , 1235       ),
                            ('sample'        , '-sample'        , '19'       ),
                            ('catl_type'     , '-abopt'         , 'mr'       ),
                            ('cosmo_choice'  , '-cosmo'         , 'LasDamas' ),
                            ('nmin'          , '-nmin'          , 2          ),
                            ('mass_factor'   , '-mass_factor'   , 10         ),
                            ('n_predict'     , '-n_predict'     , 1          ),
                            ('shuffle_opt'   , '-shuffle_opt'   , True       ),
                            ('dropna_opt'    , '-dropna_opt'    , True       ),
                            ('pre_opt'       , '-pre_opt'       , 'standard' ),
                            ('test_train_opt', '-test_train_opt', 'boxes_n'  ),
                            ('box_idx'       , '-box_idx'       , '0_4_5'    ),
                            ('box_test'      , '-box_test'      , 0          ),
                            ('sample_frac'   , '-sample_frac'   , 0.01       ),
                            ('test_size'     , '-test_size'     , 0.25       ),
                            ('n_feat_use'    , '-n_feat_use'    , 'sub'      ),
                            ('dens_calc'     , '-dens_calc'     , True       ),
                            ('kf_splits'     , '-kf_splits'     , 3          ),
                            ('hidden_layers' , '-hidden_layers' , 3          ),
                            ('unit_layer'    , '-unit_layer'    , 100        ),
                            ('score_method'  , '-score_method'  , 'threshold'),
                            ('threshold'     , '-threshold'     , 0.1        ),
                            ('perc_val'      , '-perc_val'      , 0.68       ),
                            ('sample_method' , '-sample_method' , 'binning'  ),
                            ('bin_val'       , '-bin_val'       , 'fixed'    ),
                            ('ml_analysis'   , '-ml_analysis'   , 'hod_dv_fixed'),
                            ('resample_opt'  , '-resample_opt'  , 'under'    ),
                            ('plot_opt'      , '-plot_opt'      , 'mhalo'    ),
                            ('rank_opt'      , '-rank_opt'      , 'idx'      ),
                            ('cpu_frac'      , '-cpu'           , 0.75       ),
                            ('remove_files'  , '-remove'        , False      ),
                            ('verbose'       , '-v'             , False      ),
                            ('perf_opt'      , '-perf'          , False      ),
                            ('seed'          , '-seed'          , 1         )])
    ##
    ## Converting to pandas DataFrame
    colnames = ['Name', 'Flag', 'Value']
    main_plot_df = pd.DataFrame(catl_params_main_plot_arr, columns=colnames)
    ##
    ## Sorting out DataFrame by `name`
    main_plot_df = main_plot_df.sort_values(by='Name')
    main_plot_df.reset_index(inplace=True, drop=True)
    ##
    ## HOD Model to use
    main_plot_df = df_value_modifier(main_plot_df, 'hod_n', param_dict)
    ##
    ## Type of dark matter halo to use in the simulation
    main_plot_df = df_value_modifier(main_plot_df, 'halotype', param_dict)
    ##
    ## CLF Method for assigning galaxy properties
    main_plot_df = df_value_modifier(main_plot_df, 'clf_method', param_dict)
    ##
    ## Random seed used during the CLF assignment
    main_plot_df = df_value_modifier(main_plot_df, 'clf_seed', param_dict)
    ##
    ## Difference between galaxy and mass velocity profiles
    main_plot_df = df_value_modifier(main_plot_df, 'dv', param_dict)
    ##
    ## SDSS luminosity sample to analyze
    main_plot_df = df_value_modifier(main_plot_df, 'sample', param_dict)
    ##
    ## Type of Abundance matching
    main_plot_df = df_value_modifier(main_plot_df, 'catl_type', param_dict)
    ##
    ## Cosmology choice
    main_plot_df = df_value_modifier(main_plot_df, 'cosmo_choice', param_dict)
    ##
    ## Minimum number of galaxies in a group
    main_plot_df = df_value_modifier(main_plot_df, 'nmin', param_dict)
    ##
    ## Total number of properties to predict. Default = 1
    main_plot_df = df_value_modifier(main_plot_df, 'n_predict', param_dict)
    ##
    ## Option for shuffling dataset when creating `testing` and `training`
    ## datasets
    main_plot_df = df_value_modifier(main_plot_df, 'shuffle_opt', param_dict)
    ##
    ## Option for Shuffling dataset when separing `training` and `testing` sets
    main_plot_df = df_value_modifier(main_plot_df, 'dropna_opt', param_dict)
    ##
    ## Option for which preprocessing of the data to use.
    main_plot_df = df_value_modifier(main_plot_df, 'pre_opt', param_dict)
    ##
    ## Option for which kind of separation of training/testing to use for the 
    ## datasets.
    main_plot_df = df_value_modifier(main_plot_df, 'test_train_opt', param_dict)
    ##
    ## Initial and final indices of the simulation boxes to use for the 
    ## testing and training datasets.
    main_plot_df = df_value_modifier(main_plot_df, 'box_idx', param_dict)
    ##
    ## Index of the simulation box to use for the `training` and `testing
    main_plot_df = df_value_modifier(main_plot_df, 'box_test', param_dict)
    ##
    ## Fraction of the sample to be used.
    ## Only if `test_train_opt == 'sample_frac'`
    main_plot_df = df_value_modifier(main_plot_df, 'sample_frac', param_dict)
    ##
    ## Testing size for ML
    main_plot_df = df_value_modifier(main_plot_df, 'test_size', param_dict)
    ##
    ## Option for using all features or just a few
    main_plot_df = df_value_modifier(main_plot_df, 'n_feat_use', param_dict)
    ##
    ## Option for calculating densities or not
    main_plot_df = df_value_modifier(main_plot_df, 'dens_calc', param_dict)
    ##
    ## Number of hidden layers to use for neural network
    main_plot_df = df_value_modifier(main_plot_df, 'hidden_layers', param_dict)
    ##
    ## Number of units per hidden layer for the neural. network.
    ## Default = `100`.
    main_plot_df = df_value_modifier(main_plot_df, 'unit_layer', param_dict)
    ##
    ## Option for determining which scoring method to use.
    main_plot_df = df_value_modifier(main_plot_df, 'score_method', param_dict)
    ##
    ## Threshold value used for when ``score_method == 'threshold'``
    main_plot_df = df_value_modifier(main_plot_df, 'threshold', param_dict)
    ##
    ## Percentage value used for when ``score_method == 'perc'``
    main_plot_df = df_value_modifier(main_plot_df, 'perc_val', param_dict)
    ##
    ## Method for binning or sumsample the array of the estimated group mass.
    main_plot_df = df_value_modifier(main_plot_df, 'sample_method', param_dict)
    ##
    ## Type of binning to use for the mass
    main_plot_df = df_value_modifier(main_plot_df, 'bin_val', param_dict)
    ##
    ## Type of analysis to perform.
    main_plot_df = df_value_modifier(main_plot_df, 'ml_analysis', param_dict)
    ##
    ## Option for which variable to plot on x-axis
    main_plot_df = df_value_modifier(main_plot_df, 'plot_opt', param_dict)
    ##
    ## Option for which type of ranking to plot
    main_plot_df = df_value_modifier(main_plot_df, 'rank_opt', param_dict)
    ##
    ## Type of resampling to use if necessary
    main_plot_df = df_value_modifier(main_plot_df, 'resample_opt', param_dict)
    ##
    ## Percentage of CPU to use
    main_plot_df = df_value_modifier(main_plot_df, 'cpu_frac', param_dict)
    ##
    ## Option for removing files or not
    main_plot_df = df_value_modifier(main_plot_df, 'remove_files', param_dict)
    ##
    ## Option for displaying outputs or not
    main_plot_df = df_value_modifier(main_plot_df, 'verbose', param_dict)
    ##
    ## Option for looking at `perfect` mock catalogues
    main_plot_df = df_value_modifier(main_plot_df, 'perf_opt', param_dict)
    ##
    ## Random seed for the analysis
    main_plot_df = df_value_modifier(main_plot_df, 'seed', param_dict)

    return [main_df, main_plot_df]

def get_analysis_hod_diff_params(param_dict):
    """
    Parameters for the data analysis step, right after training and
    testing ML algorithms.

    Parameters
    -----------
    param_dict : `dict`
        dictionary with project variables

    Returns
    --------
    catl_feat_df : `pd.DataFrame`
        DataFrame with necessary parameters to run `catl_feature_calculation`
        script.
    """
    ##
    ## Array of values used for the analysis.
    ## Format: (name of variable, flag, value)
    #
    ## --------------------------------------------------------------------- ##
    ## Calculation for the ML analysis predictions - Parameters
    ## --------------------------------------------------------------------- ##
    catl_params_main_arr = num.array([
                            ('hod_n'         , '-hod_model_n'   , 0          ),
                            ('hod_models_n'  , '-hod_models_n'  , '0_1_2_3_4_5_6_7_8'),
                            ('halotype'      , '-halotype'      , 'so'       ),
                            ('clf_method'    , '-clf_method'    , 1          ),
                            ('dv'            , '-dv'            , 1.0        ),
                            ('clf_seed'      , '-clf_seed'      , 1235       ),
                            ('sample'        , '-sample'        , '19'       ),
                            ('catl_type'     , '-abopt'         , 'mr'       ),
                            ('cosmo_choice'  , '-cosmo'         , 'LasDamas' ),
                            ('nmin'          , '-nmin'          , 2          ),
                            ('mass_factor'   , '-mass_factor'   , 10         ),
                            ('n_predict'     , '-n_predict'     , 1          ),
                            ('shuffle_opt'   , '-shuffle_opt'   , True       ),
                            ('dropna_opt'    , '-dropna_opt'    , True       ),
                            ('pre_opt'       , '-pre_opt'       , 'standard' ),
                            ('test_train_opt', '-test_train_opt', 'boxes_n'  ),
                            ('box_idx'       , '-box_idx'       , '0_4_5'    ),
                            ('box_test'      , '-box_test'      , 0          ),
                            ('sample_frac'   , '-sample_frac'   , 0.01       ),
                            ('test_size'     , '-test_size'     , 0.25       ),
                            ('n_feat_use'    , '-n_feat_use'    , 'sub'      ),
                            ('dens_calc'     , '-dens_calc'     , True       ),
                            ('kf_splits'     , '-kf_splits'     , 3          ),
                            ('hidden_layers' , '-hidden_layers' , 3          ),
                            ('unit_layer'    , '-unit_layer'    , 100        ),
                            ('score_method'  , '-score_method'  , 'threshold'),
                            ('threshold'     , '-threshold'     , 0.1        ),
                            ('perc_val'      , '-perc_val'      , 0.68       ),
                            ('sample_method' , '-sample_method' , 'binning'  ),
                            ('bin_val'       , '-bin_val'       , 'fixed'    ),
                            ('ml_analysis'   , '-ml_analysis'   , 'dv_fixed' ),
                            ('resample_opt'  , '-resample_opt'  , 'under'    ),
                            ('include_nn'    , '-include_nn'    , False      ),
                            ('cpu_frac'      , '-cpu'           , 0.75       ),
                            ('remove_files'  , '-remove'        , False      ),
                            ('verbose'       , '-v'             , False      ),
                            ('perf_opt'      , '-perf'          , False      ),
                            ('seed'          , '-seed'          , 1         )])
    ##
    ## Converting to pandas DataFrame
    colnames = ['Name', 'Flag', 'Value']
    main_df = pd.DataFrame(catl_params_main_arr, columns=colnames)
    ##
    ## Sorting out DataFrame by `name`
    main_df = main_df.sort_values(by='Name')
    main_df.reset_index(inplace=True, drop=True)
    ##
    ## HOD Model to use
    main_df = df_value_modifier(main_df, 'hod_n', param_dict)
    ##
    ## HOD Model to use
    main_df = df_value_modifier(main_df, 'hod_models_n', param_dict)
    ##
    ## Type of dark matter halo to use in the simulation
    main_df = df_value_modifier(main_df, 'halotype', param_dict)
    ##
    ## CLF Method for assigning galaxy properties
    main_df = df_value_modifier(main_df, 'clf_method', param_dict)
    ##
    ## Random seed used during the CLF assignment
    main_df = df_value_modifier(main_df, 'clf_seed', param_dict)
    ##
    ## Difference between galaxy and mass velocity profiles
    main_df = df_value_modifier(main_df, 'dv', param_dict)
    ##
    ## SDSS luminosity sample to analyze
    main_df = df_value_modifier(main_df, 'sample', param_dict)
    ##
    ## Type of Abundance matching
    main_df = df_value_modifier(main_df, 'catl_type', param_dict)
    ##
    ## Cosmology choice
    main_df = df_value_modifier(main_df, 'cosmo_choice', param_dict)
    ##
    ## Minimum number of galaxies in a group
    main_df = df_value_modifier(main_df, 'nmin', param_dict)
    ##
    ## Total number of properties to predict. Default = 1
    main_df = df_value_modifier(main_df, 'n_predict', param_dict)
    ##
    ## Option for shuffling dataset when creating `testing` and `training`
    ## datasets
    main_df = df_value_modifier(main_df, 'shuffle_opt', param_dict)
    ##
    ## Option for Shuffling dataset when separing `training` and `testing` sets
    main_df = df_value_modifier(main_df, 'dropna_opt', param_dict)
    ##
    ## Option for which preprocessing of the data to use.
    main_df = df_value_modifier(main_df, 'pre_opt', param_dict)
    ##
    ## Option for which kind of separation of training/testing to use for the 
    ## datasets.
    main_df = df_value_modifier(main_df, 'test_train_opt', param_dict)
    ##
    ## Initial and final indices of the simulation boxes to use for the 
    ## testing and training datasets.
    main_df = df_value_modifier(main_df, 'box_idx', param_dict)
    ##
    ## Index of the simulation box to use for the `training` and `testing
    main_df = df_value_modifier(main_df, 'box_test', param_dict)
    ##
    ## Fraction of the sample to be used.
    ## Only if `test_train_opt == 'sample_frac'`
    main_df = df_value_modifier(main_df, 'sample_frac', param_dict)
    ##
    ## Testing size for ML
    main_df = df_value_modifier(main_df, 'test_size', param_dict)
    ##
    ## Option for using all features or just a few
    main_df = df_value_modifier(main_df, 'n_feat_use', param_dict)
    ##
    ## Option for calculating densities or not
    main_df = df_value_modifier(main_df, 'dens_calc', param_dict)
    ##
    ## Number of hidden layers to use for neural network
    main_df = df_value_modifier(main_df, 'hidden_layers', param_dict)
    ##
    ## Number of units per hidden layer for the neural. network.
    ## Default = `100`.
    main_df = df_value_modifier(main_df, 'unit_layer', param_dict)
    ##
    ## Option for determining which scoring method to use.
    main_df = df_value_modifier(main_df, 'score_method', param_dict)
    ##
    ## Threshold value used for when ``score_method == 'threshold'``
    main_df = df_value_modifier(main_df, 'threshold', param_dict)
    ##
    ## Percentage value used for when ``score_method == 'perc'``
    main_df = df_value_modifier(main_df, 'perc_val', param_dict)
    ##
    ## Method for binning or sumsample the array of the estimated group mass.
    main_df = df_value_modifier(main_df, 'sample_method', param_dict)
    ##
    ## Type of binning to use for the mass
    main_df = df_value_modifier(main_df, 'bin_val', param_dict)
    ##
    ## Type of analysis to perform.
    main_df = df_value_modifier(main_df, 'ml_analysis', param_dict)
    ##
    ## Type of resampling to use if necessary
    main_df = df_value_modifier(main_df, 'resample_opt', param_dict)
    ##
    ## Percentage of CPU to use
    main_df = df_value_modifier(main_df, 'cpu_frac', param_dict)
    ##
    ## Option for removing files or not
    main_df = df_value_modifier(main_df, 'remove_files', param_dict)
    ##
    ## Option for displaying outputs or not
    main_df = df_value_modifier(main_df, 'verbose', param_dict)
    ##
    ## Option for looking at `perfect` mock catalogues
    main_df = df_value_modifier(main_df, 'perf_opt', param_dict)
    ##
    ## Random seed for the analysis
    main_df = df_value_modifier(main_df, 'seed', param_dict)
    ##
    ## --------------------------------------------------------------------- ##
    ## Calculation for the ML analysis predictions - Parameters - Plot
    ## --------------------------------------------------------------------- ##
    catl_params_main_plot_arr = num.array([
                            ('hod_n'         , '-hod_model_n'   , 0          ),
                            ('hod_models_n'  , '-hod_models_n'  , '0_1_2_3_4_5_6_7_8'),
                            ('halotype'      , '-halotype'      , 'so'       ),
                            ('clf_method'    , '-clf_method'    , 1          ),
                            ('dv'            , '-dv'            , 1.0        ),
                            ('clf_seed'      , '-clf_seed'      , 1235       ),
                            ('sample'        , '-sample'        , '19'       ),
                            ('catl_type'     , '-abopt'         , 'mr'       ),
                            ('cosmo_choice'  , '-cosmo'         , 'LasDamas' ),
                            ('nmin'          , '-nmin'          , 2          ),
                            ('mass_factor'   , '-mass_factor'   , 10         ),
                            ('n_predict'     , '-n_predict'     , 1          ),
                            ('shuffle_opt'   , '-shuffle_opt'   , True       ),
                            ('dropna_opt'    , '-dropna_opt'    , True       ),
                            ('pre_opt'       , '-pre_opt'       , 'standard' ),
                            ('test_train_opt', '-test_train_opt', 'boxes_n'  ),
                            ('box_idx'       , '-box_idx'       , '0_4_5'    ),
                            ('box_test'      , '-box_test'      , 0          ),
                            ('sample_frac'   , '-sample_frac'   , 0.01       ),
                            ('test_size'     , '-test_size'     , 0.25       ),
                            ('n_feat_use'    , '-n_feat_use'    , 'sub'      ),
                            ('dens_calc'     , '-dens_calc'     , True       ),
                            ('kf_splits'     , '-kf_splits'     , 3          ),
                            ('hidden_layers' , '-hidden_layers' , 3          ),
                            ('unit_layer'    , '-unit_layer'    , 100        ),
                            ('score_method'  , '-score_method'  , 'threshold'),
                            ('threshold'     , '-threshold'     , 0.1        ),
                            ('perc_val'      , '-perc_val'      , 0.68       ),
                            ('sample_method' , '-sample_method' , 'binning'  ),
                            ('bin_val'       , '-bin_val'       , 'fixed'    ),
                            ('ml_analysis'   , '-ml_analysis'   , 'dv_fixed' ),
                            ('resample_opt'  , '-resample_opt'  , 'under'    ),
                            ('include_nn'    , '-include_nn'    , False      ),
                            ('plot_opt'      , '-plot_opt'      , 'mhalo'    ),
                            ('rank_opt'      , '-rank_opt'      , 'idx'      ),
                            ('chosen_ml_alg' , '-chosen_ml_alg' , 'xgboost'  ),
                            ('cpu_frac'      , '-cpu'           , 0.75       ),
                            ('remove_files'  , '-remove'        , False      ),
                            ('verbose'       , '-v'             , False      ),
                            ('perf_opt'      , '-perf'          , False      ),
                            ('seed'          , '-seed'          , 1         )])
    ##
    ## Converting to pandas DataFrame
    colnames = ['Name', 'Flag', 'Value']
    main_plot_df = pd.DataFrame(catl_params_main_plot_arr, columns=colnames)
    ##
    ## Sorting out DataFrame by `name`
    main_plot_df = main_plot_df.sort_values(by='Name')
    main_plot_df.reset_index(inplace=True, drop=True)
    ##
    ## HOD Model to use
    main_plot_df = df_value_modifier(main_plot_df, 'hod_n', param_dict)
    ##
    ## HOD Model to use
    main_plot_df = df_value_modifier(main_plot_df, 'hod_models_n', param_dict)
    ##
    ## Type of dark matter halo to use in the simulation
    main_plot_df = df_value_modifier(main_plot_df, 'halotype', param_dict)
    ##
    ## CLF Method for assigning galaxy properties
    main_plot_df = df_value_modifier(main_plot_df, 'clf_method', param_dict)
    ##
    ## Random seed used during the CLF assignment
    main_plot_df = df_value_modifier(main_plot_df, 'clf_seed', param_dict)
    ##
    ## Difference between galaxy and mass velocity profiles
    main_plot_df = df_value_modifier(main_plot_df, 'dv', param_dict)
    ##
    ## SDSS luminosity sample to analyze
    main_plot_df = df_value_modifier(main_plot_df, 'sample', param_dict)
    ##
    ## Type of Abundance matching
    main_plot_df = df_value_modifier(main_plot_df, 'catl_type', param_dict)
    ##
    ## Cosmology choice
    main_plot_df = df_value_modifier(main_plot_df, 'cosmo_choice', param_dict)
    ##
    ## Minimum number of galaxies in a group
    main_plot_df = df_value_modifier(main_plot_df, 'nmin', param_dict)
    ##
    ## Total number of properties to predict. Default = 1
    main_plot_df = df_value_modifier(main_plot_df, 'n_predict', param_dict)
    ##
    ## Option for shuffling dataset when creating `testing` and `training`
    ## datasets
    main_plot_df = df_value_modifier(main_plot_df, 'shuffle_opt', param_dict)
    ##
    ## Option for Shuffling dataset when separing `training` and `testing` sets
    main_plot_df = df_value_modifier(main_plot_df, 'dropna_opt', param_dict)
    ##
    ## Option for which preprocessing of the data to use.
    main_plot_df = df_value_modifier(main_plot_df, 'pre_opt', param_dict)
    ##
    ## Option for which kind of separation of training/testing to use for the 
    ## datasets.
    main_plot_df = df_value_modifier(main_plot_df, 'test_train_opt', param_dict)
    ##
    ## Initial and final indices of the simulation boxes to use for the 
    ## testing and training datasets.
    main_plot_df = df_value_modifier(main_plot_df, 'box_idx', param_dict)
    ##
    ## Index of the simulation box to use for the `training` and `testing
    main_plot_df = df_value_modifier(main_plot_df, 'box_test', param_dict)
    ##
    ## Fraction of the sample to be used.
    ## Only if `test_train_opt == 'sample_frac'`
    main_plot_df = df_value_modifier(main_plot_df, 'sample_frac', param_dict)
    ##
    ## Testing size for ML
    main_plot_df = df_value_modifier(main_plot_df, 'test_size', param_dict)
    ##
    ## Option for using all features or just a few
    main_plot_df = df_value_modifier(main_plot_df, 'n_feat_use', param_dict)
    ##
    ## Option for calculating densities or not
    main_plot_df = df_value_modifier(main_plot_df, 'dens_calc', param_dict)
    ##
    ## Number of hidden layers to use for neural network
    main_plot_df = df_value_modifier(main_plot_df, 'hidden_layers', param_dict)
    ##
    ## Number of units per hidden layer for the neural. network.
    ## Default = `100`.
    main_plot_df = df_value_modifier(main_plot_df, 'unit_layer', param_dict)
    ##
    ## Option for determining which scoring method to use.
    main_plot_df = df_value_modifier(main_plot_df, 'score_method', param_dict)
    ##
    ## Threshold value used for when ``score_method == 'threshold'``
    main_plot_df = df_value_modifier(main_plot_df, 'threshold', param_dict)
    ##
    ## Percentage value used for when ``score_method == 'perc'``
    main_plot_df = df_value_modifier(main_plot_df, 'perc_val', param_dict)
    ##
    ## Method for binning or sumsample the array of the estimated group mass.
    main_plot_df = df_value_modifier(main_plot_df, 'sample_method', param_dict)
    ##
    ## Type of binning to use for the mass
    main_plot_df = df_value_modifier(main_plot_df, 'bin_val', param_dict)
    ##
    ## Type of analysis to perform.
    main_plot_df = df_value_modifier(main_plot_df, 'ml_analysis', param_dict)
    ##
    ## Option for which variable to plot on x-axis
    main_plot_df = df_value_modifier(main_plot_df, 'plot_opt', param_dict)
    ##
    ## Option for which type of ranking to plot
    main_plot_df = df_value_modifier(main_plot_df, 'rank_opt', param_dict)
    ##
    ## Option for which algorithm to plot
    main_plot_df = df_value_modifier(main_plot_df, 'chosen_ml_alg', param_dict)
    ##
    ## Type of resampling to use if necessary
    main_plot_df = df_value_modifier(main_plot_df, 'resample_opt', param_dict)
    ##
    ## Option to include results from neural network or not
    main_plot_df = df_value_modifier(main_plot_df, 'include_nn', param_dict)
    ##
    ## Percentage of CPU to use
    main_plot_df = df_value_modifier(main_plot_df, 'cpu_frac', param_dict)
    ##
    ## Option for removing files or not
    main_plot_df = df_value_modifier(main_plot_df, 'remove_files', param_dict)
    ##
    ## Option for displaying outputs or not
    main_plot_df = df_value_modifier(main_plot_df, 'verbose', param_dict)
    ##
    ## Option for looking at `perfect` mock catalogues
    main_plot_df = df_value_modifier(main_plot_df, 'perf_opt', param_dict)
    ##
    ## Random seed for the analysis
    main_plot_df = df_value_modifier(main_plot_df, 'seed', param_dict)

    return [main_df, main_plot_df]

def get_analysis_dv_diff_params(param_dict):
    """
    Parameters for the data analysis step, right after training and
    testing ML algorithms.

    Parameters
    -----------
    param_dict : `dict`
        dictionary with project variables

    Returns
    --------
    catl_feat_df : `pd.DataFrame`
        DataFrame with necessary parameters to run `catl_feature_calculation`
        script.
    """
    ##
    ## Array of values used for the analysis.
    ## Format: (name of variable, flag, value)
    #
    ## --------------------------------------------------------------------- ##
    ## Calculation for the ML analysis predictions - Parameters
    ## --------------------------------------------------------------------- ##
    catl_params_main_arr = num.array([
                            ('hod_n'         , '-hod_model_n'   , 0          ),
                            ('dv_models_n'  , '-dv_models_n'  , '0.9_0.925_0.95_0.975_1.0_1.025_1.05_1.10'),
                            ('halotype'      , '-halotype'      , 'so'       ),
                            ('clf_method'    , '-clf_method'    , 1          ),
                            ('dv'            , '-dv'            , 1.0        ),
                            ('clf_seed'      , '-clf_seed'      , 1235       ),
                            ('sample'        , '-sample'        , '19'       ),
                            ('catl_type'     , '-abopt'         , 'mr'       ),
                            ('cosmo_choice'  , '-cosmo'         , 'LasDamas' ),
                            ('nmin'          , '-nmin'          , 2          ),
                            ('mass_factor'   , '-mass_factor'   , 10         ),
                            ('n_predict'     , '-n_predict'     , 1          ),
                            ('shuffle_opt'   , '-shuffle_opt'   , True       ),
                            ('dropna_opt'    , '-dropna_opt'    , True       ),
                            ('pre_opt'       , '-pre_opt'       , 'standard' ),
                            ('test_train_opt', '-test_train_opt', 'boxes_n'  ),
                            ('box_idx'       , '-box_idx'       , '0_4_5'    ),
                            ('box_test'      , '-box_test'      , 0          ),
                            ('sample_frac'   , '-sample_frac'   , 0.01       ),
                            ('test_size'     , '-test_size'     , 0.25       ),
                            ('n_feat_use'    , '-n_feat_use'    , 'sub'      ),
                            ('dens_calc'     , '-dens_calc'     , True       ),
                            ('kf_splits'     , '-kf_splits'     , 3          ),
                            ('hidden_layers' , '-hidden_layers' , 3          ),
                            ('unit_layer'    , '-unit_layer'    , 100        ),
                            ('score_method'  , '-score_method'  , 'threshold'),
                            ('threshold'     , '-threshold'     , 0.1        ),
                            ('perc_val'      , '-perc_val'      , 0.68       ),
                            ('sample_method' , '-sample_method' , 'binning'  ),
                            ('bin_val'       , '-bin_val'       , 'fixed'    ),
                            ('ml_analysis'   , '-ml_analysis'   , 'hod_fixed'),
                            ('resample_opt'  , '-resample_opt'  , 'under'    ),
                            ('include_nn'    , '-include_nn'    , False      ),
                            ('cpu_frac'      , '-cpu'           , 0.75       ),
                            ('remove_files'  , '-remove'        , False      ),
                            ('verbose'       , '-v'             , False      ),
                            ('perf_opt'      , '-perf'          , False      ),
                            ('seed'          , '-seed'          , 1         )])
    ##
    ## Converting to pandas DataFrame
    colnames = ['Name', 'Flag', 'Value']
    main_df = pd.DataFrame(catl_params_main_arr, columns=colnames)
    ##
    ## Sorting out DataFrame by `name`
    main_df = main_df.sort_values(by='Name')
    main_df.reset_index(inplace=True, drop=True)
    ##
    ## HOD Model to use
    main_df = df_value_modifier(main_df, 'hod_n', param_dict)
    ##
    ## Different models of DV to use for comparison
    main_df = df_value_modifier(main_df, 'dv_models_n', param_dict)
    ##
    ## Type of dark matter halo to use in the simulation
    main_df = df_value_modifier(main_df, 'halotype', param_dict)
    ##
    ## CLF Method for assigning galaxy properties
    main_df = df_value_modifier(main_df, 'clf_method', param_dict)
    ##
    ## Random seed used during the CLF assignment
    main_df = df_value_modifier(main_df, 'clf_seed', param_dict)
    ##
    ## Difference between galaxy and mass velocity profiles
    main_df = df_value_modifier(main_df, 'dv', param_dict)
    ##
    ## SDSS luminosity sample to analyze
    main_df = df_value_modifier(main_df, 'sample', param_dict)
    ##
    ## Type of Abundance matching
    main_df = df_value_modifier(main_df, 'catl_type', param_dict)
    ##
    ## Cosmology choice
    main_df = df_value_modifier(main_df, 'cosmo_choice', param_dict)
    ##
    ## Minimum number of galaxies in a group
    main_df = df_value_modifier(main_df, 'nmin', param_dict)
    ##
    ## Total number of properties to predict. Default = 1
    main_df = df_value_modifier(main_df, 'n_predict', param_dict)
    ##
    ## Option for shuffling dataset when creating `testing` and `training`
    ## datasets
    main_df = df_value_modifier(main_df, 'shuffle_opt', param_dict)
    ##
    ## Option for Shuffling dataset when separing `training` and `testing` sets
    main_df = df_value_modifier(main_df, 'dropna_opt', param_dict)
    ##
    ## Option for which preprocessing of the data to use.
    main_df = df_value_modifier(main_df, 'pre_opt', param_dict)
    ##
    ## Option for which kind of separation of training/testing to use for the 
    ## datasets.
    main_df = df_value_modifier(main_df, 'test_train_opt', param_dict)
    ##
    ## Initial and final indices of the simulation boxes to use for the 
    ## testing and training datasets.
    main_df = df_value_modifier(main_df, 'box_idx', param_dict)
    ##
    ## Index of the simulation box to use for the `training` and `testing
    main_df = df_value_modifier(main_df, 'box_test', param_dict)
    ##
    ## Fraction of the sample to be used.
    ## Only if `test_train_opt == 'sample_frac'`
    main_df = df_value_modifier(main_df, 'sample_frac', param_dict)
    ##
    ## Testing size for ML
    main_df = df_value_modifier(main_df, 'test_size', param_dict)
    ##
    ## Option for using all features or just a few
    main_df = df_value_modifier(main_df, 'n_feat_use', param_dict)
    ##
    ## Option for calculating densities or not
    main_df = df_value_modifier(main_df, 'dens_calc', param_dict)
    ##
    ## Number of hidden layers to use for neural network
    main_df = df_value_modifier(main_df, 'hidden_layers', param_dict)
    ##
    ## Number of units per hidden layer for the neural. network.
    ## Default = `100`.
    main_df = df_value_modifier(main_df, 'unit_layer', param_dict)
    ##
    ## Option for determining which scoring method to use.
    main_df = df_value_modifier(main_df, 'score_method', param_dict)
    ##
    ## Threshold value used for when ``score_method == 'threshold'``
    main_df = df_value_modifier(main_df, 'threshold', param_dict)
    ##
    ## Percentage value used for when ``score_method == 'perc'``
    main_df = df_value_modifier(main_df, 'perc_val', param_dict)
    ##
    ## Method for binning or sumsample the array of the estimated group mass.
    main_df = df_value_modifier(main_df, 'sample_method', param_dict)
    ##
    ## Type of binning to use for the mass
    main_df = df_value_modifier(main_df, 'bin_val', param_dict)
    ##
    ## Type of analysis to perform.
    main_df = df_value_modifier(main_df, 'ml_analysis', param_dict)
    ##
    ## Type of resampling to use if necessary
    main_df = df_value_modifier(main_df, 'resample_opt', param_dict)
    ##
    ## Percentage of CPU to use
    main_df = df_value_modifier(main_df, 'cpu_frac', param_dict)
    ##
    ## Option for removing files or not
    main_df = df_value_modifier(main_df, 'remove_files', param_dict)
    ##
    ## Option for displaying outputs or not
    main_df = df_value_modifier(main_df, 'verbose', param_dict)
    ##
    ## Option for looking at `perfect` mock catalogues
    main_df = df_value_modifier(main_df, 'perf_opt', param_dict)
    ##
    ## Random seed for the analysis
    main_df = df_value_modifier(main_df, 'seed', param_dict)
    ##
    ## --------------------------------------------------------------------- ##
    ## Calculation for the ML analysis predictions - Parameters - Plot
    ## --------------------------------------------------------------------- ##
    catl_params_main_plot_arr = num.array([
                            ('hod_n'         , '-hod_model_n'   , 0          ),
                            ('dv_models_n'   , '-dv_models_n'   , '0.9_0.925_0.95_0.975_1.0_1.025_1.05_1.10'),
                            ('halotype'      , '-halotype'      , 'so'       ),
                            ('clf_method'    , '-clf_method'    , 1          ),
                            ('dv'            , '-dv'            , 1.0        ),
                            ('clf_seed'      , '-clf_seed'      , 1235       ),
                            ('sample'        , '-sample'        , '19'       ),
                            ('catl_type'     , '-abopt'         , 'mr'       ),
                            ('cosmo_choice'  , '-cosmo'         , 'LasDamas' ),
                            ('nmin'          , '-nmin'          , 2          ),
                            ('mass_factor'   , '-mass_factor'   , 10         ),
                            ('n_predict'     , '-n_predict'     , 1          ),
                            ('shuffle_opt'   , '-shuffle_opt'   , True       ),
                            ('dropna_opt'    , '-dropna_opt'    , True       ),
                            ('pre_opt'       , '-pre_opt'       , 'standard' ),
                            ('test_train_opt', '-test_train_opt', 'boxes_n'  ),
                            ('box_idx'       , '-box_idx'       , '0_4_5'    ),
                            ('box_test'      , '-box_test'      , 0          ),
                            ('sample_frac'   , '-sample_frac'   , 0.01       ),
                            ('test_size'     , '-test_size'     , 0.25       ),
                            ('n_feat_use'    , '-n_feat_use'    , 'sub'      ),
                            ('dens_calc'     , '-dens_calc'     , True       ),
                            ('kf_splits'     , '-kf_splits'     , 3          ),
                            ('hidden_layers' , '-hidden_layers' , 3          ),
                            ('unit_layer'    , '-unit_layer'    , 100        ),
                            ('score_method'  , '-score_method'  , 'threshold'),
                            ('threshold'     , '-threshold'     , 0.1        ),
                            ('perc_val'      , '-perc_val'      , 0.68       ),
                            ('sample_method' , '-sample_method' , 'binning'  ),
                            ('bin_val'       , '-bin_val'       , 'fixed'    ),
                            ('ml_analysis'   , '-ml_analysis'   , 'hod_fixed'),
                            ('resample_opt'  , '-resample_opt'  , 'under'    ),
                            ('include_nn'    , '-include_nn'    , False      ),
                            ('plot_opt'      , '-plot_opt'      , 'mhalo'    ),
                            ('rank_opt'      , '-rank_opt'      , 'idx'      ),
                            ('chosen_ml_alg' , '-chosen_ml_alg' , 'xgboost'  ),
                            ('cpu_frac'      , '-cpu'           , 0.75       ),
                            ('remove_files'  , '-remove'        , False      ),
                            ('verbose'       , '-v'             , False      ),
                            ('perf_opt'      , '-perf'          , False      ),
                            ('seed'          , '-seed'          , 1         )])
    ##
    ## Converting to pandas DataFrame
    colnames = ['Name', 'Flag', 'Value']
    main_plot_df = pd.DataFrame(catl_params_main_plot_arr, columns=colnames)
    ##
    ## Sorting out DataFrame by `name`
    main_plot_df = main_plot_df.sort_values(by='Name')
    main_plot_df.reset_index(inplace=True, drop=True)
    ##
    ## HOD Model to use
    main_plot_df = df_value_modifier(main_plot_df, 'hod_n', param_dict)
    ##
    ## Different models of DV to use for comparison
    main_plot_df = df_value_modifier(main_plot_df, 'dv_models_n', param_dict)
    ##
    ## Type of dark matter halo to use in the simulation
    main_plot_df = df_value_modifier(main_plot_df, 'halotype', param_dict)
    ##
    ## CLF Method for assigning galaxy properties
    main_plot_df = df_value_modifier(main_plot_df, 'clf_method', param_dict)
    ##
    ## Random seed used during the CLF assignment
    main_plot_df = df_value_modifier(main_plot_df, 'clf_seed', param_dict)
    ##
    ## Difference between galaxy and mass velocity profiles
    main_plot_df = df_value_modifier(main_plot_df, 'dv', param_dict)
    ##
    ## SDSS luminosity sample to analyze
    main_plot_df = df_value_modifier(main_plot_df, 'sample', param_dict)
    ##
    ## Type of Abundance matching
    main_plot_df = df_value_modifier(main_plot_df, 'catl_type', param_dict)
    ##
    ## Cosmology choice
    main_plot_df = df_value_modifier(main_plot_df, 'cosmo_choice', param_dict)
    ##
    ## Minimum number of galaxies in a group
    main_plot_df = df_value_modifier(main_plot_df, 'nmin', param_dict)
    ##
    ## Total number of properties to predict. Default = 1
    main_plot_df = df_value_modifier(main_plot_df, 'n_predict', param_dict)
    ##
    ## Option for shuffling dataset when creating `testing` and `training`
    ## datasets
    main_plot_df = df_value_modifier(main_plot_df, 'shuffle_opt', param_dict)
    ##
    ## Option for Shuffling dataset when separing `training` and `testing` sets
    main_plot_df = df_value_modifier(main_plot_df, 'dropna_opt', param_dict)
    ##
    ## Option for which preprocessing of the data to use.
    main_plot_df = df_value_modifier(main_plot_df, 'pre_opt', param_dict)
    ##
    ## Option for which kind of separation of training/testing to use for the 
    ## datasets.
    main_plot_df = df_value_modifier(main_plot_df, 'test_train_opt', param_dict)
    ##
    ## Initial and final indices of the simulation boxes to use for the 
    ## testing and training datasets.
    main_plot_df = df_value_modifier(main_plot_df, 'box_idx', param_dict)
    ##
    ## Index of the simulation box to use for the `training` and `testing
    main_plot_df = df_value_modifier(main_plot_df, 'box_test', param_dict)
    ##
    ## Fraction of the sample to be used.
    ## Only if `test_train_opt == 'sample_frac'`
    main_plot_df = df_value_modifier(main_plot_df, 'sample_frac', param_dict)
    ##
    ## Testing size for ML
    main_plot_df = df_value_modifier(main_plot_df, 'test_size', param_dict)
    ##
    ## Option for using all features or just a few
    main_plot_df = df_value_modifier(main_plot_df, 'n_feat_use', param_dict)
    ##
    ## Option for calculating densities or not
    main_plot_df = df_value_modifier(main_plot_df, 'dens_calc', param_dict)
    ##
    ## Number of hidden layers to use for neural network
    main_plot_df = df_value_modifier(main_plot_df, 'hidden_layers', param_dict)
    ##
    ## Number of units per hidden layer for the neural. network.
    ## Default = `100`.
    main_plot_df = df_value_modifier(main_plot_df, 'unit_layer', param_dict)
    ##
    ## Option for determining which scoring method to use.
    main_plot_df = df_value_modifier(main_plot_df, 'score_method', param_dict)
    ##
    ## Threshold value used for when ``score_method == 'threshold'``
    main_plot_df = df_value_modifier(main_plot_df, 'threshold', param_dict)
    ##
    ## Percentage value used for when ``score_method == 'perc'``
    main_plot_df = df_value_modifier(main_plot_df, 'perc_val', param_dict)
    ##
    ## Method for binning or sumsample the array of the estimated group mass.
    main_plot_df = df_value_modifier(main_plot_df, 'sample_method', param_dict)
    ##
    ## Type of binning to use for the mass
    main_plot_df = df_value_modifier(main_plot_df, 'bin_val', param_dict)
    ##
    ## Type of analysis to perform.
    main_plot_df = df_value_modifier(main_plot_df, 'ml_analysis', param_dict)
    ##
    ## Option for which variable to plot on x-axis
    main_plot_df = df_value_modifier(main_plot_df, 'plot_opt', param_dict)
    ##
    ## Option for which type of ranking to plot
    main_plot_df = df_value_modifier(main_plot_df, 'rank_opt', param_dict)
    ##
    ## Option for which ML algorithm to plot
    main_plot_df = df_value_modifier(main_plot_df, 'chosen_ml_alg', param_dict)
    ##
    ## Type of resampling to use if necessary
    main_plot_df = df_value_modifier(main_plot_df, 'resample_opt', param_dict)
    ##
    ## Option to include results from neural network or not
    main_plot_df = df_value_modifier(main_plot_df, 'include_nn', param_dict)
    ##
    ## Percentage of CPU to use
    main_plot_df = df_value_modifier(main_plot_df, 'cpu_frac', param_dict)
    ##
    ## Option for removing files or not
    main_plot_df = df_value_modifier(main_plot_df, 'remove_files', param_dict)
    ##
    ## Option for displaying outputs or not
    main_plot_df = df_value_modifier(main_plot_df, 'verbose', param_dict)
    ##
    ## Option for looking at `perfect` mock catalogues
    main_plot_df = df_value_modifier(main_plot_df, 'perf_opt', param_dict)
    ##
    ## Random seed for the analysis
    main_plot_df = df_value_modifier(main_plot_df, 'seed', param_dict)

    return [main_df, main_plot_df]

### --------------- Executing script --------------- ###

def get_exec_string(df_arr, param_dict):
    """
    Produces the string to be executed in the bash file.
    It also concatenates the strings to each of the elements in `df_arr`

    Parameters
    -----------
    df_arr : array-like
        List of DataFrames that will be used to create the `main` string that
        will get executed.

    param_dict: python dictionary
        dictionary with project variables

    Returns
    -----------
    main_str_cmd: `str`
        String that will get executed along the main file.
    """
    ##
    ## Current working directory
    working_dir = os.path.abspath(os.path.dirname(__file__))
    # Maing string
    main_str_cmd = ''
    ## Creating main string
    for ii, df_ii in enumerate(df_arr):
        # Name of the file to get executed
        catl_makefile_ii = param_dict['run_file_dict'][ii]['file']
        ## Getting the filepath to `catl_makefile_ii`
        catl_makefile_ii_path = os.path.join(working_dir, catl_makefile_ii)
        ## Checking if file exists
        if not (os.path.exists(catl_makefile_ii_path)):
            msg = '{0} `catl_makefile_ii_path` ({1}) does not exists!'.format(
                param_dict['Prog_msg'], catl_makefile_ii_path)
            raise FileNotFoundError(msg)
        ##
        ## Constructing ith string
        catl_ii_str = 'python {0} '.format(catl_makefile_ii_path)
        for ii in range(df_ii.shape[0]):
            # Appending to string
            catl_ii_str += ' {0} {1}'.format(    df_ii['Flag'][ii],
                                                df_ii['Value'][ii])
        ## Appending a comma at the end
        catl_ii_str += '; '
        ## Appending to main string command `main_str_cmd`
        main_str_cmd += catl_ii_str

    return main_str_cmd

def project_const(param_dict):
    """
    Contains the name of variables used in the project

    Parameters
    -----------
    param_dict: python dictionary
        dictionary with project variables

    Returns
    ----------
    param_dict: python dictionary
        dictionary with project variables
    """
    ## Constants
    # Environment name
    env_name        = 'sdss_groups_ml'
    ##
    ## Choosing the script(s) that will be ran
    ##
    ## Fixed HOD and DV value - Default
    if (param_dict['ml_analysis'] == 'hod_dv_fixed'):
        window_name     = 'SDSS_ML_DA_fixed_hodn_{0}_dv_{1}'.format(
            param_dict['hod_n'], param_dict['dv'])
    ##
    ## Fixed DV and alternating HOD models
    if (param_dict['ml_analysis'] == 'dv_fixed'):
        window_name     = 'SDSS_ML_DA_fixed_hodn_{0}_dv_{1}'.format(
            param_dict['hod_models_n'], param_dict['dv'])
    ##
    ## Fixed HOD and alternating DV models
    if (param_dict['ml_analysis'] == 'hod_fixed'):
        window_name     = 'SDSS_ML_DA_fixed_dv_{0}_dv_{1}'.format(
            param_dict['dv_models_n'], param_dict['dv'])
    ##
    ## File or files to run
    sub_window_name = 'DA'
    file_exe_name   = 'catl_data_analysis_{0}_run.sh'.format(
                        param_dict['ml_analysis'])
    ##
    ## Fixed HOD and DV
    if (param_dict['ml_analysis'] == 'hod_dv_fixed'):
        run_file_dict    = {}
        run_file_dict[0] = {'file': 'catl_algorithm_comparison.py'}
        run_file_dict[1] = {'file': 'catl_algorithm_comparison_plots.py'}
    ## Fixed DV and multiple HOD models
    if (param_dict['ml_analysis'] == 'dv_fixed'):
        run_file_dict    = {}
        run_file_dict[0] = {'file': 'catl_hod_diff_comparison.py'}
        run_file_dict[1] = {'file': 'catl_hod_diff_comparison_plots.py'}
    ## Fixed HOD and multiple DV models
    if (param_dict['ml_analysis'] == 'hod_fixed'):
        run_file_dict    = {}
        run_file_dict[0] = {'file': 'catl_velocity_bias_diff_comparison.py'}
        run_file_dict[1] = {'file': 'catl_velocity_bias_diff_comparison_plots.py'}
    ##
    ## Saving to main dictionary
    param_dict['env_name'       ] = env_name
    param_dict['window_name'    ] = window_name
    param_dict['sub_window_name'] = sub_window_name
    param_dict['file_exe_name'  ] = file_exe_name
    param_dict['run_file_dict'  ] = run_file_dict

    return param_dict

def file_construction_and_execution(df_arr, param_dict, str_interval=200):
    """
    1) Creates file that has shell commands to run executable
    2) Executes the file, which creates a screen session with the executables

    Parameters:
    -----------
    params_pd: pandas DataFrame
        DataFrame with necessary parameters to run `sdss mocks create`

    param_dict: python dictionary
        dictionary with project variables

    str_interval : `float`, optional
        Maximum length of the string to send at once. This variable is set to
        `200` by default.
    """
    ##
    ## Getting today's date
    now_str = datetime.datetime.now().strftime("%x %X")
    ##
    ## Obtain MCF strings
    main_str_cmd = get_exec_string(df_arr, param_dict)
    ##
    ## Parsing text that will go in file
    # Working directory
    working_dir = os.path.abspath(os.path.dirname(__file__))
    ## Obtaining path to file
    outfile_name = param_dict['file_exe_name']
    outfile_path = os.path.join(working_dir, outfile_name)
    ##
    ## Splitting main command
    if len(main_str_cmd) > str_interval:
        main_str_cmd_mod = [main_str_cmd[i:i+str_interval] for i in range(0, len(main_str_cmd), str_interval)]
    else:
        main_str_cmd_mod = [main_str_cmd]
    ##
    ## Opening file
    with open(outfile_path, 'wb') as out_f:
        out_f.write(b"""#!/usr/bin/env bash\n\n""")
        out_f.write( """## Author: {0}\n\n""".format(param_dict['setup_cfg']['author']).encode())
        out_f.write( """## Last Edited: {0}\n\n""".format(now_str).encode())
        out_f.write(b"""### --- Variables\n""")
        out_f.write( """ENV_NAME="{0}"\n""".format(param_dict['env_name']).encode())
        out_f.write( """WINDOW_NAME="{0}"\n""".format(param_dict['window_name']).encode())
        out_f.write( """SUB_WINDOW="{0}"\n""".format(param_dict['sub_window_name']).encode())
        out_f.write(b"""# Home Directory\n""")
        out_f.write(b"""home_dir=`eval echo "~$different_user"`\n""")
        out_f.write(b"""# Type of OS\n""")
        out_f.write(b"""ostype=`uname`\n""")
        out_f.write(b"""# Sourcing profile\n""")
        out_f.write(b"""if [[ $ostype == "Linux" ]]; then\n""")
        out_f.write(b"""    source $home_dir/.bashrc\n""")
        out_f.write(b"""else\n""")
        out_f.write(b"""    source $home_dir/.bash_profile\n""")
        out_f.write(b"""fi\n""")
        out_f.write(b"""# Activating Environment\n""")
        out_f.write(b"""source activate ${ENV_NAME}\n""")
        out_f.write(b"""###\n""")
        out_f.write(b"""### --- Python Strings\n""")
        if (len(main_str_cmd_mod)== 1):
            out_f.write( """SCRIPT_CMD="{0}"\n""".format(main_str_cmd).encode())
        else:
            for kk, cmd_kk in enumerate(main_str_cmd_mod):
                out_f.write( """SCRIPT_CMD_{0}="{1}"\n""".format(kk, cmd_kk).encode())
        out_f.write(b"""###\n""")
        out_f.write(b"""### --- Deleting previous Screen Session\n""")
        out_f.write(b"""screen -S ${WINDOW_NAME} -X quit\n""")
        out_f.write(b"""###\n""")
        out_f.write(b"""### --- Screen Session\n""")
        out_f.write(b"""screen -mdS ${WINDOW_NAME}\n""")
        out_f.write(b"""## Mocks\n""")
        out_f.write(b"""screen -S ${WINDOW_NAME} -X screen -t ${SUB_WINDOW}\n""")
        out_f.write(b"""screen -S ${WINDOW_NAME} -p ${SUB_WINDOW} -X stuff $"conda deactivate;"\n""")
        out_f.write(b"""screen -S ${WINDOW_NAME} -p ${SUB_WINDOW} -X stuff $'\\n'\n""")
        out_f.write(b"""screen -S ${WINDOW_NAME} -p ${SUB_WINDOW} -X stuff $"conda activate ${ENV_NAME};"\n""")
        if (len(main_str_cmd_mod)== 1):
            out_f.write(b"""screen -S ${WINDOW_NAME} -p ${SUB_WINDOW} -X stuff $"${SCRIPT_CMD}"\n""")
        else:
            for kk, cmd_kk in enumerate(main_str_cmd_mod):
                out_f.write(("""screen -S ${WINDOW_NAME} -p ${SUB_WINDOW} -X stuff $"${SCRIPT_CMD_%s}"\n""" %(kk)).encode())
        out_f.write(b"""screen -S ${WINDOW_NAME} -p ${SUB_WINDOW} -X stuff $'\\n'\n""")
        out_f.write(b"""\n""")
    ##
    ## Check if File exists
    if os.path.isfile(outfile_path):
        pass
    else:
        msg = '{0} `outfile_path` ({1}) not found!! Exiting!'.format(
            param_dict['Prog_msg'], outfile_path)
        raise ValueError(msg)
    ##
    ## Make file executable
    print(".>>> Making file executable....")
    print("     chmod +x {0}".format(outfile_path))
    os.system("chmod +x {0}".format(outfile_path))
    ##
    ## Running script
    print(".>>> Running Script...")
    os.system("{0}".format(outfile_path))

def main(args):
    """
    Computes the analysis and
    """
    ## Reading all elements and converting to python dictionary
    param_dict = vars(args)
    ##
    ## Extra arguments
    param_dict = add_to_dict(param_dict)
    ##
    ## Parameters for the analysis
    if (param_dict['ml_analysis'] == 'hod_dv_fixed'):
        df_arr = get_analysis_alg_comp_params(param_dict)
    elif (param_dict['ml_analysis'] == 'dv_fixed'):
        df_arr = get_analysis_hod_diff_params(param_dict)
    elif (param_dict['ml_analysis'] == 'hod_fixed'):
        df_arr = get_analysis_dv_diff_params(param_dict)
    ##
    ## Running analysis
    file_construction_and_execution(df_arr, param_dict)

# Main function
if __name__ == '__main__':
    ## Input arguments
    args = get_parser()
    # Main Function
    main(args)
