#! /usr/bin/env python
# -*- coding: utf-8 -*-

# Victor Calderon
# Created      : 2018-05-27
# Last Modified: 2018-05-28
# Vanderbilt University
from __future__ import print_function, division, absolute_import
__author__     =['Victor Calderon']
__copyright__  =["Copyright 2017 Victor Calderon, SDSS Mocks Create - Make"]
__email__      =['victor.calderon@vanderbilt.edu']
__maintainer__ =['Victor Calderon']
"""
Main script to run the data preprocessing pipeline.
The pipeline includes:
    - Calculates the features for each galaxy/group mock catalogues
    - Transforms the features into proper inputs for the ML algoriths
    - Saves the features arrays as pickle files
"""

# Path to Custom Utilities folder
import os
import sys
import git

# Importing Modules
from cosmo_utils.utils import file_utils as cfutils
from cosmo_utils.utils import work_paths as cwpaths

import numpy as num
import os
import sys
import pandas as pd

# Extra-modules
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
    description_msg =   """
    Main script to run the data preprocessing pipeline.
    The pipeline includes:
        - Calculates the features for each galaxy/group mock catalogues
        - Transforms the features into proper inputs for the ML algoriths
        - Saves the features arrays as pickle files
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
                        default=1)
    ## Difference between galaxy and mass velocity profiles (v_g-v_c)/(v_m-v_c)
    parser.add_argument('-dv',
                        dest='dv',
                        help="""
                        Difference between galaxy and mass velocity profiles 
                        (v_g-v_c)/(v_m-v_c)
                        """,
                        type=_check_pos_val,
                        default=1.0)
    ## Random Seed for CLF
    parser.add_argument('-clf_seed',
                        dest='clf_seed',
                        help='Random seed to be used for CLF',
                        type=int,
                        metavar='[0-4294967295]',
                        default=1235)
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
    ## Minimum of galaxies in a group
    parser.add_argument('-mass_factor',
                        dest='mass_factor',
                        help="""
                        Factor by which to evaluate the distance to closest 
                        cluster""",
                        type=int,
                        choices=range(2,100),
                        metavar='[2-100]',
                        default=10)
    ## Removing group for when determining density
    parser.add_argument('-remove_group',
                        dest='remove_group',
                        help="""
                        Option for removing the group when calculating 
                        densities at different radii""",
                        type=_str2bool,
                        default=True)
    ## Option for calculating densities or not
    parser.add_argument('-dens_calc',
                        dest='dens_calc',
                        help='Option for calculating densities.',
                        type=_str2bool,
                        default=True)
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
    ## Option for which kind of separation of training/testing to use for the 
    ## datasets.
    parser.add_argument('-test_train_opt',
                        dest='test_train_opt',
                        help="""
                        Option for which kind of separation of training/testing
                        to use for the datasets.
                        """,
                        type=str,
                        choices=['sample_frac', 'boxes_n'],
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
                        nargs=str,
                        default='0_4_5')
    ## Fraction of the sample to be used.
    ## Only if `test_train_opt == 'sample_frac'`
    ## Fraction of the sample to use
    parser.add_argument('-sample_frac',
                        dest='sample_frac',
                        help='fraction of the total dataset to use',
                        type=float,
                        default=0.01)
    ## Testing size for ML
    parser.add_argument('-test_size',
                        dest='test_size',
                        help='Percentage size of the catalogue used for testing',
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
    ## CPU Counts
    parser.add_argument('-cpu',
                        dest='cpu_frac',
                        help='Fraction of total number of CPUs to use',
                        type=float,
                        default=0.75)
    ## Option for removing file
    parser.add_argument('-remove',
                        dest='remove_files',
                        help='Removed main files ',
                        type=_str2bool,
                        default=False)
    ## Verbose
    parser.add_argument('-v','--verbose',
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

def get_analysis_params(param_dict):
    """
    Parameters for the data pre-processing step, before training and 
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
    ## Catalogue Feature Calculatiion - Parameters
    catl_feat_arr = num.array([
                            ('hod_n'       , '-hod_model_n' , 0         ),
                            ('halotype'    , '-halotype'    , 'so'      ),
                            ('clf_method'  , '-clf_method'  , 1         ),
                            ('dv'          , '-dv'          , 1.0       ),
                            ('clf_seed'    , '-clf_seed'    , 1235      ),
                            ('sample'      , '-sample'      , '19'      ),
                            ('catl_type'   , '-abopt'       , 'mr'      ),
                            ('cosmo_choice', '-cosmo'       , 'LasDamas'),
                            ('nmin'        , '-nmin'        , 2         ),
                            ('mass_factor' , '-mass_factor' , 10        ),
                            ('remove_group', '-remove_group', True      ),
                            ('dens_calc'   , '-dens_calc'   , True      ),
                            ('cpu_frac'    , '-cpu'         , 0.75      ),
                            ('remove_files', '-remove'      , False     ),
                            ('verbose'     , '-v'           , False     ),
                            ('perf_opt'    , '-perf'        , False     ),
                            ('seed'        , '-seed'        , 1         )])
    ##
    ## Converting to pandas DataFrame
    colnames = ['Name','Flag','Value']
    catl_feat_df = pd.DataFrame(catl_feat_arr, columns=colnames)
    ##
    ## Sorting out DataFrame by `name`
    catl_feat_df = catl_feat_df.sort_values(by='Name')
    catl_feat_df.reset_index(inplace=True, drop=True)
    ##
    ## HOD Model to use
    catl_feat_df = df_value_modifier(catl_feat_df, 'hod_n', param_dict)
    ##
    ## Type of dark matter halo to use in the simulation
    catl_feat_df = df_value_modifier(catl_feat_df, 'halotype', param_dict)
    ##
    ## CLF Method for assigning galaxy properties
    catl_feat_df = df_value_modifier(catl_feat_df, 'clf_method', param_dict)
    ##
    ## Random seed used during the CLF assignment
    catl_feat_df = df_value_modifier(catl_feat_df, 'clf_seed', param_dict)
    ##
    ## Difference between galaxy and mass velocity profiles
    catl_feat_df = df_value_modifier(catl_feat_df, 'dv', param_dict)
    ##
    ## SDSS luminosity sample to analyze
    catl_feat_df = df_value_modifier(catl_feat_df, 'sample', param_dict)
    ##
    ## Type of Abundance matching
    catl_feat_df = df_value_modifier(catl_feat_df, 'catl_type', param_dict)
    ##
    ## Cosmology choice
    catl_feat_df = df_value_modifier(catl_feat_df, 'cosmo_choice', param_dict)
    ##
    ## Minimum number of galaxies in a group
    catl_feat_df = df_value_modifier(catl_feat_df, 'nmin', param_dict)
    ##
    ## Factor by which to evaluate the distance to closest cluster.
    catl_feat_df = df_value_modifier(catl_feat_df, 'mass_factor', param_dict)
    ##
    ## Removing group for when determining density or not
    catl_feat_df = df_value_modifier(catl_feat_df, 'remove_group', param_dict)
    ##
    ## Option for calculating densities or not
    catl_feat_df = df_value_modifier(catl_feat_df, 'dens_calc', param_dict)
    ##
    ## Percentage of CPU to use
    catl_feat_df = df_value_modifier(catl_feat_df, 'cpu_frac', param_dict)
    ##
    ## Option for removing files or not
    catl_feat_df = df_value_modifier(catl_feat_df, 'remove_files', param_dict)
    ##
    ## Option for displaying outputs or not
    catl_feat_df = df_value_modifier(catl_feat_df, 'verbose', param_dict)
    ##
    ## Option for looking at `perfect` mock catalogues
    catl_feat_df = df_value_modifier(catl_feat_df, 'perf_opt', param_dict)
    ##
    ## Random seed for the analysis
    catl_feat_df = df_value_modifier(catl_feat_df, 'seed', param_dict)
    ##
    ## --------------------------------------------------------------------- ##
    ## Catalogue Feature Processing - Parameters
    ## --------------------------------------------------------------------- ##
    feat_proc_arr = num.array([
                            ('hod_n'         , '-hod_model_n'   , 0         ),
                            ('halotype'      , '-halotype'      , 'so'      ),
                            ('clf_method'    , '-clf_method'    , 1         ),
                            ('dv'            , '-dv'            , 1.0       ),
                            ('clf_seed'      , '-clf_seed'      , 1235      ),
                            ('sample'        , '-sample'        , '19'      ),
                            ('catl_type'     , '-abopt'         , 'mr'      ),
                            ('cosmo_choice'  , '-cosmo'         , 'LasDamas'),
                            ('nmin'          , '-nmin'          , 2         ),
                            ('n_predict'     , '-n_predict'     , 1         ),
                            ('shuffle_opt'   , '-shuffle_opt'   , True      ),
                            ('dropna_opt'    , '-dropna_opt'    , True      ),
                            ('pre_opt'       , '-pre_opt'       , 'standard'),
                            ('test_train_opt', '-test_train_opt', 'boxes_n' ),
                            ('box_idx'       , '-box_idx'       , '0_4_5'   ),
                            ('sample_frac'   , '-sample_frac'   , 0.01      ),
                            ('test_size'     , '-test_size'     , 0.25      ),
                            ('n_feat_use'    , '-n_feat_use'    , 'sub'     ),
                            ('dens_calc'     , '-dens_calc'     , True      ),
                            ('cpu_frac'      , '-cpu'           , 0.75      ),
                            ('remove_files'  , '-remove'        , False     ),
                            ('verbose'       , '-v'             , False     ),
                            ('perf_opt'      , '-perf'          , False     ),
                            ('seed'          , '-seed'          , 1         )])
    ##
    ## Converting to pandas DataFrame
    colnames = ['Name','Flag','Value']
    feat_proc_df = pd.DataFrame(feat_proc_arr, columns=colnames)
    ##
    ## Sorting out DataFrame by `name`
    feat_proc_df = feat_proc_df.sort_values(by='Name')
    feat_proc_df.reset_index(inplace=True, drop=True)
    ##
    ## HOD Model to use
    feat_proc_df = df_value_modifier(feat_proc_df, 'hod_n', param_dict)
    ##
    ## Type of dark matter halo to use in the simulation
    feat_proc_df = df_value_modifier(feat_proc_df, 'halotype', param_dict)
    ##
    ## CLF Method for assigning galaxy properties
    feat_proc_df = df_value_modifier(feat_proc_df, 'clf_method', param_dict)
    ##
    ## Random seed used during the CLF assignment
    feat_proc_df = df_value_modifier(feat_proc_df, 'clf_seed', param_dict)
    ##
    ## Difference between galaxy and mass velocity profiles
    feat_proc_df = df_value_modifier(feat_proc_df, 'dv', param_dict)
    ##
    ## SDSS luminosity sample to analyze
    feat_proc_df = df_value_modifier(feat_proc_df, 'sample', param_dict)
    ##
    ## Type of Abundance matching
    feat_proc_df = df_value_modifier(feat_proc_df, 'catl_type', param_dict)
    ##
    ## Cosmology choice
    feat_proc_df = df_value_modifier(feat_proc_df, 'cosmo_choice', param_dict)
    ##
    ## Minimum number of galaxies in a group
    feat_proc_df = df_value_modifier(feat_proc_df, 'nmin', param_dict)
    ##
    ## Total number of properties to predict. Default = 1
    feat_proc_df = df_value_modifier(feat_proc_df, 'n_predict', param_dict)
    ##
    ## Option for shuffling dataset when creating `testing` and `training`
    ## datasets
    feat_proc_df = df_value_modifier(feat_proc_df, 'shuffle_opt', param_dict)
    ##
    ## Option for Shuffling dataset when separing `training` and `testing` sets
    feat_proc_df = df_value_modifier(feat_proc_df, 'dropna_opt', param_dict)
    ##
    ## Option for which preprocessing of the data to use.
    feat_proc_df = df_value_modifier(feat_proc_df, 'pre_opt', param_dict)
    ##
    ## Option for which kind of separation of training/testing to use for the 
    ## datasets.
    feat_proc_df = df_value_modifier(feat_proc_df, 'test_train_opt', param_dict)
    ##
    ## Initial and final indices of the simulation boxes to use for the 
    ## testing and training datasets.
    feat_proc_df = df_value_modifier(feat_proc_df, 'box_idx', param_dict)
    ##
    ## Fraction of the sample to be used.
    ## Only if `test_train_opt == 'sample_frac'`
    feat_proc_df = df_value_modifier(feat_proc_df, 'sample_frac', param_dict)
    ##
    ## Testing size for ML
    feat_proc_df = df_value_modifier(feat_proc_df, 'test_size', param_dict)
    ##
    ## Option for using all features or just a few
    feat_proc_df = df_value_modifier(feat_proc_df, 'n_feat_use', param_dict)
    ##
    ## Option for calculating densities or not
    feat_proc_df = df_value_modifier(feat_proc_df, 'dens_calc', param_dict)
    ##
    ## Percentage of CPU to use
    feat_proc_df = df_value_modifier(feat_proc_df, 'cpu_frac', param_dict)
    ##
    ## Option for removing files or not
    feat_proc_df = df_value_modifier(feat_proc_df, 'remove_files', param_dict)
    ##
    ## Option for displaying outputs or not
    feat_proc_df = df_value_modifier(feat_proc_df, 'verbose', param_dict)
    ##
    ## Option for looking at `perfect` mock catalogues
    feat_proc_df = df_value_modifier(feat_proc_df, 'perf_opt', param_dict)
    ##
    ## Random seed for the analysis
    feat_proc_df = df_value_modifier(feat_proc_df, 'seed', param_dict)

    return [catl_feat_df, feat_proc_df]

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
            catl_ii_str += ' {0} {1}'.format(    df_ii['Flag' ][ii],
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
    window_name     = 'SDSS_ML_data_preprocessing'
    sub_window_name = 'data_preprocessing'
    file_exe_name   = 'catl_data_preprocessing_run.sh'
    ##
    ## File or files to run
    run_file_dict    = {}
    run_file_dict[0] = {'file': 'catl_feature_calculations.py'}
    run_file_dict[1] = {'file': 'feature_processing.py'}
    ##
    ## Saving to main dictionary
    param_dict['env_name'       ] = env_name
    param_dict['window_name'    ] = window_name
    param_dict['sub_window_name'] = sub_window_name
    param_dict['file_exe_name'  ] = file_exe_name
    param_dict['run_file_dict'  ] = run_file_dict

    return param_dict

def file_construction_and_execution(df_arr, param_dict):
    """
    1) Creates file that has shell commands to run executable
    2) Executes the file, which creates a screen session with the executables

    Parameters:
    -----------
    params_pd: pandas DataFrame
        DataFrame with necessary parameters to run `sdss mocks create`

    param_dict: python dictionary
        dictionary with project variables
    
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
        out_f.write( """SCRIPT_CMD="{0}"\n""".format(main_str_cmd).encode())
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
        out_f.write(b"""screen -S ${WINDOW_NAME} -p ${SUB_WINDOW} -X stuff $"${SCRIPT_CMD}"\n""")
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
    print(".>>> Running Script....")
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
    df_arr = get_analysis_params(param_dict)
    ##
    ## Running analysis
    file_construction_and_execution(df_arr, param_dict)

# Main function
if __name__=='__main__':
    ## Input arguments
    args = get_parser()
    # Main Function
    main(args)
