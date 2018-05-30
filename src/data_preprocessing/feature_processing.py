#! /usr/bin/env python
# -*- coding: utf-8 -*-

# Victor Calderon
# Created      : 2018-05-27
# Last Modified: 2018-05-29
# Vanderbilt University
from __future__ import print_function, division, absolute_import
__author__     =['Victor Calderon']
__copyright__  =["Copyright 2018 Victor Calderon, "]
__email__      =['victor.calderon@vanderbilt.edu']
__maintainer__ =['Victor Calderon']
"""
Selects the features to analyze and saves the outputs for creating the 
training and testing datasets.
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
from cosmo_utils.ml    import ml_utils        as cmlu

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
import seaborn as sns
#sns.set()
from progressbar import (Bar, ETA, FileTransferSpeed, Percentage, ProgressBar,
                        ReverseBar, RotatingMarker)
from tqdm import tqdm

# Extra-modules
from argparse import ArgumentParser
from argparse import HelpFormatter
from operator import attrgetter
import astropy.constants as ac
import astropy.units     as u
from glob import glob

# ML modules
import sklearn
from   sklearn import utils as skutils

## Functions

#### ---------------------- General Functions -----------------------------###

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
    Selects the features to analyze and saves the outputs for creating the 
    training and testing datasets.
                    """
    parser = ArgumentParser(description=description_msg,
                            formatter_class=SortingHelpFormatter,)
    ## 
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
    ## Total number of properties to predict. Default = 1
    parser.add_argument('-n_predict',
                        dest='n_predict',
                        help="""
                        Number of properties to predict. Default = 1""",
                        type=int,
                        choices=range(1,4),
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
    ## Option for calculating densities or not
    parser.add_argument('-dens_calc',
                        dest='dens_calc',
                        help='Option for calculating densities.',
                        type=_str2bool,
                        default=True)
    ## CPU Counts
    parser.add_argument('-cpu',
                        dest='cpu_frac',
                        help='Fraction of total number of CPUs to use',
                        type=float,
                        default=0.75)
    ## Option for removing file
    parser.add_argument('-remove',
                        dest='remove_files',
                        help='Removed main files',
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

def is_tool(name):
    """Check whether `name` is on PATH and marked as executable."""

    # from whichcraft import which
    from shutil import which

    return which(name) is not None

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
        if not (all(box_n_diff > 0)):
            msg = '{0} The value of `box_idx` ({1}) is not valid!'.format(
                file_msg, param_dict['box_idx'])
            raise ValueError(msg)

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
    ## Sample volume
    # Units (Mpc/h)**3
    volume_sample = {   '18':37820 / 0.01396,
                        '19':6046016.60311  ,
                        '20':2.40481e7      ,
                        '21':8.79151e7      }
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
    ## Catalogue Prefix for input catalogue
    catl_input_arr = [  sample_Mr,
                        param_dict['hod_n'],
                        param_dict['clf_method'],
                        param_dict['cosmo_choice'],
                        param_dict['nmin'],
                        param_dict['halotype'],
                        param_dict['perf_opt']]
    catl_input_str  = '{0}_hodn_{1}_clf_{2}_cosmo_{3}_nmin_{4}_halotype_{5}_'
    catl_input_str += 'perf_{6}'
    catl_input_str  = catl_input_str.format(*catl_input_arr)
    ##
    ## Catalogue main string
    catl_pre_arr = [    sample_Mr,
                        param_dict['hod_n'],
                        param_dict['halotype'],
                        param_dict['clf_method'],
                        param_dict['clf_seed'],
                        param_dict['dv'],
                        param_dict['catl_type'],
                        param_dict['cosmo_choice'],
                        param_dict['nmin'],
                        param_dict['perf_opt']]
    # String
    catl_pre_str  = '{0}_hodn_{1}_halotype_{2}_clfmethod_{3}_clfseed_{4}_'
    catl_pre_str += 'dv_{5}_catltype_{6}_cosmo_{7}_nmin_{8}_perf_{9}_'
    catl_pre_str  = catl_pre_str.format(catl_pre_arr)
    ##
    ## Saving to `param_dict`
    param_dict['sample_s'      ] = sample_s
    param_dict['sample_Mr'     ] = sample_Mr
    param_dict['vol_mr'        ] = vol_mr
    param_dict['cens'          ] = cens
    param_dict['sats'          ] = sats
    param_dict['speed_c'       ] = speed_c
    param_dict['cpu_number'    ] = cpu_number
    param_dict['catl_input_str'] = catl_input_str
    param_dict['catl_pre_str'  ] = catl_pre_str

    return param_dict

def test_feat_file(param_dict, proj_dict):
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
    file_msg = param_dict['Prog_msg']
    ##
    ## Filename, under which to save all of the information
    # Main String
    # `sample_frac`
    if (param_dict['test_train_opt'] == 'sample_frac'):
        filename_str_arr = [    param_dict['catl_pre_str'],
                                param_dict['shuffle_opt'],
                                param_dict['n_predict'],
                                param_dict['pre_opt'],
                                param_dict['n_feat_use'],
                                param_dict['test_train_opt'],
                                param_dict['test_size'],
                                param_dict['sample_frac'],
                                param_dict['dens_calc']]
        ## Main string
        filename_str  = '{0}_sh_{1}_npredict_{2}_preopt_{3}_nfeat_{4}_'
        filename_str += 'testtrain_{5}_testsize_{6}_samplefrac_{7}_'
        filename_str += 'dens_{8}'
        filename_str  = filename_str.format(*filename_str_arr)
    # `boxes_n
    if (param_dict['test_train_opt'] == 'boxes_n'):
        filename_str_arr = [    param_dict['catl_pre_str'],
                                param_dict['shuffle_opt'],
                                param_dict['n_predict'],
                                param_dict['pre_opt'],
                                param_dict['n_feat_use'],
                                param_dict['test_train_opt'],
                                param_dict['box_idx'],
                                param_dict['dens_calc']]
        ## Main string
        filename_str  = '{0}_sh_{1}_npredict_{2}_preopt_{3}_nfeat_{4}_'
        filename_str += 'testtrain_{5}_boxidx_{6}_dens_{7}'
        filename_str  = filename_str.format(*filename_str_arr)
    ##
    ## Path to output file
    filepath = os.path.join(proj_dict['catl_feat_dir'],
                            '{0}_feature_processing_out.p'.format(filename_str))
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
    param_dict['filepath'    ] = filepath
    param_dict['filename_str'] = filename_str

    return run_opt, param_dict

#### ---------------------- Project Structure -----------------------------###

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
    ##
    ## Catalogue prefix string
    prefix_str = os.path.join(  'SDSS',
                                'mocks',
                                'halos_{0}'.format(param_dict['halotype']),
                                'dv_{0}'.format(param_dict['dv']),
                                'hod_model_{0}'.format(param_dict['hod_n']),
                                'clf_seed_{0}'.format(param_dict['clf_seed']),
                                'clf_method_{0}'.format(param_dict['clf_method']),
                                param_dict['catl_type'],
                                param_dict['sample_Mr'],
                                'dens_{0}'.format(param_dict['dens_calc']))
    ##
    ## Directory of the catalogues being analyzed - Compilation of all mocks
    catl_dir = os.path.join(    proj_dict['int_dir'],
                                'merged_feat_catl',
                                prefix_str,
                                'merged_vac_combined')
    ##
    ## Directory for the processed features
    catl_feat_dir = os.path.join(   proj_dict['int_dir'],
                                    'catl_features',
                                    prefix_str,
                                    'feat_processing')
    ##
    ## Checking that directories exists
    for dir_ii in [catl_dir]:
        if not os.path.exists(dir_ii):
            msg = '{0} `dir_ii` ({1}) does not exist!'.format(
                param_dict['Prog_msg'], dir_ii)
            raise FileNotFoundError(msg)
    ## Creating directories
    cfutils.Path_Folder(catl_feat_dir)
    ##
    ## Saving directory to dictionary
    proj_dict['catl_dir'     ] = catl_dir
    proj_dict['catl_feat_dir'] = catl_feat_dir

    return proj_dict

#### ---------------------- Feature Selection -----------------------------###

def feat_selection(param_dict, proj_dict, random_state=0, shuffle_opt=True,
    dropna_opt=True, sample_frac=0.1, test_size=0.25, pre_opt='standard',
    test_train_opt='boxes_n', ext='hdf5'):
    """
    Selects the features used for the ML analysis

    Parameters
    ----------
    param_dict : `dict`
        Dictionary with `project` variables.

    proj_dict : `dict`
        Dictionary with the `project` paths and directories.

    random_state : `int`, optional
        Random state used for when selecting `training` and `testing`
        datasets. If set, it iwll always have the same seed `random_state`.
        This variable is set to `0` by default.

    shuffle_opt : `bool`
        If True, the data will be shuffled before being splitted.
        If True, `stratify` must be set to `None`.

    dropna_opt : `bool`
        If True, all the `NaN` will be dropped from the dataset.

    sample_frac : `float`
        Fraction of the total dataset ot use.

    test_size : float, optional
        Percentage of the catalogue that represents the `test` size of 
        the testing dataset. This variable must be between (0,1).
        This variable is set to `0.25` by default.

    pre_opt : {'min_max', 'standard', 'normalize', 'no'} `str`, optional
        Type of preprocessing to do on `feat_arr`.

        Options:
            - 'min_max' : Turns `feat_arr` to values between (0,1)
            - 'standard' : Uses the `~sklearn.preprocessing.StandardScaler` method
            - 'normalize' : Uses the `~sklearn.preprocessing.Normalizer` method
            - 'no' : No preprocessing on `feat_arr`

    test_train_opt : {'sample_frac', 'boxes_n'} `str`
        Option for which kind of separation to use for the training/testing 
        splitting. This variable is set to 'boxes_n' by default.

        Options:
            - 'sample_frac' : Selects a fraction of the total sample
            - 'boxes_n' : Uses a set of the simulation boxes for the `training` and `testing`

    ext: string, optional (default = 'hdf5')
        file extension of the `merged` catalogue

    Returns
    ---------
    train_dict : `dict`
        Dictionary containing the 'training' data from the catalogue

    test_dict : `dict`
        Dictionary containing the 'testing' data from the catalogue

    param_dict : `dict`
        Dictionary with `project` variables + list of `predicted` and `features`
    """
    file_msg = param_dict['Prog_msg']
    ##
    ## Reading in list of catalogues
    catl_arr = glob(os.path.join(   proj_dict['catl_dir'],
                                    '*{0}*.{1}').format(
                                    param_dict['catl_input_str'],
                                    ext))
    ## Checking if file exists
    if not (len(catl_arr) == 1):
        msg = '{0} The length of `catl_arr` ({1}) must be equal to 1!'
        msg += 'Exiting ...'
        msg  = msg.format(file_msg, len(catl_arr))
        raise ValueError(msg)
    ##
    ## Reading in catalogue
    catl_pd_tot = cfreaders.read_hdf5_file_to_pandas_DF(catl_arr[0])
    ##
    ## Temporarily fixing `GG_mdyn_rproj`
    catl_pd_tot.loc[:, 'GG_mdyn_rproj'] /= 0.96
    ##
    ## List of column names
    catl_cols = catl_pd_tot.columns.values
    ##
    ## List of `features` and `predicted values`
    if param_dict['n_predict'] == 1
        predicted_cols = ['M_h']
    elif param_dict['n_predict'] == 2:
        predicted_cols = ['M_h', 'galtype']
    ##
    ## List of features to use
    if (param_dict['n_feat_use'] == 'all'):
        features_cols = [s for s in catl_cols if s not in predicted_cols]
    elif (param_dict['n_feat_use'] == 'sub'):
        features_cols = [   'M_r',
                            'GG_mr_brightest',
                            'g_r',
                            'GG_rproj',
                            'GG_sigma_v',
                            'GG_M_r',
                            'GG_ngals',
                            'GG_M_group',
                            'GG_mdyn_rproj']
    ## Dropping NaN's
    if dropna_opt:
        catl_pd_tot.dropna(how='any', inplace=True)

    ## Choosing which type to use for the training/testing datasets
    if test_train_opt == 'sample_frac':
        ## Fraction of the total dataset
        catl_pd = catl_pd_tot.sample(   frac=sample_frac,
                                        random_state=random_state)
        ## Deleting `total` catalogue
        catl_pd_tot = None
        ##
        ## Creating new DataFrames
        pred_arr = catl_pd.loc[:, predicted_cols].values
        feat_arr = catl_pd.loc[:, features_cols ].values
        # Scaled Feature array
        feat_arr_scaled = cmlu.data_preprocessing(  feat_arr,
                                                    pre_opt=pre_opt)
        ##
        ## Rescaling and computing training and testing datasets
        (   train_dict,
            test_dict ) = cmlu.train_test_dataset(  pred_arr,
                                                    feat_arr,
                                                    pre_opt=pre_opt,
                                                    shuffle_opt=shuffle_opt,
                                                    random_state=random_state,
                                                    test_size=test_size)
        ##
        ## Saving to dictionary
        param_dict['predicted_cols' ] = predicted_cols
        param_dict['features_cols'  ] = features_cols
        param_dict['feat_arr_scaled'] = feat_arr_scaled
        param_dict['feat_arr'       ] = feat_arr
    ##
    ## If selecting testing/training based on which boxes
    if test_train_opt =='boxes_n':
        ## Simualation boxes - Indices
        (   box_train_start,
            box_train_end  ,
            box_test_idx   ) = (num.array(param_dict['box_idx'].split('_'))
                                    .astype(int))
        ##
        ## Selecting subsample of the main catalogue
        # Training
        catl_train_pd = catl_pd_tot.loc[(catl_pd_tot['box_n']).between(
                            box_train_start,
                            box_train_end,
                            inclusive=True)]
        # Testing
        catl_test_pd  = catl_pd_tot.loc[(catl_pd_tot['box_n'] == box_test_idx)]
        ##
        ## Shuffling if needed
        if shuffle_opt:
            # Train
            catl_train_pd = skutils.shuffle(catl_train_pd,
                                            random_state=random_state)
            # Test
            catl_test_pd  = skutils.shuffle(catl_test_pd,
                                            random_state=random_state)
        ##
        ## `Features` and `predictions`
        # Training
        pred_train_arr        = catl_train_pd.loc[:, predicted_cols].values
        feat_train_arr        = catl_train_pd.loc[:, features_cols ].values
        feat_train_arr_scaled = cmlu.data_preprocessing(feat_train_arr,
                                                        pre_opt=pre_opt)
        # Testing
        pred_test_arr        = catl_test_pd.loc[:, predicted_cols].values
        feat_test_arr        = catl_test_pd.loc[:, features_cols ].values
        feat_test_arr_scaled = cmlu.data_preprocessing( feat_test_arr,
                                                        pre_opt=pre_opt)
        ##
        ## Assigning `training` and `testing` datasets to dictionary
        # Training
        train_dict = {  'X_train'   :feat_train_arr_scaled,
                        'Y_train'   :pred_train_arr,
                        'X_train_ns':feat_train_arr,
                        'Y_train_ns':pred_train_arr}
        # Testing
        test_dict  = {  'X_test'   :feat_test_arr_scaled,
                        'Y_test'   :pred_test_arr,
                        'X_test_ns':feat_test_arr,
                        'Y_test_ns':pred_test_arr}
        ##
        ## Saving to dictionary
        param_dict['predicted_cols' ] = predicted_cols
        param_dict['features_cols'  ] = features_cols
        param_dict['feat_arr_scaled'] = feat_train_arr_scaled
        param_dict['feat_arr'       ] = feat_train_arr

    return train_dict, test_dict, param_dict

def train_test_save(param_dict, train_dict, test_dict):
    """
    Saves the `training` and `testing` dictionaries to a file, so that 
    it can be used for future analyses.

    Parameters
    ------------
    param_dict : `dict`
        Dictionary with `project` variables.

    train_dict : `dict`
        Dictionary containing the 'training' data from the catalogue

    test_dict : `dict`
        Dictionary containing the 'testing' data from the catalogue.
    """
    file_msg = param_dict['Prog_msg']
    filepath = param_dict['filepath']
    ##
    ## Saving new file if necessary
    ##
    ## Data to be saved in the pickle file
    if not (os.path.exists(filepath)):
        ##
        ## List of objects to save in pickle file.
        obj_arr = [train_dict, test_dict]
        ## Savng to pickle file
        with open(filepath, 'wb') as file_p:
            pickle.dump(obj_arr, file_p)
        ##
        ## Checking that file exists
        cfutils.File_Exists(filepath)
    ##
    ## Output message
    msg = '{0} Output file: {1}'.format(file_msg, filepath)
    print(msg)

#### ---------------------- Main Selection --------------------------------###

def main():
    """
    Selects the features to analyze and saves the outputs for creating the 
    training and testing datasets.
    """
    ## Reading all elements and converting to python dictionary
    param_dict = vars(args)
    ## Checking for correct input
    param_vals_test(param_dict)
    ## Adding extra variables
    param_dict = add_to_dict(param_dict)
    ## Program message
    Prog_msg = param_dict['Prog_msg']
    ##
    ## Testing of whether or not to run the analysis
    (   run_opt   ,
        param_dict) = test_feat_file(param_dict, proj_dict)
    ##
    ## Creating Folder Structure
    # proj_dict  = directory_skeleton(param_dict, cwpaths.cookiecutter_paths(__file__))
    proj_dict  = directory_skeleton(param_dict, cwpaths.cookiecutter_paths('./'))
    ##
    ## Printing out project variables
    print('\n'+50*'='+'\n')
    for key, key_val in sorted(param_dict.items()):
        if key !='Prog_msg':
            print('{0} `{1}`: {2}'.format(Prog_msg, key, key_val))
    print('\n'+50*'='+'\n')
    ##
    ## Reading in `merged` catalogue and separating training and 
    ## testing datasets
    if run_opt:
        (   train_dict,
            test_dict ,
            param_dict) = feat_selection(   param_dict,
                                            proj_dict,
                                            random_state=param_dict['seed'],
                                            shuffle_opt=param_dict['shuffle_opt'],
                                            dropna_opt=param_dict['dropna_opt'],
                                            sample_frac=param_dict['sample_frac'],
                                            test_size=param_dict['test_size'],
                                            pre_opt=param_dict['pre_opt'],
                                            test_train_opt=param_dict['test_train_opt'])
    ##
    ## Saving dictionaries and more
    train_test_save(param_dict, train_dict, test_dict)

# Main function
if __name__=='__main__':
    ## Input arguments
    args = get_parser()
    # Main Function
    main()
