#! /usr/bin/env python
# -*- coding: utf-8 -*-

# Victor Calderon
# Created      : 2018-05-27
# Last Modified: 2018-05-28
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
    ## Fraction of the sample to use
    parser.add_argument('-sample_frac',
                        dest='sample_frac',
                        help='fraction of the total dataset to use',
                        type=float,
                        default=0.01)
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
    ## Minimum of galaxies in a group
    parser.add_argument('-nmin',
                        dest='nmin',
                        help='Minimum number of galaxies in a galaxy group',
                        type=int,
                        choices=range(2,1000),
                        metavar='[1-1000]',
                        default=2)
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
    parser.add_argument('-train_frac',
                        dest='train_frac',
                        help='Percentage size of the catalogue used for training',
                        type=_check_pos_val,
                        default=0.25)
    ## CPU Counts
    parser.add_argument('-cpu',
                        dest='cpu_frac',
                        help='Fraction of total number of CPUs to use',
                        type=float,
                        default=0.75)
    ## Option for removing file
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
    ## Checking input different types of `test_train_opt`
    #
    # `sample_frac`
    if (param_dict['test_train_opt'] == 'sample_frac'):
        # `sample_frac`
        if not ((param_dict['sample_frac'] > 0) and 
                (param_dict['sample_frac'] <= 1.)):
            msg = '{0} `sample_frac` ({1}) must be between (0,1]'.format(
                param_dict['Prog_msg'], param_dict['sample_frac'])
            raise ValueError(msg)
        # `test_size`
        if not ((param_dict['train_frac'] > 0) and
                (param_dict['train_frac'] < 1)):
            msg = '{0} `train_frac` ({1}) must be between (0,1)'.format(
                param_dict['Prog_msg'], param_dict['train_frac'])
            raise ValueError(msg)
    #
    # boxes_n
    if (param_dict['test_train_opt'] == 'boxes_n'):
        box_n_arr = num.array(param_dict['box_idx'].split('_')).astype(int)
        box_n_diff = num.diff(box_n_arr)
        if not (all(box_n_diff > 0)):
            msg = '{0} The value of `box_idx` ({1}) is not valid!'.format(
                param_dict['Prog_msg'], param_dict['box_idx'])
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
    catl_pre_arr = [    sample_Mr                 , param_dict['hod_n'],
                        param_dict['halotype']    , param_dict['clf_method'],
                        param_dict['dv']          , param_dict['catl_type'],
                        param_dict['sample_frac'] , param_dict['n_predict'],
                        param_dict['shuffle_opt'] , param_dict['pre_opt'],
                        param_dict['perf_opt']    , param_dict['nmin']]
    catl_pre_str = '{0}_hodn_{1}_halotype_{2}_clfmethod_{3}_dv_{4}_catltype_{5}'
    catl_pre_str += '_samplefrac_{6}_npred_{7}_shuffle_{8}_preopt_{9}_'
    catl_pre_str += 'perf_{10}_nmin_{11}'
    catl_pre_str  = catl_pre_str.format(*catl_pre_arr)
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
                                param_dict['sample_Mr'])
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

def feat_selection(param_dict, proj_dict, random_state=0, shuffle_opt=True,
    dropna_opt=True, sample_frac=0.1, train_size=0.25, ext='hdf5'):
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

    train_size : `float`
        Percentage size of the catalogue used for `training`.

    ext: string, optional (default = 'hdf5')
        file extension of the `merged` catalogue

    Returns
    ---------
    """
    file_msg = param_dict['Prog_msg']
    ##
    ## Reading in list of catalogues
    catl_arr = glob(os.path.join(   proj_dict['catl_dir'],
                                    '*{0}*.{1}').format(
                                    param_dict['catl_input_str'],
                                    ext))
    ## Checking if file exists
    if not (len(catl_arr) ==1):
        msg = '{0} The length of `catl_arr` ({1}) must be equal to 1!'
        msg += 'Exiting ...'
        msg  = msg.format(file_msg, len(catl_arr))
        raise ValueError(msg)
    ##
    ## Reading in catalogue
    catl_pd_tot = cfreaders.read_hdf5_file_to_pandas_DF(catl_arr[0])




def main():
    """

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


# Main function
if __name__=='__main__':
    ## Input arguments
    args = get_parser()
    # Main Function
    main()
