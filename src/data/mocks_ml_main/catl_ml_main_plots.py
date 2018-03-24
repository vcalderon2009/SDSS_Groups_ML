#! /usr/bin/env python
# -*- coding: utf-8 -*-

# Victor Calderon
# Created      : 03/21/2018
# Last Modified: 03/21/2018
# Vanderbilt University
from __future__ import print_function, division, absolute_import
__author__     =['Victor Calderon']
__copyright__  =["Copyright 2017 Victor Calderon, "]
__email__      =['victor.calderon@vanderbilt.edu']
__maintainer__ =['Victor Calderon']
"""
Produces plots of the merged Dataset to check correlations, etc.
"""
# Path to Custom Utilities folder
import os
import sys
import git
from path_variables import git_root_dir
sys.path.insert(0, os.path.realpath(git_root_dir(__file__)))

# Importing Modules
import src.data.utilities_python as cu
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
from matplotlib.colors import ListedColormap

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
                    Plots the results of ML algorithms that were applied to 
                    galaxy and galaxy group in the catalogue
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
    ## `Perfect Catalogue` Option
    parser.add_argument('-perf',
                        dest='perf_opt',
                        help='Option for using a `Perfect` catalogue',
                        type=_str2bool,
                        default=False)
    ## Total number of properties to predict. Default = 1
    parser.add_argument('-n_predict',
                        dest='n_predict',
                        help="""
                        Number of properties to predict. Default = 1""",
                        type=int,
                        choices=range(1,4),
                        default=1)
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
    ## Program message
    parser.add_argument('-progmsg',
                        dest='Prog_msg',
                        help='Program message to use throught the script',
                        type=str,
                        default=cu.Program_Msg(__file__))
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
    if (param_dict['n_predict'] < 1):
        msg  = '{0} The value for `n_predict` ({1}) must be LARGER than `1`'
        msg += 'Exiting...'
        msg  = msg.format(  param_dict['Prog_msg' ],
                            param_dict['n_predict'])

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
    catl_str_read     = '{0}_hodn_{1}_clf_{2}_cosmo_{3}_nmin_{4}_halotype_{5}_perf_'
    catl_str_read    += '{6}'
    catl_str_read     = catl_str_read.format(*catl_str_arr)
    ##
    ## Figure catalogue string
    catl_str_fig_arr = [catl_str_read,
                        param_dict['n_predict']]
    catl_str_fig = '{0}_n_predict_{1}'
    catl_str_fig = catl_str_fig.format(*catl_str_fig_arr)
    ##
    ## Column names
    ml_dict_cols_names = ml_file_data_cols()
    ##
    ## Saving to `param_dict`
    param_dict['sample_s'          ] = sample_s
    param_dict['sample_Mr'         ] = sample_Mr
    param_dict['cens'              ] = cens
    param_dict['sats'              ] = sats
    param_dict['speed_c'           ] = speed_c
    param_dict['catl_str_read'     ] = catl_str_read
    param_dict['catl_str_fig'      ] = catl_str_fig
    param_dict['ml_dict_cols_names'] = ml_dict_cols_names

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
    ##
    ## Figure directory
    figure_dir = os.path.join(  proj_dict['plot_dir'],
                                proj_str)
    ##
    ## Creating Directories
    catl_dir_arr = [ext_dir, processed_dir, int_dir, raw_dir, test_train_dir]
    for catl_ii in catl_dir_arr:
        try:
            assert(os.path.exists(catl_ii))
        except:
            msg = '{0} `{1}` does not exist! Exiting'.format(
                param_dict['Prog_msg'], catl_ii)
            raise ValueError(msg)
    ##
    ## Creating directories
    cu.Path_Folder(figure_dir)
    ##
    ## Adding to `proj_dict`
    proj_dict['ext_dir'       ] = ext_dir
    proj_dict['processed_dir' ] = processed_dir
    proj_dict['int_dir'       ] = int_dir
    proj_dict['raw_dir'       ] = raw_dir
    proj_dict['test_train_dir'] = test_train_dir
    proj_dict['figure_dir'    ] = figure_dir

    return proj_dict

## --------- Preparing Data ------------##

## Galaxy and Property Names
def ml_file_data_cols():
    """
    Substitutes for the column names in the `ml_file`

    Returns
    ---------
    ml_dict_cols_names: python dictionary
        dictionary with column names for each column in the ML file
    """
    ml_dict_cols_names = {  'GG_r_tot':"Total Radius (G)",
                            'GG_sigma_v': "Velocity Dispersion (G)",
                            'GG_mr_brightest':"Lum. of Brightest Galaxy (G)",
                            'g_galtype':"Group galaxy type",
                            'GG_r_med':"Median radius (G)",
                            'GG_mr_ratio': "Luminosity ratio (G,1-2)",
                            'GG_logssfr': "log(sSFR) (G)",
                            'GG_mdyn_rmed':"Dynamical mass at median radius (G)",
                            'GG_dist_cluster':"Distance to closest cluster (G)",
                            'GG_M_r':"Total Brightness (G)",
                            'GG_rproj':"Total Rproj (G)",
                            'GG_shape':"Group's shape (G)",
                            'GG_mdyn_rproj':"Dynamical mass at Rproj (G)",
                            'GG_dens_10.0':"Density at 10 Mpc/h (G)",
                            'GG_dens_5.0':"Density at 5 Mpc/h (G)",
                            'GG_dens_2.0':"Density at 2 Mpc/h (G)",
                            'GG_M_group':"Group's Ab. Matched Mass (G)",
                            'GG_sigma_v_rmed':"Velocity Dispersion at Rmed (G)",
                            'GG_ngals':"Group richness (G)",
                            'M_r':"Galaxy's luminosity",
                            'g_r':"(g-r) galaxy color",
                            'dist_centre_group':"Distance to Group's centre",
                            'g_brightest':"If galaxy is group's brightest galaxy",
                            'logssfr':"Log of Specific star formation rate ",
                            'sersic': "Galaxy's morphology"}

    return ml_dict_cols_names

## Reading in File
def ml_file_read(proj_dict, param_dict):
    """
    Reads in the ML file, which contains the info of different 
    ML algorithms

    Parameters
    -----------
    param_dict: python dictionary
        dictionary with `project` variables

    proj_dict: python dictionary
        dictionary with info of the project that uses the
        `Data Science` Cookiecutter template.

    Returns
    -----------
    obj_arr: tuple, shape (4,)
        List of elements from the `trained` algorithms
        It includes:
            - 'model_fits_dict': python dictionary
                Dictionary for storing 'fit' and 'score' data for different algorithms
            - 'train_dict': python dictionary
                dictionary containing the 'training' data from the catalogue
            - 'test_dict': python dictionary
                dictionary containing the 'testing' data from the catalogue
            - 'param_dict': python dictionary
                dictionary with `project` variables used when training the 
                algorithms.
    """
    ## Filename
    filepath_str  = '{0}_model_fits_dict.p'.format(param_dict['catl_str_fig'])
    filepath = os.path.join(    proj_dict['test_train_dir'],
                                filepath_str)
    ## Checking if file exists
    try:
        assert(os.path.exists(filepath))
    except:
        msg = '{0} File `{1}` was not found... Exiting'.format(
            param_dict['Prog_msg'], filepath)
    ##
    ## Reading in data
    with open(filepath, 'rb') as file_p:
        obj_arr = pickle.load(file_p)

    return obj_arr

## --------- Plotting Functions ------------##

## Scores for `General` and `K-Folds` Methods

## Feature importance - Barchart
def feature_imp_chart(model_fits_dict, param_dict, proj_dict,
    fig_fmt='pdf', figsize=(10,8)):
    """
    Plots the importance of each feature for the ML algorithm

    Parameters
    -----------
    model_fits_dict: python dictionary
        Dictionary for storing 'fit' and 'score' data for different algorithms

    param_dict: python dictionary
        dictionary with `project` variables

    proj_dict: python dictionary
        dictionary with info of the project that uses the
        `Data Science` Cookiecutter template.

    fig_fmt: string, optional (default = 'pdf')
        extension used to save the figure

    figsize: tuple, optional (12,15.5)
        size of the output figure
    """
    ## Constants
    ml_dict_cols_names = param_dict['ml_dict_cols_names']
    ## List of algorithms being used
    skem_key_arr = num.sort(list(model_fits_dict.keys()))
    ## Looping over all algorithms
    for skem_key in tqdm(skem_key_arr):
        ## Model being analyzed
        ## Figure name
        fname = os.path.join(   proj_dict['figure_dir'],
                                '{0}_feature_importance_{1}.pdf'.format(
                                    param_dict['catl_str_fig'],
                                    skem_key))
        ## Reading in data
        feat_imp_gen_sort = model_fits_dict[skem_key]['feat_imp_gen_sort']
        feat_imp_kf_sort  = model_fits_dict[skem_key]['feat_imp_kf_sort' ]
        ## Converting to DataFrames
        # General
        feat_imp_gen_sort_pd = pd.DataFrame(feat_imp_gen_sort[:,1].astype(float),
                                            index=feat_imp_gen_sort[:,0],
                                            columns=['General'])
        # K-Folds
        feat_imp_kf_sort_pd  = pd.DataFrame(feat_imp_kf_sort[:,1].astype(float),
                                            index=feat_imp_kf_sort[:,0],
                                            columns=['K-Fold'])
        ## Joining DataFrames
        feat_gen_kf_merged   = pd.merge(    feat_imp_gen_sort_pd,
                                            feat_imp_kf_sort_pd,
                                            left_index=True,
                                            right_index=True)
        ## Renaming indices
        # Finding set of common property labels
        feat_gen_kf_merged_idx       = feat_gen_kf_merged.index.values
        feat_gen_kf_merged_intersect = num.intersect1d(feat_gen_kf_merged_idx,
                                                list(ml_dict_cols_names.keys()))
        ml_dict_cols_names_select = {key:ml_dict_cols_names[key] for key in \
                                        feat_gen_kf_merged_intersect}
        # Renaming indices
        feat_gen_kf_merged.rename(  index=ml_dict_cols_names, inplace=True)
        ## Sorting by descending values
        feat_gen_kf_merged_sort = feat_gen_kf_merged.sort_values(
                                        by=['K-Fold','General'], ascending=False)
        ##
        ## Figure details
        fig_title = skem_key.replace('_', ' ').title()
        # Figure
        plt.clf()
        plt.close()
        fig = plt.figure(figsize=figsize)
        ax1 = fig.add_subplot(111, facecolor='white')
        # Constants
        feat_gen_kf_merged_sort.plot(   kind='barh',
                                        stacked=False,
                                        # colormap=ListedColormap(sns.color_palette("GnBu", 10)),
                                        ax=ax1,
                                        title=fig_title,
                                        legend=True)
        ## Ticks
        ax_data_major_loc  = ticker.MultipleLocator(0.05)
        ax1.xaxis.set_major_locator(ax_data_major_loc)
        ##
        ## Saving figure
        if fig_fmt=='pdf':
            plt.savefig(fname, bbox_inches='tight')
        else:
            plt.savefig(fname, bbox_inches='tight', dpi=400)
        print('{0} Figure saved as: {1}'.format(Prog_msg, fname))
        plt.clf()
        plt.close()

## Feature Importance - Cumulative Score



## --------- Main Function ------------##

def main(args):
    """
    Plots the results of a train ML algorithm that predict 
    galaxy and group properties.
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
    # proj_dict  = directory_skeleton(param_dict, cu.cookiecutter_paths(__file__))
    proj_dict  = directory_skeleton(param_dict, cu.cookiecutter_paths('./'))
    ##
    ## Printing out project variables
    keys_avoid_arr = ['Prog_msg', 'ml_dict_cols_names']
    print('\n'+50*'='+'\n')
    for key, key_val in sorted(param_dict.items()):
        if (key not in keys_avoid_arr):
            print('{0} `{1}`: {2}'.format(Prog_msg, key, key_val))
    print('\n'+50*'='+'\n')
    ##
    ## ----- Plotting Section ----- ##
    # Reading in file
    (   model_fits_dict,
        train_dict     ,
        test_dict      ,
        param_dict_ml  ) = ml_file_read(proj_dict, param_dict)
    ##
    ## Feature Importance - Bar Chart
    feature_imp_chart(model_fits_dict, param_dict, proj_dict)










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
    if args.sample == 'all':
        for sample_ii in [19, 20, 21]:
            print('\n'+50*'='+'\n')
            print('\n Sample: {0}\n'.format(sample_ii))
            print('\n'+50*'='+'\n')
            ## Copy of `args`
            args_c = copy.deepcopy(args)
            ## Changing galaxy sample int
            args_c.sample = sample_ii
            main(args_c)
    else:
        args.sample = int(args.sample)
        main(args)
