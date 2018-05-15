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

# Importing Modules
from cosmo_utils.utils import file_utils      as cfutils
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
    ## CPU Counts
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
    ## Total number of properties to predict. Default = 1
    parser.add_argument('-n_predict',
                        dest='n_predict',
                        help="""
                        Number of properties to predict. Default = 1""",
                        type=int,
                        choices=range(1,4),
                        default=1)
    ## Option for removing file
    parser.add_argument('-pre_opt',
                        dest='pre_opt',
                        help="""
                        Option for which preprocessing of the data to use.
                        """,
                        type=str,
                        choices=['min_max','standard','normalize','no', 'all'],
                        default='normalize')
    ## Option for determining scoring
    parser.add_argument('-score_method',
                        dest='score_method',
                        help="""
                        Option for determining which scoring method to use.
                        """,
                        type=str,
                        choices=['perc', 'threshold', 'model_score'],
                        default='threshold')
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
                        param_dict['n_predict'],
                        param_dict['pre_opt'  ],
                        param_dict['sample_frac'],
                        param_dict['score_method']]
    catl_str_fig  = '{0}_n_predict_{1}_pre_opt_{2}_sample_frac_{3}'
    catl_str_fig += '_score_method_{4}'
    catl_str_fig  = catl_str_fig.format(*catl_str_fig_arr)
    ##
    ## Plotting constants
    plot_dict = {   'size_label':23,
                    'size_title':25,
                    'color_ham' :'red',
                    'color_dyn' :'blue'}
    ##
    ## Saving to `param_dict`
    param_dict['sample_s'          ] = sample_s
    param_dict['sample_Mr'         ] = sample_Mr
    param_dict['cens'              ] = cens
    param_dict['sats'              ] = sats
    param_dict['speed_c'           ] = speed_c
    param_dict['catl_str_read'     ] = catl_str_read
    param_dict['catl_str_fig'      ] = catl_str_fig
    param_dict['plot_dict'         ] = plot_dict

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
    ##
    ## Figure directory
    figure_dir = os.path.join(  proj_dict['plot_dir'],
                                proj_str,
                                'ml_training_figs',
                                'sample_frac_{0}'.format(param_dict['sample_frac']),
                                'pre_opt_{0}'.format(param_dict['pre_opt']),
                                'score_method_{0}'.format(param_dict['score_method']))
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
    cfutils.Path_Folder(figure_dir)
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
def ml_file_data_cols(param_dict, param_dict_ml):
    """
    Substitutes for the column names in the `ml_file`

    Parameters
    ------------
    param_dict: python dictionary
        dictionary with `project` variables

    param_dict_ml: python dictionary
        dictionary with `project` variables used when training the algorithms

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
                            'GG_mr_ratio': "Luminosity ratio (G)",
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
    ##
    ## Feature labels
    features_cols_ml = param_dict_ml['features_cols']
    ##
    ## Intersection between column names
    feat_cols_intersect = num.intersect1d(  list(ml_dict_cols_names.keys()),
                                            features_cols_ml)
    ##
    ## New dictionary
    feat_cols_dict = {key:ml_dict_cols_names[key] for key in \
                        feat_cols_intersect}
    ##
    ## Saving to `param_dict`
    param_dict['feat_cols_dict'] = feat_cols_dict

    return param_dict

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
            - 'param_dict_ml': python dictionary
                dictionary with `project` variables used when training the 
                algorithms.
    """
    ## Filename
    filepath_str  = '{0}_model_fits_dict.p'.format(param_dict['catl_str_fig'])
    filepath = os.path.join(    proj_dict['test_train_dir'],
                                filepath_str)
    print('{0} Read `filepath` ({1})...Checking if it exists!'.format(
        param_dict['Prog_msg'], filepath))
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

## Feature importance - Bar Chart
def feature_imp_chart(model_fits_dict, param_dict, proj_dict,
    fig_fmt='pdf', figsize=(10,8), fig_number=1):
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

    fig_number: int, optional (default = 1)
        number of figure in the workflow
    """
    Prog_msg  = param_dict['Prog_msg']
    plot_dict = param_dict['plot_dict']
    ## Constants
    feat_cols_dict = param_dict['feat_cols_dict']
    ## List of algorithms being used
    skem_key_arr = num.sort(list(model_fits_dict.keys()))
    ## Removing `neural_network` from list
    if 'neural_network' in skem_key_arr:
        skem_key_arr = num.array([i for i in skem_key_arr if i != 'neural_network'])
    ## Looping over all algorithms
    for skem_key in tqdm(skem_key_arr):
        ## Model being analyzed
        ## Figure name
        fname = os.path.join(   proj_dict['figure_dir'],
                                'Fig_{0}_{1}_feature_importance_{2}.pdf'.format(
                                    fig_number,
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
        # Renaming indices
        feat_gen_kf_merged.rename(  index=feat_cols_dict, inplace=True)
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
        ## Labels
        x_label = r'$\textrm{Feature Importance} \rightarrow$'
        ax1.set_xlabel(x_label, fontsize=plot_dict['size_label'])
        ##
        ## Saving figure
        if fig_fmt=='pdf':
            plt.savefig(fname, bbox_inches='tight')
        else:
            plt.savefig(fname, bbox_inches='tight', dpi=400)
        print('{0} Figure saved as: {1}'.format(Prog_msg, fname))
        plt.clf()
        plt.close()

## Model Score - Different algorithms - Bar Chart
def model_score_chart(model_fits_dict, param_dict, proj_dict,
    fig_fmt='pdf', figsize=(10,8), fig_number=2):
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

    fig_number: int, optional (default = 2)
        number of figure in the workflow
    """
    Prog_msg  = param_dict['Prog_msg']
    plot_dict = param_dict['plot_dict']
    ##
    ## Figure name
    fname = os.path.join(   proj_dict['figure_dir'],
                            'Fig_{0}_{1}_ml_algorithms_scores.pdf'.format(
                                fig_number,
                                param_dict['catl_str_fig']))
    ## Algorithm names - Thought as indices for the plot
    ml_algs_names = num.sort(list(model_fits_dict.keys()))
    ## Initializing DataFrame
    zero_arr   = num.zeros(len(ml_algs_names))
    col_names  = ['General','K-Fold']
    ml_algs_pd = pd.DataFrame(dict(zip(col_names,[zero_arr.copy() for x in range(len(col_names))])))
    ## Reading in data
    for kk, ml_kk in enumerate(ml_algs_names):
        ## General Score
        ml_score_gen_kk = model_fits_dict[ml_kk]['model_score_tot']
        ## K-Fold Score
        ml_score_kf_kk  = model_fits_dict[ml_kk]['kf_scores'].mean()
        ## Assigning to DataFrame
        ml_algs_pd.loc[kk, 'General'] = ml_score_gen_kk
        ml_algs_pd.loc[kk, 'K-Fold' ] = ml_score_kf_kk
    ##
    ## HAM and Dynamical
    score_ham = model_fits_dict[ml_kk]['score_ham']
    score_dyn = model_fits_dict[ml_kk]['score_dyn']
    ##
    ## Rename indices
    ml_algs_indices = [xx.replace('_',' ').title() for xx in ml_algs_names]
    ml_algs_pd.rename(index=dict(zip(range(len(ml_algs_names)),ml_algs_indices)),
                        inplace=True)
    ##
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
    ax1.axvline(    x=score_ham,
                    color=plot_dict['color_ham'],
                    label='HAM Mass')
    ax1.axvline(    x=score_dyn,
                    color=plot_dict['color_dyn'],
                    label='Dynamical mass')
    ## Ticks
    ax_data_minor_loc  = ticker.MultipleLocator(0.05)
    ax_data_major_loc  = ticker.MultipleLocator(0.1)
    ax1.xaxis.set_minor_locator(ax_data_minor_loc)
    ax1.xaxis.set_major_locator(ax_data_major_loc)
    ##
    ## Axis label
    if param_dict['score_method'] =='perc':
        xlabel = r'$1\sigma$ error in $\Delta \log M_{halo} [\mathrm{dex}]$'
    else:
        xlabel = 'Score'
    ax1.set_xlabel(xlabel)
    ## Legend
    leg = ax1.legend(loc='upper right', numpoints=1, frameon=False,
        prop={'size':14})
    leg.get_frame().set_facecolor('none')
    ##
    ## Saving figure
    if fig_fmt=='pdf':
        plt.savefig(fname, bbox_inches='tight')
    else:
        plt.savefig(fname, bbox_inches='tight', dpi=400)
    print('{0} Figure saved as: {1}'.format(Prog_msg, fname))
    plt.clf()
    plt.close()

## Fractional difference of truth and predicted
def frac_diff_model(model_fits_dict, test_dict, param_dict, proj_dict,
    param_dict_ml, bin_width = 0.4, arr_len=10, bin_statval='left',
    fig_fmt='pdf', figsize=(10,8), fig_number=3):
    """
    Plots the importance of each feature for the ML algorithm

    Parameters
    -----------
    model_fits_dict: python dictionary
        Dictionary for storing 'fit' and 'score' data for different algorithms

    test_dict: python dictionary
        dictionary containing the 'testing' data from the catalogue

    param_dict: python dictionary
        dictionary with `project` variables

    proj_dict: python dictionary
        dictionary with info of the project that uses the
        `Data Science` Cookiecutter template.

    param_dict_ml: python dictionary
        dictionary with `project` variables used when training the 
        algorithms.

    bin_width: float, optional (default = 0.4)
        width of the bin used for the `truth` axis

    arr_len: int, optional (default=0)
        Minimum number of elements in bins

    bin_statval: string, optional (default='average')
        Option for where to plot the bin values of X1_arr and Y1_arr.
        - 'average': Returns the x-points at the average x-value of the bin
        - 'left'   : Returns the x-points at the left-edge of the x-axis bin
        - 'right'  : Returns the x-points at the right-edge of the x-axis bin

    fig_fmt: string, optional (default = 'pdf')
        extension used to save the figure

    figsize: tuple, optional (12,15.5)
        size of the output figure

    fig_number: int, optional (default = )
        number of figure in the workflow

    Note
    -------
    I'm using the `General` predictions for all algorithms
    """
    Prog_msg     = param_dict['Prog_msg']
    # Constants
    cm           = plt.cm.get_cmap('viridis')
    plot_dict    = param_dict['plot_dict']
    ham_color    = 'red'
    alpha        = 0.6
    alpha_mass   = 0.2
    zorder_mass  = 10
    zorder_shade = zorder_mass - 1
    zorder_ml    = zorder_mass + 1
    ##
    ## Figure name
    fname = os.path.join(   proj_dict['figure_dir'],
                            'Fig_{0}_{1}_frac_diff_predicted.pdf'.format(
                                fig_number,
                                param_dict['catl_str_fig']))
    ## Algorithm names - Thought as indices for the plot
    ml_algs_names = num.sort(list(model_fits_dict.keys()))
    n_ml_algs     = len(ml_algs_names)
    # Initializing dictionary that will contain the necessary information
    # on each model
    frac_diff_dict = {}
    ## Reading in arrays for different models
    for kk, model_kk in enumerate(ml_algs_names):
        # X and Y coordinates
        model_kk_x = test_dict['Y_test']
        model_kk_y = model_fits_dict[model_kk]['model_gen_frac_diff_arr']
        ## Calculating error in bins
        (   x_stat_arr,
            y_stat_arr,
            y_std_arr ,
            y_std_err ) = cstats.Stats_one_arr( model_kk_x,
                                                model_kk_y,
                                                base=bin_width,
                                                arr_len=arr_len,
                                                bin_statval=bin_statval)
        ## Saving to dictionary
        frac_diff_dict[model_kk]      = {}
        frac_diff_dict[model_kk]['x_val' ] = model_kk_x
        frac_diff_dict[model_kk]['y_val' ] = model_kk_y
        frac_diff_dict[model_kk]['x_stat'] = x_stat_arr
        frac_diff_dict[model_kk]['y_stat'] = y_stat_arr
        frac_diff_dict[model_kk]['y_err' ] = y_std_arr
    ## Abundance matched mass
    # HAM
    mgroup_ham    = model_fits_dict[model_kk]['pred_ham'     ]
    mh_true_ham   = model_fits_dict[model_kk]['true_halo_ham']
    frac_diff_ham = model_fits_dict[model_kk]['frac_diff_ham']
    # Dynamical
    mgroup_dyn    = model_fits_dict[model_kk]['pred_dyn_mod'     ]
    mh_true_dyn   = model_fits_dict[model_kk]['true_halo_dyn_mod']
    frac_diff_dyn = model_fits_dict[model_kk]['frac_diff_dyn'    ]
    ##
    ## Binning data
    # HAM
    (   x_stat_ham   ,
        y_stat_ham   ,
        y_std_ham    ,
        y_std_err_ham) = cstats.Stats_one_arr(  mh_true_ham,
                                                frac_diff_ham,
                                                base=bin_width,
                                                arr_len=arr_len,
                                                bin_statval=bin_statval)
    y1_ham = y_stat_ham - y_std_ham
    y2_ham = y_stat_ham + y_std_ham
    # Dynamical
    (   x_stat_dyn   ,
        y_stat_dyn   ,
        y_std_dyn    ,
        y_std_err_dyn) = cstats.Stats_one_arr(  mh_true_dyn,
                                                frac_diff_dyn,
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
    xlabel = r'\boldmath$\log M_{halo,\textrm{true}}\left[ h^{-1} M_{\odot}\right]$'
    ylabel = r'Fractional Difference \boldmath$[\%]$'
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
    leg = ax1.legend(loc='upper right', numpoints=1, frameon=False,
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
    ## Saving figure
    if fig_fmt=='pdf':
        plt.savefig(fname, bbox_inches='tight')
    else:
        plt.savefig(fname, bbox_inches='tight', dpi=400)
    print('{0} Figure saved as: {1}'.format(Prog_msg, fname))
    plt.clf()
    plt.close()

## Overall score as each feature is being added sequentially
## For each algorithm separately
def cumulative_score_feature_alg(model_fits_dict, param_dict, proj_dict,
    fig_fmt='pdf', figsize=(10,8), fig_number=4, grid_opt=True):
    """
    Plots the overall score of an algorithm, as each important feature 
    is added sequentially

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

    fig_number: int, optional (default = 4)
        number of figure in the workflow

    grid_opt: boolean, optional (default = True)
        option for plotting a grid

    Note
    -------
    I'm using the `General` predictions for all algorithms
    """
    Prog_msg = param_dict['Prog_msg']
    ## Constants
    feat_cols_dict = param_dict['feat_cols_dict']
    plot_dict      = param_dict['plot_dict'     ]
    ## List of algorithms being used
    skem_key_arr = num.sort(list(model_fits_dict.keys()))
    ## Removing `neural_network` from list
    if 'neural_network' in skem_key_arr:
        skem_key_arr = num.array([i for i in skem_key_arr if i != 'neural_network'])
    ## Looping over all algorithms
    for skem_key in tqdm(skem_key_arr):
        ## Model being analyzed
        ## Figure name
        fname = os.path.join(   proj_dict['figure_dir'],
                                'Fig_{0}_{1}_cumu_score_feats_{2}.pdf'.format(
                                    fig_number,
                                    param_dict['catl_str_fig'],
                                    skem_key))
        ## Reading in data
        feat_score_gen_cumu = model_fits_dict[skem_key]['feat_score_gen_cumu']
        ## Converting to pandas DataFrame
        feat_score_cumu_pd  = pd.DataFrame( feat_score_gen_cumu[:,1].astype(float),
                                            index=feat_score_gen_cumu[:,0],
                                            columns=['score_cumu'])
        # feat_score_cumu_pd.loc[:,'idx'] = num.arange(len(feat_score_cumu_pd))
        feat_score_cumu_pd = feat_score_cumu_pd.astype(float)
        ## Finding set of common property labels
        ## Renaming index
        feat_score_cumu_pd.rename(index=feat_cols_dict, inplace=True)
        # Number of features
        n_feat = len(feat_score_cumu_pd)
        ##
        ## Figure details
        ml_alg_accuracy = 100*feat_score_cumu_pd['score_cumu'].max()
        fig_title       = skem_key.replace('_', ' ').title()
        fig_title      += ' - {0:.2f}\% Accuracy'.format(ml_alg_accuracy)
        # Figure
        plt.clf()
        plt.close()
        fig = plt.figure(figsize=figsize)
        ax1 = fig.add_subplot(111, facecolor='white')
        # Axes labels
        if param_dict['score_method'] == 'model_score':
            xlabel = r'Score $[\%]$'
        elif param_dict['score_method'] == 'perc':
            xlabel = r'$1\sigma$ error in $\Delta \log M_{halo} [\mathrm{dex}]$'
        else:
            xlabel = r'Score $[\%]$'
        ylabel = r'$\leftarrow \textrm{Adding importance}$'
        ax1.set_xlabel(xlabel, fontsize=plot_dict['size_label'])
        ax1.set_ylabel(ylabel, fontsize=plot_dict['size_label'])
        # Title
        ax1.set_title(fig_title, fontsize=plot_dict['size_title'])
        ## Limits
        # X-axis
        if param_dict['score_method'] == 'model_score':
            ax1.set_xlim(0, 100.)
        else:
            ax1.set_xlim(0, 1.)
        # Y-axis
        y_offset = 0.5
        ax1.set_ylim(0-y_offset, n_feat-y_offset)
        # Major Tick marks
        ax_yaxis_ticks_loc = ticker.MultipleLocator(1.)
        ax1.yaxis.set_major_locator(ax_yaxis_ticks_loc)
        ## Plotting
        # Factor - x-axis
        if param_dict['score_method'] == 'model_score':
            x_arr = 100.*feat_score_cumu_pd['score_cumu'].values
        else:
            x_arr = feat_score_cumu_pd['score_cumu'].values    
        y_arr = num.arange(feat_score_cumu_pd.shape[0])
        ax1.plot(   x_arr,
                    y_arr,
                    color='blue',
                    marker='o',
                    linestyle='-',
                    label='General')
        # Setting ticks
        ax1.set_yticks(y_arr)
        # Changing tick marks
        yaxis_new_ticks = feat_score_cumu_pd.index.values
        ax1.yaxis.set_ticklabels(yaxis_new_ticks)
        ## X-axis ticks
        if param_dict['score_method'] == 'model_score':
            x_major = 10.
            x_minor = 5.
        else:
            x_major = 0.1
            x_minor = 0.05
        ax_xaxis_major_ticks_loc = ticker.MultipleLocator(x_major)
        ax_xaxis_minor_ticks_loc = ticker.MultipleLocator(x_minor)
        ax1.xaxis.set_major_locator(ax_xaxis_major_ticks_loc)
        ax1.xaxis.set_major_locator(ax_xaxis_minor_ticks_loc)
        # grid
        if grid_opt:
            ax1.grid(which='major', color='grey', linestyle='--')
        ## Reversing array
        ax1.invert_yaxis()
        ## Legend
        leg = ax1.legend(loc='upper right', numpoints=1, frameon=False,
            prop={'size':14})
        leg.get_frame().set_facecolor('none')
        ##
        ## Saving figure
        if fig_fmt=='pdf':
            plt.savefig(fname, bbox_inches='tight')
        else:
            plt.savefig(fname, bbox_inches='tight', dpi=400)
        print('{0} Figure saved as: {1}'.format(Prog_msg, fname))
        plt.clf()
        plt.close()
        
##
## Ranking of each Galaxy property for each different algorithm
def feature_ranking_ml_algs(model_fits_dict, param_dict, proj_dict,
    param_dict_ml, fig_fmt='pdf', figsize=(15,12), fig_number=5,
    stacked_opt=True):
    """
    Plots the `ranking` of each galaxy property based on the different 
    ML algorithms used.

    Parameters
    -----------
    model_fits_dict: python dictionary
        Dictionary for storing 'fit' and 'score' data for different algorithms
    
    param_dict: python dictionary
        dictionary with `project` variables

    proj_dict: python dictionary
        dictionary with info of the project that uses the
        `Data Science` Cookiecutter template.

    param_dict_ml: python dictionary
        dictionary with `project` variables used when training the 
        algorithms.
    
    fig_fmt: string, optional (default = 'pdf')
        extension used to save the figure

    figsize: tuple, optional (12,15.5)
        size of the output figure

    fig_number: int, optional (default = 5)
        number of figure in the workflow

    stacked_opt: boolean, optional (default = True)
        option to stack the bar plots
    """
    Prog_msg = param_dict['Prog_msg']
    ## Figure name
    fname = os.path.join(   proj_dict['figure_dir'],
                            'Fig_{0}_{1}_feature_ranking.pdf'.format(
                                fig_number,
                                param_dict['catl_str_fig']))
    ## Constants
    feat_cols_dict = param_dict['feat_cols_dict']
    plot_dict      = param_dict['plot_dict'     ]
    ## List of algorithms being used
    skem_key_arr = num.sort(list(model_fits_dict.keys()))
    ## Removing `neural_network` from list
    if 'neural_network' in skem_key_arr:
        skem_key_arr = num.array([i for i in skem_key_arr if i != 'neural_network'])
    # Number of ML algorithms
    n_ml_algs = len(skem_key_arr)
    # Features used
    feat_arr = num.array(param_dict_ml['features_cols'])
    # Number of features
    n_feat = len(feat_arr)
    # Initializing array
    feat_rank_arr = num.zeros((n_feat, n_ml_algs))
    # Initializing DataFrame
    feat_rank_pd = pd.DataFrame(num.zeros(n_feat), index=feat_arr, columns=['temp'])
    # Looping over ML algorithms
    for kk, skem_key in tqdm(enumerate(skem_key_arr)):
        # Reading in Data
        feat_imp_gen_sort = model_fits_dict[skem_key]['feat_imp_gen_sort']
        feat_imp_kf_sort  = model_fits_dict[skem_key]['feat_imp_kf_sort' ]
        ## Converting to DataFrame
        # General
        feat_imp_gen_sort_pd = pd.DataFrame(feat_imp_gen_sort[:,1].astype(float),
                                            index=feat_imp_gen_sort[:,0],
                                            columns=['{0}_gen'.format(skem_key)])
        feat_imp_gen_sort_pd.loc[:,'{0}_gen_rank'.format(skem_key)] = \
                                feat_imp_gen_sort_pd.rank(ascending=False)
        # K-Folds
        # feat_imp_kf_sort_pd  = pd.DataFrame(feat_imp_kf_sort[:,1].astype(float),
        #                                     index=feat_imp_kf_sort[:,0],
        #                                     columns=['{0}_kf'.format(skem_key)])
        # feat_imp_kf_sort_pd.loc[:,'{0}_kf_rank'.format(skem_key)] = \
        #                         feat_imp_kf_sort_pd.rank(ascending=False)
        # ## Joining DataFrames
        # feat_gen_kf_merged   = pd.merge(    feat_imp_gen_sort_pd,
        #                                     feat_imp_kf_sort_pd,
        #                                     left_index=True,
        #                                     right_index=True)
        feat_gen_kf_merged = feat_imp_gen_sort_pd
        ## Dropping non-rank columns
        feat_cols_merged = feat_gen_kf_merged.columns.values
        feat_cols_drop   = [s for s in feat_cols_merged if 'rank' not in s]
        feat_gen_kf_merged.drop(feat_cols_drop, axis=1, inplace=True)
        ## Merging with main DataFrame
        feat_rank_pd = pd.merge(    feat_rank_pd,
                                    feat_gen_kf_merged,
                                    left_index=True,
                                    right_index=True)
    ##
    ## Deleting temporary column
    feat_rank_pd.drop(['temp'], axis=1, inplace=True)
    ##
    ## Calculating ranking
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
    xlabel = r'$\leftarrow \textrm{Importance ranking}$'
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
    ax_data_major_loc  = ticker.MultipleLocator(10)
    ax_data_minor_loc  = ticker.MultipleLocator(5.)
    ax1.xaxis.set_major_locator(ax_data_major_loc)
    ax1.xaxis.set_minor_locator(ax_data_minor_loc)
    # Inverting axis
    ax1.invert_yaxis()
    ##
    ## Saving figure
    if fig_fmt=='pdf':
        plt.savefig(fname, bbox_inches='tight')
    else:
        plt.savefig(fname, bbox_inches='tight', dpi=400)
    print('{0} Figure saved as: {1}'.format(Prog_msg, fname))
    plt.clf()
    plt.close()





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
    # proj_dict  = directory_skeleton(param_dict, cwpaths.cookiecutter_paths(__file__))
    proj_dict  = directory_skeleton(param_dict, cwpaths.cookiecutter_paths('./'))
    ##
    ## Printing out project variables
    keys_avoid_arr = ['Prog_msg', 'feat_cols_dict']
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
    ## Feature keys
    param_dict = ml_file_data_cols(param_dict, param_dict_ml)
    ##
    ## Feature Importance - Bar Chart - Different Algorithms
    feature_imp_chart(model_fits_dict, param_dict, proj_dict)
    ##
    ## Score for each Algorithm
    model_score_chart(model_fits_dict, param_dict, proj_dict)
    ##
    ## Fractional difference of predicted and truth
    frac_diff_model(model_fits_dict, test_dict, param_dict, proj_dict,
        param_dict_ml)
    ##
    ## Overall score as each feature is being added sequentially
    cumulative_score_feature_alg(model_fits_dict, param_dict, proj_dict)
    ##
    ## Ranking of each Galaxy property for each different algorithm
    feature_ranking_ml_algs(model_fits_dict, param_dict, proj_dict,
        param_dict_ml)












    ##
    ## End time for running the catalogues
    end_time   = datetime.now()
    total_time = end_time - start_time
    print('{0} Total Time taken (Create): {1}'.format(param_dict['Prog_msg'],
        total_time))
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
                ## Copy of `args`
                args_c = copy.deepcopy(args)
                args_c.pre_opt = pre_opt_ii
                args_c.sample  = int(args.sample)
                main(args_c)
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
