#! /usr/bin/env python
# -*- coding: utf-8 -*-

# Victor Calderon
# Created      : 03/26/2018
# Last Modified: 03/28/2018
# Vanderbilt University
from __future__ import print_function, division, absolute_import
__author__     =['Victor Calderon']
__copyright__  =["Copyright 2017 Victor Calderon, Catl Properties - Plots"]
__email__      =['victor.calderon@vanderbilt.edu']
__maintainer__ =['Victor Calderon']
"""
Script to plot the correlations between properties of the dataset used, 
as well as to explore the dataset.
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
from progressbar import (Bar, ETA, FileTransferSpeed, Percentage, ProgressBar,
                        ReverseBar, RotatingMarker)
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
    description_msg = """   Plots figures that show correlations and relations
                            between the different features used for the 
                            ML training.
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
    ## Option for Shuffling dataset when selecting sub-sample
    parser.add_argument('-shuffle_opt',
                        dest='shuffle_opt',
                        help="""
                        Option for Shuffling dataset when selecting sub-sample
                        """,
                        type=_str2bool,
                        default=True)
    ## Option dropping NaN's from the dataset
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
                        default='normalize')
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
    ## This is where the tests for `param_dict` input parameters go.

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
    ## Plotting constants
    plot_dict = {   'size_label':18,
                    'size_title':20}
    ##
    ## Saving to `param_dict`
    param_dict['sample_s'    ] = sample_s
    param_dict['sample_Mr'   ] = sample_Mr
    param_dict['cens'        ] = cens
    param_dict['sats'        ] = sats
    param_dict['speed_c'     ] = speed_c
    param_dict['catl_str'    ] = catl_str
    param_dict['cpu_number'  ] = cpu_number
    param_dict['plot_dict'   ] = plot_dict

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
    ## Figure directory
    figure_dir = os.path.join(  proj_dict['plot_dir'],
                                proj_str,
                                'catl_properties_exploration')
    ##
    ## Checking that directories exist
    catl_dir_arr = [ext_dir, processed_dir, int_dir, raw_dir, catl_outdir,
                    merged_gal_dir, merged_gal_perf_dir, merged_gal_all_dir,
                    merged_gal_perf_all_dir]
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
    proj_dict['ext_dir'                ] = ext_dir
    proj_dict['processed_dir'          ] = processed_dir
    proj_dict['int_dir'                ] = int_dir
    proj_dict['raw_dir'                ] = raw_dir
    proj_dict['figure_dir'             ] = figure_dir
    proj_dict['merged_gal_dir'         ] = merged_gal_dir
    proj_dict['merged_gal_perf_dir'    ] = merged_gal_perf_dir
    proj_dict['merged_gal_all_dir'     ] = merged_gal_all_dir
    proj_dict['merged_gal_perf_all_dir'] = merged_gal_perf_all_dir

    return proj_dict

## --------- Tools function ------------##

## Fractional difference
def frac_diff_calc(pred, truth, perc=True, log_opt=False):
    """
    Computes the fractional difference between `pred` and `truth`

    Parameters
    ------------
    pred: numpy.ndarray
        array consisting of the `predicted` values

    truth: numpy.ndarray
        array consisting of the `truth` values

    perc: boolean, optional (default = True)
        option for returning the `frac_diff` in percentage or plain fraction

    log_opt: boolean, optional (default = False)
        If True, it assumes both `pred` and `truth` are in `log10` base.
        If False, it performs the calculation as usual.
    
    Returns
    ------------
    frac_diff: numpy.ndarray
        array with the `fractional difference` values between 
        `pred` and `truth`
    """
    assert(pred.shape == truth.shape)
    ## Initializing array
    frac_diff = num.zeros(pred.shape[0])*num.nan
    ## Filter out results
    pred_finite_idx = num.where(pred != 0)[0]
    pred_finite     = pred [pred_finite_idx]
    truth_finite    = truth[pred_finite_idx]
    ## If values are in log
    if log_opt:
        pred_finite  = 10**pred_finite
        truth_finite = 10**truth_finite
    ## Calculating fractional difference
    frac_diff_vals = (pred_finite - truth_finite)/truth_finite
    ## If precentage
    if perc:
        frac_diff_vals *= 100.
    ##
    ## Assigning to array
    frac_diff[pred_finite_idx] = frac_diff_vals

    return frac_diff


## --------- Preparing Data ------------##

# Reading in data and cleaning it in
def catl_file_read_clean(param_dict, proj_dict, random_state=0,
    shuffle_opt=True, dropna_opt=True, sample_frac=0.1, ext='hdf5'):
    """
    Reads in the catalogue and cleans it

    Parameters
    -------------
    param_dict: python dictionary
        dictionary with `project` variables

    proj_dict: python dictionary
        dictionary with info of the project that uses the
        `Data Science` Cookiecutter template.

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
    catl_pd: pandas DataFrame
        DataFrame containing galaxy and group information
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
    catl_pd_tot  = cu.read_hdf5_file_to_pandas_DF(catl_arr[0])
    ## Selecting only a fraction of the dataset
    catl_pd      = catl_pd_tot.sample(  frac=sample_frac,
                                        random_state=random_state)
    catl_pd_tot  = None
    ## Reindexing
    catl_pd.reset_index(drop=True, inplace=True)
    ## Dropping NaN's
    if dropna_opt:
        catl_pd.dropna(how='any', inplace=True)
    ## Temporarily fixing 'rmed'
    ## Unit constant
    unit_const = ((3*num.pi/2.) * ((u.km/u.s)**2) * (u.Mpc) / ac.G).to(u.Msun)
    unit_const_val = unit_const.value
    # Median Radius
    for mdyn_kk in ['GG_mdyn_rmed', 'GG_mdyn_rproj']:
        mdyn_val               = catl_pd[mdyn_kk].values * unit_const_val
        mdyn_val_idx           = num.where(mdyn_val != 0)[0]
        mdyn_val_idx_val       = num.log10(mdyn_val[mdyn_val_idx])
        mdyn_val[mdyn_val_idx] = mdyn_val_idx_val
        # Saving to `catl_pd`
        catl_pd.loc[:,mdyn_kk] = mdyn_val

    return catl_pd

## --------- Plotting Functions ------------##

## Comparison between estimated group masses via HAM and dynamical mass
def group_mass_comparison(catl_pd, param_dict, proj_dict,
    bin_width = 0.4, arr_len=10, bin_statval='left', statfunction=num.nanmean,
    fig_fmt='pdf', figsize=(10,8), fig_number=1):
    """
    Comparison of estimated group masses.

    Parameters
    -----------
    catl_pd: pandas DataFrame
        DataFrame containing galaxy and group information

    param_dict: python dictionary
        dictionary with `project` variables

    proj_dict: python dictionary
        dictionary with info of the project that uses the
        `Data Science` Cookiecutter template.

    bin_width: float, optional (default = 0.4)
        width of the bin used for the `truth` axis

    arr_len: int, optional (default=0)
        Minimum number of elements in bins

    bin_statval: string, optional (default='average')
        Option for where to plot the bin values of X1_arr and Y1_arr.
        - 'average': Returns the x-points at the average x-value of the bin
        - 'left'   : Returns the x-points at the left-edge of the x-axis bin
        - 'right'  : Returns the x-points at the right-edge of the x-axis bin

    statfunction: statistical function, optional (default=numpy.mean)
        Numerical function to calculate on bins of data.
        - numpy.nanmean  : mean value for each bin + error in the mean.
        - numpy.nanmedian: median value for each bin + error in the median.

    fig_fmt: string, optional (default = 'pdf')
        extension used to save the figure

    figsize: tuple, optional (12,15.5)
        size of the output figure

    fig_number: int, optional (default = 1)
        number of figure in the workflow
    """
    Prog_msg   = param_dict['Prog_msg']
    ## Constants
    cm         = plt.cm.get_cmap('viridis')
    plot_dict  = param_dict['plot_dict']
    ham_color  = 'red'
    dyn_color  = 'blue'
    halo_color = 'black'
    alpha      = 0.3
    ## Filename
    fname    = os.path.join(    proj_dict['figure_dir'],
                                'Fig_{0}_{1}_group_mass_comparison.{2}'.format(
                                    fig_number,
                                    param_dict['catl_str'],
                                    fig_fmt))
    ## Determining values
    mass_ham  = catl_pd['GG_M_group'   ].values
    mass_dyn  = catl_pd['GG_mdyn_rproj'].values
    mass_halo = catl_pd['M_h'          ].values
    ## Fractional differences
    frac_diff_ham = frac_diff_calc(mass_ham, mass_halo, perc=True, log_opt=False)
    frac_diff_dyn = frac_diff_calc(mass_dyn, mass_halo, perc=True, log_opt=False)
    ## Statistics
    mass_dict = {}
    for kk, (type_kk, mass_kk) in enumerate(zip(['ham', 'dyn'],
                                                [mass_ham, mass_dyn])):
        ## Binning
        (   x_stat   ,
            y_stat   ,
            y_std    ,
            y_std_err) = cu.Mean_Std_calculations_One_array(mass_halo,
                                                            mass_kk,
                                                            base=bin_width,
                                                            arr_len=arr_len,
                                                            bin_statval=bin_statval,
                                                            statfunction=statfunction)
        # Limits
        y_lower_kk = y_stat - y_std
        y_upper_kk = y_stat + y_std
        ##
        ## Saving to dictionary
        mass_dict[type_kk]           = {}
        mass_dict[type_kk]['x_val' ] = mass_halo
        mass_dict[type_kk]['y_val' ] = mass_kk
        mass_dict[type_kk]['x_stat'] = x_stat
        mass_dict[type_kk]['y_stat'] = y_stat
        mass_dict[type_kk]['y_err' ] = y_std
        mass_dict[type_kk]['y_low' ] = y_lower_kk
        mass_dict[type_kk]['y_high'] = y_upper_kk
    ##
    ## Figure details
    #
    # Labels
    xlabel = r'\boldmath$\log M_{halo,\textrm{true}}\left[ h^{-1} M_{\odot}\right]$'
    ylabel = r'Fractional Difference \boldmath$[\%]$'
    ##
    plt.clf()
    plt.close()
    fig = plt.figure(figsize=figsize)
    ax1 = fig.add_subplot(111, facecolor='white')
    ## Horizontal line
    ax1.axhline(y=0, color='black', linestyle='--', zorder=10)
    ##
    ## Plotttin Fractional differences
    for kk, (type_kk, color_kk) in enumerate(zip(   ['ham', 'dyn'],
                                                    [ham_color, dyn_color])):
        # Reading data
        x_stat = mass_dict[type_kk]['x_stat']
        y_stat = mass_dict[type_kk]['y_stat']
        y_err  = mass_dict[type_kk]['y_err' ]
        y_low  = mass_dict[type_kk]['y_low' ]
        y_high = mass_dict[type_kk]['y_high']
        # Title
        title_kk = type_kk.title()
        # Plotting
        ax1.plot(   x_stat,
                    y_stat,
                    color=color_kk,
                    linestyle='-',
                    marker='o')
        ax1.fill_between(x_stat, y_low, y_high, color=color_kk, alpha=alpha,
                        label=title_kk)
    ##
    ## Legend
    leg = ax1.legend(loc='upper left', numpoints=1, frameon=False,
        prop={'size':14})
    leg.get_frame().set_facecolor('none')
    ## Ticks
    # X-axis
    ax_xaxis_major_loc = ticker.MultipleLocator(bin_width)
    ax1.xaxis.set_major_locator(ax_xaxis_major_loc)
    # Y-axis
    yaxis_major_ticker = 5
    yaxis_minor_ticker = 1
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



## --------- Main Function ------------##

def main(args):
    """
    Exploration plot for the catalogue being analyzed by ML algorithms.
    """
    ## Starting time
    start_time = datetime.now()
    ## Reading all elements and converting to python dictionary
    param_dict = vars(args)
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
    print('\n'+50*'='+'\n')
    for key, key_val in sorted(param_dict.items()):
        if key !='Prog_msg':
            print('{0} `{1}`: {2}'.format(Prog_msg, key, key_val))
    print('\n'+50*'='+'\n')
    ## Reading in catalogue
    catl_pd = catl_file_read_clean( param_dict,
                                    proj_dict,
                                    random_state=param_dict['seed'],
                                    shuffle_opt =param_dict['shuffle_opt'],
                                    dropna_opt  =param_dict['dropna_opt'],
                                    sample_frac =param_dict['sample_frac'])
    ###
    ### ------ Figures ------ ###
    ###
    ## Comparison of estimated group masses via HAM and Dynamical Masses
    group_mass_comparison(catl_pd, param_dict, proj_dict)






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
