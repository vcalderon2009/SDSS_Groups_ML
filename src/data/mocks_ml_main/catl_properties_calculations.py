#! /usr/bin/env python
# -*- coding: utf-8 -*-

# Victor Calderon
# Created      : 03/12/2018
# Last Modified: 03/12/2018
# Vanderbilt University
from __future__ import print_function, division, absolute_import
__author__     =['Victor Calderon']
__copyright__  =["Copyright 2017 Victor Calderon, "]
__email__      =['victor.calderon@vanderbilt.edu']
__maintainer__ =['Victor Calderon']
"""
Computes the necessary group and galaxy features for each 
galaxy and galaxy group in the catalogue
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
                    Computes the necessary group and galaxy features for each 
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
                        default='fof')
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
                        choices=range(1,1000),
                        metavar='[1-1000]',
                        default=1)
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
    ### URL for downloading files
    host_url = 'http://lss.phy.vanderbilt.edu/groups/data_vc/DR7'
    ### Mr volume-limited catalogue URL
    mr_url = os.path.join(  host_url,
                            'mr-vollim-catalogues',
                            'vollim_{0}_fib0.groups1'.format(sample_Mr))
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
    ##
    ## Saving to `param_dict`
    param_dict['sample_s'    ] = sample_s
    param_dict['sample_Mr'   ] = sample_Mr
    param_dict['host_url'    ] = host_url
    param_dict['mr_url'      ] = mr_url
    param_dict['vol_mr'      ] = vol_mr
    param_dict['cens'        ] = cens
    param_dict['sats'        ] = sats
    param_dict['speed_c'     ] = speed_c

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
    ## Output file for all catalogues
    catl_outdir    = os.path.join(  proj_dict['data_dir'],
                                    'processed',
                                    'SDSS',
                                    'mocks',
                                    'halos_{0}'.format(param_dict['halotype']),
                                    'hod_model_{0}'.format(param_dict['hod_n']),
                                    'clf_method_{0}'.format(param_dict['clf_method']),
                                    param_dict['catl_type'],
                                    param_dict['sample_Mr'],
                                    'merged_vac')
    ## Creating output folders for the catalogues
    merged_gal_dir          = os.path.join(catl_outdir, 'merged_vac'         )
    merged_gal_perf_dir     = os.path.join(catl_outdir, 'merged_vac_perf'    )
    merged_gal_all_dir      = os.path.join(catl_outdir, 'merged_vac_all'     )
    merged_gal_perf_all_dir = os.path.join(catl_outdir, 'merged_vac_perf_all')
    ##
    ## Creating Directories
    cu.Path_Folder(catl_outdir)
    cu.Path_Folder(merged_gal_dir)
    cu.Path_Folder(merged_gal_perf_dir)
    cu.Path_Folder(merged_gal_all_dir)
    cu.Path_Folder(merged_gal_perf_all_dir)
    ##
    ## Adding to `proj_dict`
    proj_dict['catl_outdir'            ] = catl_outdir
    proj_dict['merged_gal_dir'         ] = merged_gal_dir
    proj_dict['merged_gal_perf_dir'    ] = merged_gal_perf_dir
    proj_dict['merged_gal_all_dir'     ] = merged_gal_all_dir
    proj_dict['merged_gal_perf_all_dir'] = merged_gal_perf_all_dir

    return proj_dict

## --------- Cosmology and Halo Mass Function ------------##

def cosmo_create(cosmo_choice='Planck', H0=100., Om0=0.25, Ob0=0.04,
    Tcmb0=2.7255):
    """
    Creates instance of the cosmology used throughout the project.

    Parameters
    ----------
    cosmo_choice: string, optional (default = 'Planck')
        choice of cosmology
        Options:
            - Planck: Cosmology from Planck 2015
            - LasDamas: Cosmology from LasDamas simulation

    h: float, optional (default = 1.0)
        value for small cosmological 'h'.

    Returns
    ----------                  
    cosmo_obj: astropy cosmology object
        cosmology used throughout the project
    """
    ## Checking cosmology choices
    cosmo_choice_arr = ['Planck', 'LasDamas']
    assert(cosmo_choice in cosmo_choice_arr)
    ## Choosing cosmology
    if cosmo_choice == 'Planck':
        cosmo_model = astrocosmo.Planck15.clone(H0=H0)
    elif cosmo_choice == 'LasDamas':
        cosmo_model = astrocosmo.FlatLambdaCDM(H0=H0, Om0=Om0, Ob0=Ob0, 
            Tcmb0=Tcmb0)
    ## Cosmo Paramters
    cosmo_params         = {}
    cosmo_params['H0'  ] = cosmo_model.H0.value
    cosmo_params['Om0' ] = cosmo_model.Om0
    cosmo_params['Ob0' ] = cosmo_model.Ob0
    cosmo_params['Ode0'] = cosmo_model.Ode0
    cosmo_params['Ok0' ] = cosmo_model.Ok0

    return cosmo_model

## --------- Catalogue Analysis ------------##

def catalogue_analysis(ii, catl_ii_name, param_dict, proj_dict):
    """
    Function to analyze the catalogue and compute all of the group/galaxy 
    properties, and saves the resulting merged catalogue

    Parameters
    -----------
    ii: int
        integer of the catalogue being analyzed, after having order the 
        list of catalogues alphabetically.

    catl_ii_name: string
        name of the catalogue file being analyzed

    param_dict: python dictionary
        dictionary with `project` variables

    proj_dict: python dictionary
        Dictionary with current and new paths to project directories
    """
    ##
    ## Reading in `galaxy` and `group` catalogues, and merging 
    ## into a single DataFrame
    (   merged_gal_pd,
        memb_ii_pd   ,
        group_ii_pd  ) = cu.catl_sdss_merge(ii,
                                            catl_kind='mocks',
                                            catl_type=param_dict['catl_type'],
                                            sample_s=param_dict['sample_s'],
                                            halotype=param_dict['halotype'],
                                            clf_method=param_dict['clf_method'],
                                            hod_n=param_dict['hod_n'],
                                            perf_opt=param_dict['perf_opt'],
                                            print_filedir=False,
                                            return_memb_group=True)
    ##
    ## Constants
    n_gals   = len(memb_ii_pd )
    n_groups = len(group_ii_pd)
    ## Cartesian Coordinates for both `memb_ii_pd` and `group_ii_pd`
    (   memb_ii_pd ,
        group_ii_pd) = gals_cartesian(memb_ii_pd, group_ii_pd, param_dict)
    ## Creating new DataFrames
    group_mod_pd = pd.DataFrame({'groupid':num.sort(group_ii_pd['groupid'])})
    ## Indices for each galaxy group
    group_gals_dict = group_gals_idx(memb_ii_pd, group_ii_pd)
    ## Brightness of brightness galaxy
    group_mod_pd = group_brightest_gal( memb_ii_pd  , group_ii_pd    , 
                                        group_mod_pd, group_gals_dict)
    ## Brighness ratio between 1st and 2nd brightest galaxies in galaxy group
    group_mod_pd = group_brightness_gal_ratio(  memb_ii_pd  , group_ii_pd    ,
                                    group_mod_pd, group_gals_dict,
                                    nmin=param_dict['nmin'])
    ## Group Shape
    group_mod_pd = group_shape(memb_ii_pd, group_ii_pd, group_mod_pd, 
        group_gals_dict, nmin=param_dict['nmin'])



def group_gals_idx(memb_ii_pd, group_ii_pd):
    """
    Gets the indices of galaxies for each galaxy group

    Parameters
    ------------
    memb_ii_pd: pandas DataFrame
        DataFrame with info about galaxy members

    group_ii_pd: pandas DataFrame
        DataFrame with group properties

    Returns
    ------------
    group_gals_dict: python dictionary
        dictionary with indices of galaxies for each galaxy group
    """
    group_gals_dict = {}
    ## Looping over groups
    for group_kk in tqdm(range(len(group_ii_pd))):
        group_idx = memb_ii_pd.loc[memb_ii_pd['groupid']==group_kk].index.values
        group_gals_dict[group_kk] = group_idx

    return group_gals_dict

def gals_cartesian(memb_ii_pd, group_ii_pd, param_dict):
    """
    Computes the Cartesian coordinates of galaxies from the observer's 
    perspective

    Parameters
    -----------
    memb_ii_pd: pandas DataFrame
        DataFrame with info about galaxy members

    group_ii_pd: pandas DataFrame
        DataFrame with group properties

    param_dict: python dictionary
        dictionary with `project` variables

    Returns
    -----------
    memb_ii_pd: pandas DataFrame
        DataFrame with info about galaxy members + cartesian coordinates for 
        galaxies

    group_ii_pd: pandas DataFrame
        DataFrame with group properties + cartesian coordinates for 
        galaxies

    """
    ## Constants
    cosmo_model = param_dict['cosmo_model']
    speed_c     = param_dict['speed_c']
    ## Galaxy distances
    # Galaxies
    gals_dist = cosmo_model.comoving_distance(memb_ii_pd['cz']/speed_c).to(u.Mpc)
    # gals_dist = gals_dist.value
    # Groups
    groups_dist = cosmo_model.comoving_distance(group_ii_pd['GG_cen_cz']/speed_c).to(u.Mpc)
    # groups_dist = groups_dist.value
    ## Spherical to Cartesian coordinates
    # Galaxies
    gals_cart = SkyCoord(   ra=memb_ii_pd['ra'].values*u.deg,
                            dec=memb_ii_pd['dec'].values*u.deg,
                            distance=gals_dist).cartesian.xyz.value
    # Groups
    group_cart = SkyCoord(  ra=group_ii_pd['GG_cen_ra'].values*u.deg,
                            dec=group_ii_pd['GG_cen_dec'].values*u.deg,
                            distance=groups_dist).cartesian.xyz.value
    ## Saving to DataFrame
    # Galaxies
    memb_ii_pd.loc[:,'x'] = gals_cart[0]
    memb_ii_pd.loc[:,'y'] = gals_cart[1]
    memb_ii_pd.loc[:,'z'] = gals_cart[2]
    # Groups
    group_ii_pd.loc[:,'GG_x'] = group_cart[0]
    group_ii_pd.loc[:,'GG_y'] = group_cart[1]
    group_ii_pd.loc[:,'GG_z'] = group_cart[2]

    return memb_ii_pd, group_ii_pd








## --------- Galaxy Properties ------------##


## --------- Group Properties ------------##

## Brightness of the brightest galaxy
def group_brightest_gal(memb_ii_pd, group_ii_pd, group_mod_pd, 
    group_gals_dict):
    """
    Determines the brightness of the brightest galaxy in the galaxy group

    Parameters
    ------------
    memb_ii_pd: pandas DataFrame
        DataFrame with info about galaxy members

    group_ii_pd: pandas DataFrame
        DataFrame with group properties

    group_mod_pd: pandas DataFrame
        DataFrame, to which to add the group properties

    group_gals_dict: python dictionary
        dictionary with indices of galaxies for each galaxy group

    Returns
    ------------
    group_mod_pd: pandas DataFrame
        DataFrame, to which to add the group properties
    """
    ## Creating array for brightest galaxy
    group_mr_max_arr = num.zeros(len(group_ii_pd))
    ## Looping over all groups
    for group_kk in tqdm(range(len(group_mod_pd))):
        ## Group indices
        group_idx = group_gals_dict[group_kk]
        ## Brightest galaxy
        group_mr_max_arr[group_kk] = memb_ii_pd.loc[group_idx,'M_r'].min()
    ##
    ## Assigning it to DataFrame
    group_mod_pd.loc[:,'mr_brightest'] = group_mr_max_arr

    return group_mod_pd

## Brightness ratio of 1st and 2nd brightest galaxies in group
def group_brightness_gal_ratio(memb_ii_pd, group_ii_pd, group_mod_pd, 
    group_gals_dict, nmin=2):
    """
    Determines the brightness ratio of the 1st and 2nd brightest 
    galaxies in galaxy group

    Parameters
    ------------
    memb_ii_pd: pandas DataFrame
        DataFrame with info about galaxy members

    group_ii_pd: pandas DataFrame
        DataFrame with group properties

    group_mod_pd: pandas DataFrame
        DataFrame, to which to add the group properties

    group_gals_dict: python dictionary
        dictionary with indices of galaxies for each galaxy group

    Returns
    ------------
    group_mod_pd: pandas DataFrame
        DataFrame, to which to add the group properties
    """
    ## Array for brightness ratio
    group_lum_ratio_arr = num.zeros(len(group_ii_pd))
    ## Groups with number of galaxies larger than `nmin`
    groups_nmin_arr = group_ii_pd.loc[group_ii_pd['GG_ngals']>=nmin,'groupid']
    groups_nmin_arr = groups_nmin_arr.values
    ## Looping over all groups
    for group_kk in tqdm(groups_nmin_arr):
        ## Group indices
        group_idx = group_gals_dict[group_kk]
        ## Brightness ratio
        group_lum_arr = num.sort(cu.absolute_magnitude_to_luminosity(
                    memb_ii_pd.loc[group_idx,'M_r'].values, 'r'))[::-1]
        ## Ratio of Brightnesses
        group_lum_ratio_arr[group_kk] = 10**(group_lum_arr[0]-group_lum_arr[1])
    ##
    ## Assigning it to DataFrame
    group_mod_pd.loc[:,'mr_ratio'] = group_lum_ratio_arr

    return group_mod_pd

## Shape of the group
def group_shape(memb_ii_pd, group_ii_pd, group_mod_pd, group_gals_dict, 
    nmin=2):
    """
    Determines the brightness ratio of the 1st and 2nd brightest 
    galaxies in galaxy group

    Parameters
    ------------
    memb_ii_pd: pandas DataFrame
        DataFrame with info about galaxy members

    group_ii_pd: pandas DataFrame
        DataFrame with group properties

    group_mod_pd: pandas DataFrame
        DataFrame, to which to add the group properties

    group_gals_dict: python dictionary
        dictionary with indices of galaxies for each galaxy group

    Returns
    ------------
    group_mod_pd: pandas DataFrame
        DataFrame, to which to add the group properties
    """
    ## Cosmology
    cosmo_model = param_dict['cosmo_model']
    speed_c     = param_dict['speed_c']
    ## Array for brightness ratio
    group_shape_arr = num.zeros(len(group_ii_pd))
    ## Groups with number of galaxies larger than `nmin`
    groups_nmin_arr = group_ii_pd.loc[group_ii_pd['GG_ngals']>=nmin,'groupid']
    groups_nmin_arr = groups_nmin_arr.values
    ## Galaxy distances
    # Galaxies
    gals_dist = cosmo_model.comoving_distance(memb_ii_pd['cz']/speed_c).to(u.Mpc)
    gals_dist = gals_dist.value
    # Groups
    group_dist = cosmo_model.comoving_distance(group_ii_pd['GG_cen_cz']/speed_c).to(u.Mpc)
    group_dist = group_dist.value
    ## Coordinates of galaxies and groups
    gals_coords   = num.column_stack((  memb_ii_pd['ra' ],
                                        memb_ii_pd['dec'],
                                        gals_dist))
    groups_coords = num.column_stack((  group_ii_pd['GG_cen_ra' ],
                                        group_ii_pd['GG_cen_dec'],
                                        group_dist))
    ## Looping over all groups
    for group_kk in tqdm(groups_nmin_arr):
        ## Group indices
        group_idx    = group_gals_dict[group_kk]
        ## Galaxies in group
        memb_gals_kk = gals_coords[group_idx].T
        group_kk_pd  = groups_coords[group_kk]
        ## Cartian Coordinates
        (   sph_dict  ,
            coord_dict) = cu.Coord_Transformation(  memb_gals_kk[0],
                                                    memb_gals_kk[1],
                                                    memb_gals_kk[2],
                                                    group_kk_pd [0] ,
                                                    group_kk_pd [1] ,
                                                    group_kk_pd [2] ,
                                                    trans_opt = 4  )
        ## Joining cartesian coordinates
        x_kk = coord_dict['X'] - num.mean(coord_dict['X'])
        y_kk = coord_dict['Y'] - num.mean(coord_dict['Y'])
        z_kk = coord_dict['Z'] - num.mean(coord_dict['Z'])
        #
        gals_cart_arr = num.vstack((x_kk, y_kk, z_kk))
        ## Covariance matrix and eigenvectors
        cov           = num.cov(gals_cart_arr)
        evals, evecs  = num.linalg.eig(cov)
        ## Sort eigenvalues in decreasing order
        sort_indices  = num.argsort(evals)[::-1]
        evals_sorted  = num.real(evals[num.argsort(evals)[::-1]])
        ## Ratio of eigenvalues
        evals_ratio   = evals_sorted[-1]/evals_sorted[0]
        ## Saving elongation
        group_shape_arr[group_kk] = evals_ratio
    ##
    ## Assigning it to DataFrame
    group_mod_pd.loc[:,'shape'] = group_shape_arr

    return group_mod_pd



## --------- Multiprocessing ------------##

def multiprocessing_catls(catl_arr, param_dict, proj_dict, memb_tuples_ii):
    """
    Distributes the analysis of the catalogues into more than 1 processor

    Parameters:
    -----------
    catl_arr: numpy.ndarray, shape(n_catls,)
        array of paths to the catalogues files to analyze

    param_dict: python dictionary
        dictionary with `project` variables

    proj_dict: python dictionary
        Dictionary with current and new paths to project directories

    memb_tuples_ii: tuple
        tuple of catalogue indices to be analyzed
    """
    ## Program Message
    Prog_msg = param_dict['Prog_msg']
    ## Reading in Catalogue IDs
    start_ii, end_ii = memb_tuples_ii
    ##
    ## Looping the desired catalogues
    for ii, catl_ii in enumerate(catl_arr[start_ii : end_ii]):
        ## Choosing 1st catalogue
        if param_dict['verbose']:
            print('{0} Analyzing `{1}`\n'.format(Prog_msg, catl_ii))
        ## Extracting `name` of the catalogue
        catl_ii_name = os.path.splitext(os.path.split(catl_ii)[1])[0]
        ## Analaysis for the Halobias file
        catalogue_analysis(ii, catl_ii_name, param_dict, proj_dict)

## --------- Main Function ------------##

def main(args):
    """
    Computes the 
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
    ## Choosing cosmological model
    cosmo_model = cosmo_create(cosmo_choice=param_dict['cosmo_choice'])
    # Assigning the cosmological model to `param_dict`
    param_dict['cosmo_model'] = cosmo_model
    ##
    ## Printing out project variables
    print('\n'+50*'='+'\n')
    for key, key_val in sorted(param_dict.items()):
        if key !='Prog_msg':
            print('{0} `{1}`: {2}'.format(Prog_msg, key, key_val))
    print('\n'+50*'='+'\n')
    ##
    ## Looping over all galaxy catalogues
    # Paths to catalogues being analyzed
    (   catl_arr,
        n_catls ) = cu.extract_catls(   catl_kind='mocks',
                                        catl_type=param_dict['catl_type'],
                                        sample_s=param_dict['sample_s'],
                                        halotype=param_dict['halotype'],
                                        clf_method=param_dict['clf_method'],
                                        hod_n=param_dict['hod_n'],
                                        return_len=True,
                                        print_filedir=False )
    ##
    ## Changing `prog_bar` to `False`
    param_dict['prog_bar'] = False
    ##
    ### ---- Analyzing Catalogues ---- ###
    ##
    ## Using `multiprocessing` to analyze merged catalogues files
    ## Number of CPU's to use
    cpu_number = int(cpu_count() * param_dict['cpu_frac'])
    ## Defining step-size for each CPU
    if cpu_number <= n_catls:
        catl_step = int(n_catls / cpu_number)
        memb_arr = num.arange(0, n_catls+1, catl_step)
    else:
        catl_step = int((n_catls / cpu_number)**-1)
        memb_arr = num.arange(0, n_catls+1)
    ## Array with designated catalogue numbers for each CPU
    memb_arr[-1] = n_catls
    ## Tuples of the ID of each catalogue
    memb_tuples  = num.asarray([(memb_arr[xx], memb_arr[xx+1])
                            for xx in range(memb_arr.size-1)])
    ## Assigning `memb_tuples` to function `multiprocessing_catls`
    procs = []
    for ii in range(len(memb_tuples)):
        # Defining `proc` element
        proc = Process(target=multiprocessing_catls, 
                        args=(catl_arr, param_dict, 
                            proj_dict, memb_tuples[ii]))
        # Appending to main `procs` list
        procs.append(proc)
        proc.start()
    ##
    ## Joining `procs`
    for proc in procs:
        proc.join()
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
