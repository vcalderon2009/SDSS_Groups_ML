#! /usr/bin/env python
# -*- coding: utf-8 -*-

# Victor Calderon
# Created      : 03/12/2018
# Last Modified: 03/21/2018
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

# Importing Modules
from cosmo_utils       import mock_catalogues as cm
from cosmo_utils       import utils           as cu
from cosmo_utils.utils import file_utils      as cfutils
from cosmo_utils.utils import file_readers    as cfreaders
from cosmo_utils.utils import work_paths      as cwpaths
from cosmo_utils.utils import stats_funcs     as cstats
from cosmo_utils.utils import geometry        as cgeom
from cosmo_utils.mock_catalogues import catls_utils as cmcu
from cosmo_utils.mock_catalogues import mags_calculations as cmag

import numpy as num
import math
import os
import sys
import pandas as pd
import pickle
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
from scipy.spatial import cKDTree
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
    ## Radii used for estimating densities
    parser.add_argument('-dist_scales',
                        dest='dist_scales',
                        help="""
                        List of distance scales to use when calculating 
                        densities""",
                        type=float,
                        nargs='+',
                        default=[2.0, 5.0, 10.0])
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
    ## Main Directories
    # External Directory
    ext_dir       = os.path.join( proj_dict['data_dir'], 'external')
    # Processes Directory
    processed_dir = os.path.join( proj_dict['data_dir'], 'processed')
    # Interim Directory
    int_dir       = os.path.join( proj_dict['data_dir'], 'interim')
    # Raw Directory
    raw_dir       = os.path.join( proj_dict['data_dir'], 'raw')
    ##
    ## Output file for all catalogues
    catl_outdir    = os.path.join(  proj_dict['data_dir'],
                                    'processed',
                                    'SDSS',
                                    'mocks',
                                    'halos_{0}'.format(param_dict['halotype']),
                                    'hod_model_{0}'.format(param_dict['hod_n']),
                                    'clf_seed_{0}'.format(param_dict['clf_seed']),
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
    cfutils.Path_Folder(ext_dir)
    cfutils.Path_Folder(processed_dir)
    cfutils.Path_Folder(int_dir)
    cfutils.Path_Folder(raw_dir)
    cfutils.Path_Folder(catl_outdir)
    cfutils.Path_Folder(merged_gal_dir)
    cfutils.Path_Folder(merged_gal_perf_dir)
    cfutils.Path_Folder(merged_gal_all_dir)
    cfutils.Path_Folder(merged_gal_perf_all_dir)
    ## Removing files if necessary
    if param_dict['remove_files']:
        for catl_ii in [merged_gal_dir, merged_gal_perf_dir, merged_gal_all_dir, merged_gal_perf_all_dir]:
            file_list = glob('{0}/*'.format(catl_ii))
            for f in file_list:
                os.remove(f)
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
    tq_msg = 'Group Gals IDX: '
    for group_kk in tqdm(range(len(group_ii_pd)), desc=tq_msg):
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
                            distance=gals_dist).cartesian.xyz.to(u.Mpc).value
    # Groups
    group_cart = SkyCoord(  ra=group_ii_pd['GG_cen_ra'].values*u.deg,
                            dec=group_ii_pd['GG_cen_dec'].values*u.deg,
                            distance=groups_dist).cartesian.xyz.to(u.Mpc).value
    ## Saving to DataFrame
    # Galaxies
    memb_ii_pd.loc[:,'x'     ] = gals_cart[0]
    memb_ii_pd.loc[:,'y'     ] = gals_cart[1]
    memb_ii_pd.loc[:,'z'     ] = gals_cart[2]
    memb_ii_pd.loc[:,'dist_c'] = gals_dist
    # Groups
    group_ii_pd.loc[:,'GG_x'     ] = group_cart[0]
    group_ii_pd.loc[:,'GG_y'     ] = group_cart[1]
    group_ii_pd.loc[:,'GG_z'     ] = group_cart[2]
    group_ii_pd.loc[:,'GG_dist_c'] = groups_dist

    return memb_ii_pd, group_ii_pd

def catalogue_analysis(ii, catl_ii_name, param_dict, proj_dict, ext='hdf5'):
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

    ext: string, optional (default = 'hdf5')
        Extension to use for the resulting catalogues
    """
    ## Merged DataFrame - Filename
    merged_vac_filename = os.path.join( proj_dict['merged_gal_dir'],
                                        '{0}_merged_vac.{1}'.format(
                                            catl_ii_name, ext))
    if os.path.exists(merged_vac_filename):
        if param_dict['remove_files']:
            os.remove(merged_vac_filename)
            vac_create_opt = True
        else:
            vac_create_opt = False
    else:
        vac_create_opt = True
    ##
    ## Onl running analysis if `merged_vac_filename` not present
    if vac_create_opt:
        ##
        ## Reading in `galaxy` and `group` catalogues, and merging 
        ## into a single DataFrame
        (   merged_gal_pd,
            memb_ii_pd   ,
            group_ii_pd  ) = cmcu.catl_sdss_merge(
                                                ii,
                                                catl_kind='mocks',
                                                catl_type=param_dict['catl_type'],
                                                sample_s=param_dict['sample_s'],
                                                halotype=param_dict['halotype'],
                                                clf_method=param_dict['clf_method'],
                                                hod_n=param_dict['hod_n'],
                                                clf_seed=param_dict['clf_seed'],
                                                perf_opt=param_dict['perf_opt'],
                                                print_filedir=False,
                                                return_memb_group=True)
        merged_gal_pd = None
        ##
        ## Constants
        n_gals   = len(memb_ii_pd )
        n_groups = len(group_ii_pd)
        ## Cartesian Coordinates for both `memb_ii_pd` and `group_ii_pd`
        (   memb_ii_pd ,
            group_ii_pd) = gals_cartesian(memb_ii_pd, group_ii_pd, param_dict)
        ## ---- Galaxy Groups - Properties ---- ##
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
            group_gals_dict, param_dict, nmin=param_dict['nmin'])
        ## Total and median radius of the group
        group_mod_pd = group_radii(memb_ii_pd, group_ii_pd, group_mod_pd, 
            group_gals_dict, nmin=param_dict['nmin'])
        ## Abundance matched mass of group
        group_mod_pd = group_general_prop(group_ii_pd, group_mod_pd)
        ## Velocity dispersion
        group_mod_pd = group_sigma_rmed(memb_ii_pd, group_ii_pd, group_mod_pd, 
            group_gals_dict, param_dict, nmin=param_dict['nmin'])
        ## Density of galaxies around group/cluster
        group_mod_pd = group_galaxy_density(memb_ii_pd, group_ii_pd, group_mod_pd,
            group_gals_dict, dist_scales=param_dict['dist_scales'],
            remove_group=param_dict['remove_group'])
        ## Dynamical mass estimates
        group_mod_pd = group_dynamical_mass(group_ii_pd, group_mod_pd)
        ## Distance to nearest cluster
        group_mod_pd = group_distance_closest_cluster(group_ii_pd, group_mod_pd,
            mass_factor=param_dict['mass_factor'])
        ##
        ## ---- Member Galaxies - Properties---- ##
        ## Creating new 'members' DataFrame
        memb_mod_pd = pd.DataFrame({'idx':num.arange(len(memb_ii_pd))})
        ## Adding to member galaxies
        memb_mod_pd = galaxy_dist_centre(memb_ii_pd, group_ii_pd, group_mod_pd, 
            group_gals_dict, memb_mod_pd)
        ## Adding member galaxy properties
        memb_mod_pd = galaxy_properties(memb_ii_pd, memb_mod_pd)
        ## Brightest galaxy in group
        memb_mod_pd = gal_brightest_in_group(memb_ii_pd, group_ii_pd, 
            group_mod_pd, group_gals_dict, memb_mod_pd)
        ## ---- Combining Member and Group Properties ---- ##
        ## Merging group and galaxy DataFrames
        memb_group_pd = memb_group_merging(memb_mod_pd, group_mod_pd)
        ## Saving DataFrames
        merging_df_save(merged_vac_filename, memb_group_pd, param_dict, 
            proj_dict)
    else:
        msg = '{0} Catalogue `{1}` ... Created\n'.format(
            param_dict['Prog_msg'], merged_vac_filename)

## --------- Galaxy Properties ------------##

## Distance to the group's centre
def galaxy_dist_centre(memb_ii_pd, group_ii_pd, group_mod_pd, 
    group_gals_dict, memb_mod_pd):
    """
    Determines the distance to the centre of the galaxy group

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

    memb_mod_pd: pandas DataFrame
        DataFrame, to which to add the `member galaxy` properties

    Returns
    ------------
    memb_mod_pd: pandas DataFrame
        DataFrame, to which to add the `member galaxy` properties
    """
    ## Array of distances to centre of the galaxy group
    gals_group_dist_sq_arr = num.zeros(len(memb_ii_pd))
    ## Galaxy columns
    gals_cols    = ['cz','x','y','z']
    memb_coords  = memb_ii_pd[gals_cols].values
    # Group Columns
    group_cols   = ['GG_cen_cz', 'GG_ngals', 'GG_x', 'GG_y', 'GG_z']
    group_coords = group_ii_pd[group_cols].values
    ## Looping over all groups
    tq_msg = 'Galaxy Dist Centre: '
    for group_kk in tqdm(range(len(group_ii_pd)), desc=tq_msg):
        ## Group indices
        group_idx    = group_gals_dict[group_kk]
        ## Galaxies in group
        memb_gals_kk = memb_coords [group_idx]
        group_kk_pd  = group_coords[group_kk ]
        ## Cartesian Coordinates
        memb_cart_arr  = memb_gals_kk[:,1:]
        group_cart_rsh = group_kk_pd [2:].reshape(1,3)
        if len(memb_cart_arr) > 1:
            ## If within `r_med`
            memb_dist_sq_arr = num.sum((memb_cart_arr - group_cart_rsh)**2,axis=1)
        else:
            memb_dist_sq_arr = 0.
        ## Assigning to each galaxy
        gals_group_dist_sq_arr[group_idx] = memb_dist_sq_arr

    ##
    ## Distance square root
    gals_group_dist_arr = gals_group_dist_sq_arr**(0.5)
    ## Adding to `memb_mod_pd`
    memb_mod_pd.loc[:,'dist_centre_group'] = gals_group_dist_arr

    return memb_mod_pd

## General properties for member galaxies
def galaxy_properties(memb_ii_pd, memb_mod_pd):
    """
    Assigns `general` galaxy properties to `memb_mod_pd`
    It assigns:
        - Luminosity/Absolute magnitude of the member galaxy
        - Central/Satellite Designation

    Parameters
    ------------
    memb_ii_pd: pandas DataFrame
        DataFrame with info about galaxy members

    memb_mod_pd: pandas DataFrame
        DataFrame, to which to add the `member galaxy` properties

    Returns
    ------------
    memb_mod_pd: pandas DataFrame
        DataFrame, to which to add the `member galaxy` properties
    """
    ## Galaxy Properties
    gals_cols         = ['galtype', 'halo_ngal', 'M_r'      , 'logssfr',\
                         'M_h'    , 'g_galtype', 'halo_rvir', 'g_r'    ,\
                         'groupid', 'sersic'   ]
    gals_cols_pd_copy = (memb_ii_pd.copy())[gals_cols]
    # Merging DataFrames
    memb_mod_pd = pd.merge(memb_mod_pd, gals_cols_pd_copy,
                    left_index=True, right_index=True)

    return memb_mod_pd

## Brightest galaxy in the group or not
def gal_brightest_in_group(memb_ii_pd, group_ii_pd, group_mod_pd, 
    group_gals_dict, memb_mod_pd):
    """
    Determines whether or not a galaxy is the brightest galaxy in 
    a galaxy group

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

    memb_mod_pd: pandas DataFrame
        DataFrame, to which to add the `member galaxy` properties

    Returns
    ------------
    memb_mod_pd: pandas DataFrame
        DataFrame, to which to add the galaxy properties
    """
    ## Constants
    bright_opt     = int(1)
    not_bright_opt = int(0)
    ## Creating array for brightest galaxy
    group_gal_brightest_gal = num.ones(len(memb_ii_pd)) * not_bright_opt
    ## IDs of brightest galaxies in groups
    group_brightest_idx = group_mod_pd['mr_brightest_idx'].values.astype(int)
    ## Determining if brighest galaxy
    group_gal_brightest_gal[group_brightest_idx] = bright_opt
    ## Saving to DataFrame
    memb_mod_pd.loc[:,'g_brightest'] = group_gal_brightest_gal.astype(int)

    return memb_mod_pd

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
    group_mr_max_arr     = num.zeros(len(group_ii_pd))
    group_mr_max_idx_arr = num.zeros(len(group_ii_pd))
    ## Looping over all groups
    tq_msg = 'Group Bright. Gal: '
    for group_kk in tqdm(range(len(group_mod_pd)), desc=tq_msg):
        ## Group indices
        group_idx = group_gals_dict[group_kk]
        ## Galaxy luminosities in galaxy group
        gals_g_lum_arr = memb_ii_pd.loc[group_idx,'M_r']
        ## Index of the brightest galaxy
        gal_brightest_idx = num.argmin(gals_g_lum_arr)
        ## Brightest galaxy
        group_mr_max_arr    [group_kk] = gals_g_lum_arr.loc[gal_brightest_idx]
        group_mr_max_idx_arr[group_kk] = gal_brightest_idx
    ##
    ## Assigning it to DataFrame
    group_mod_pd.loc[:,'mr_brightest'    ] = group_mr_max_arr
    group_mod_pd.loc[:,'mr_brightest_idx'] = group_mr_max_idx_arr.astype(int)

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
    tq_msg = 'Mr Ratio: '
    for group_kk in tqdm(groups_nmin_arr, desc=tq_msg):
        ## Group indices
        group_idx = group_gals_dict[group_kk]
        ## Brightness ratio
        group_lum_arr = num.sort(cmag.absolute_magnitude_to_luminosity(
                    memb_ii_pd.loc[group_idx,'M_r'].values, 'r'))[::-1]
        ## Ratio of Brightnesses
        group_lum_ratio_arr[group_kk] = 10**(group_lum_arr[0]-group_lum_arr[1])
    ##
    ## Assigning it to DataFrame
    group_mod_pd.loc[:,'mr_ratio'] = group_lum_ratio_arr

    return group_mod_pd

## Shape of the group
def group_shape(memb_ii_pd, group_ii_pd, group_mod_pd, group_gals_dict, 
    param_dict, nmin=2):
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
    tq_msg = 'Group Shape: '
    for group_kk in tqdm(groups_nmin_arr, desc=tq_msg):
        ## Group indices
        group_idx    = group_gals_dict[group_kk]
        ## Galaxies in group
        memb_gals_kk = gals_coords[group_idx].T
        group_kk_pd  = groups_coords[group_kk]
        ## Cartian Coordinates
        (   sph_dict  ,
            coord_dict) = cgeom.Coord_Transformation(   memb_gals_kk[0],
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

## Total and median radii of galaxy group
def group_radii(memb_ii_pd, group_ii_pd, group_mod_pd, group_gals_dict, 
    nmin=2):
    """
    Determines the total and median radii of the galaxy groups with 
    group richness of "ngal > `nmin`"

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
    ## Array for total and median radii of galaxy group
    group_tot_r_arr = num.zeros(len(group_ii_pd))
    group_med_r_arr = num.zeros(len(group_ii_pd))
    ## Groups with number of galaxies larger than `nmin`
    groups_nmin_arr = group_ii_pd.loc[group_ii_pd['GG_ngals']>=nmin,'groupid']
    groups_nmin_arr = groups_nmin_arr.values
    ## Coordinates
    memb_coords     = memb_ii_pd[['x','y','z']].values
    group_coords    = group_ii_pd[['GG_x','GG_y','GG_z']].values
    ## Looping over all groups
    tq_msg = 'Group Radii: '
    for group_kk in tqdm(groups_nmin_arr, desc=tq_msg):
        ## Group indices
        group_idx    = group_gals_dict[group_kk]
        ## Galaxies in group
        memb_gals_kk = memb_coords [group_idx]
        group_kk_pd  = group_coords[group_kk ]
        # Reshaped `group_kk_pd`
        group_kk_pd_rsh = group_kk_pd.reshape(1,3)
        ## Calculating total and median distances
        # Distances
        gals_cen_dist_sq = num.sum((memb_gals_kk - group_kk_pd_rsh)**2, axis=1)
        gals_cen_dist    = gals_cen_dist_sq**(.5)
        # Total distance
        r_tot = num.max(gals_cen_dist)
        r_med = num.median(gals_cen_dist)
        ## Saving to array
        group_tot_r_arr[group_kk] = r_tot
        group_med_r_arr[group_kk] = r_med
    ##
    ## Assigning it to DataFrame
    group_mod_pd.loc[:,'r_tot'] = group_tot_r_arr
    group_mod_pd.loc[:,'r_med'] = group_med_r_arr

    return group_mod_pd

## General properties for groups
def group_general_prop(group_ii_pd, group_mod_pd):
    """
    Assigns `general` group properties to `group_mod_pd`.
    It assigns:
        - Abundance matched mass
        - Group richness
        - Sigma_v (total velocity dispersion)
        - Total Brightness
    
    Parameters
    ------------
    group_ii_pd: pandas DataFrame
        DataFrame with group properties

    group_mod_pd: pandas DataFrame
        DataFrame, to which to add the group properties
    
    Returns
    ------------
    group_mod_pd: pandas DataFrame
        DataFrame, to which to add the group properties
    """
    ## Group properties
    groups_cols = [ 'GG_ngals'  , 'GG_sigma_v', 
                    'GG_rproj'  , 'GG_M_r'    ,
                    'GG_M_group', 'GG_logssfr',
                    'groupid'   ]
    groups_cols_mod = [xx.replace('GG_','') for xx in groups_cols]
    ## Copy of `group_ii_pd`
    group_ii_pd_copy = (group_ii_pd.copy())[groups_cols]
    ## Renaming columns
    groups_cols_dict = dict(zip(groups_cols,
                                [xx.replace('GG_','') for xx in groups_cols]))
    group_ii_pd_copy = group_ii_pd_copy.rename(columns=groups_cols_dict)
    ## Merging both DataFrames
    group_mod_pd = pd.merge(group_mod_pd, group_ii_pd_copy[groups_cols_mod], 
                            on='groupid',how='left')

    return group_mod_pd

## Total and median radii of galaxy group
def group_sigma_rmed(memb_ii_pd, group_ii_pd, group_mod_pd, group_gals_dict, 
    param_dict, nmin=2):
    """
    Determines the total and median radii of the galaxy groups with 
    group richness of "ngal > `nmin`"

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

    param_dict: python dictionary
        dictionary with `project` variables

    nmin: int, optional (default = 2)
        minimum number of galaxies in the galaxy group

    Returns
    ------------
    group_mod_pd: pandas DataFrame
        DataFrame, to which to add the group properties
    """
    ## Array for total and median radii of galaxy group
    group_sigma_rmed_arr = num.zeros(len(group_ii_pd))
    ## Groups with number of galaxies larger than `nmin`
    groups_nmin_arr = group_ii_pd.loc[group_ii_pd['GG_ngals']>=nmin,'groupid']
    groups_nmin_arr = groups_nmin_arr.values
    ## Median radius of groups
    group_rmed_arr_sq  = group_mod_pd['r_med'].values**2
    # Galaxy columns
    gals_cols    = ['cz','x','y','z']
    memb_coords  = memb_ii_pd[gals_cols].values
    # Group Columns
    group_cols   = ['GG_cen_cz', 'GG_ngals', 'GG_x', 'GG_y', 'GG_z']
    group_coords = group_ii_pd[group_cols].values
    ## Looping over all groups
    tq_msg = 'Group Sigma: '
    for group_kk in tqdm(groups_nmin_arr, desc=tq_msg):
        ## Group indices
        group_idx    = group_gals_dict[group_kk]
        ## Galaxies in group
        memb_gals_kk = memb_coords [group_idx]
        group_kk_pd  = group_coords[group_kk ]
        ## Cartesian Coordinates
        memb_cart_arr  = memb_gals_kk[:,1:]
        group_cart_rsh = group_kk_pd[2:].reshape(1,3)
        ## If within `r_med`
        memb_dist_sq_arr = num.sum((memb_cart_arr - group_cart_rsh)**2,axis=1)
        ## Selecting dispersion of galaxies
        memb_rmed_mask = (memb_dist_sq_arr <= group_rmed_arr_sq[group_kk])
        ## Calculating velocity dispersion
        memb_rmed_kk = memb_gals_kk[memb_rmed_mask]
        # Checking if more than 1 galaxy is in the `r_med` range
        ## Velocity dispersion of galaxies within group's `r_med`
        if len(memb_rmed_kk) > 1:
            memb_dz2   = num.sum((memb_rmed_kk[:,0] - group_kk_pd[0])**2)
            czdisp_num = (memb_dz2/(memb_rmed_kk.shape[0] - 1))**(.5)
            czdisp_den = 1. + (group_kk_pd[0]/param_dict['speed_c'])
            czdisp_rmed = czdisp_num / czdisp_den
        else:
            czdisp_rmed = 0.
        ## Saving to array
        group_sigma_rmed_arr[group_kk] = czdisp_rmed
    ##
    ## Assigning it to DataFrame
    group_mod_pd.loc[:,'sigma_v_rmed'] = group_sigma_rmed_arr

    return group_mod_pd

## Density of galaxies around group/cluster
def group_galaxy_density(memb_ii_pd, group_ii_pd, group_mod_pd, group_gals_dict, 
    dist_scales=[2, 5, 10], remove_group=True):
    """
    Determines the total and median radii of the galaxy groups with 
    group richness of "ngal > `nmin`"

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

    dist_scale: int, optional (default = 2)
        scale, at which to measure the density (number of galaxy over 
        volume) around the galaxy cluster

    remove_group: boolean, optional (default = True)
        option for removing the group being analyzed for when determining 
        the galaxy density around the group/cluster.

    Returns
    ------------
    group_mod_pd: pandas DataFrame
        DataFrame, to which to add the group properties
    """
    ## Distance scale
    dist_scales = num.array(dist_scales)
    ## Array for number of neighbours at some distance scale
    group_counts_r_scale_arr = num.zeros((len(group_ii_pd),len(dist_scales)))
    ## Group ID's array
    groupid_arr  = num.sort(group_ii_pd['groupid'].values)
    ## Galaxy ID array
    memb_id_arr  = memb_ii_pd.index.values
    # Galaxy columns
    gals_cols    = ['x','y','z']
    memb_coords  = memb_ii_pd[gals_cols].values
    # Group Columns
    group_cols   = ['GG_x', 'GG_y', 'GG_z']
    group_coords = group_ii_pd[group_cols].values
    ## Looping over all groups
    tq_msg = 'Group Dens: '
    for group_kk in tqdm(groupid_arr, desc=tq_msg):
        ## Group indices
        group_idx    = group_gals_dict[group_kk]
        # Indices of galaxies NOT in the galaxy group
        if remove_group:
            gals_idx = num.delete(memb_id_arr, group_idx)
        else:
            gals_idx = memb_id_arr
        ## Galaxies in group
        memb_gals_kk = memb_coords [gals_idx]
        group_kk_pd  = group_coords[group_kk ].reshape(1,3)
        ## KDTree object
        memb_kdtree  = cKDTree(memb_gals_kk)
        group_kdtree = cKDTree(group_kk_pd)
        # Galaxies within radius
        group_counts_arr = group_kdtree.count_neighbors(memb_kdtree,
                                                        dist_scales)
        ## Saving to array
        group_counts_r_scale_arr[group_kk] = group_counts_arr
    ##
    ## Volumes of each sphere
    vol_scales = (4*num.pi / 3.)*num.array(dist_scales)**3
    ## Densities at those scales
    group_density_arr = group_counts_r_scale_arr / vol_scales
    ## Assigning to DataFrame
    for zz, r_zz in enumerate(dist_scales):
        group_mod_pd.loc[:,'dens_{0}'.format(r_zz)] = group_density_arr.T[zz]

    return group_mod_pd

## Dynamical mass estimate
def group_dynamical_mass(group_ii_pd, group_mod_pd):
    """
    Calculated the dynamical mass of galaxy groups.
    Formula from:   Girardi et al. (1998)
                        http://adsabs.harvard.edu/abs/1998ApJ...505...74G
    
    Parameters
    ------------
    group_ii_pd: pandas DataFrame
        DataFrame with group properties

    group_mod_pd: pandas DataFrame
        DataFrame, to which to add the group properties
    
    Returns
    ------------
    group_mod_pd: pandas DataFrame
        DataFrame, to which to add the group properties
    """
    ## Unit constant
    unit_const = ((3*num.pi/2.) * ((u.km/u.s)**2) * (u.Mpc) / ac.G).to(u.Msun)
    unit_const_val = unit_const.value
    ## Dynamical mass at the median radius
    mdyn_med = ((group_mod_pd['sigma_v_rmed']**2)*group_mod_pd['r_med']).values
    ## Dynamical mass at the virial radius
    mdyn_proj = ((group_ii_pd['GG_sigma_v']**2)*group_ii_pd['GG_rproj']).values
    ##
    ## Correcting units
    # Median Radius
    mdyn_med_mod                   = mdyn_med * unit_const_val
    mdyn_med_mod_idx               = num.where(mdyn_med_mod != 0)[0]
    mdyn_med_mod_idx_val           = num.log10(mdyn_med_mod[mdyn_med_mod_idx])
    mdyn_med_mod[mdyn_med_mod_idx] = mdyn_med_mod_idx_val
    # virial radius
    mdyn_proj_mod                   = mdyn_proj * unit_const_val
    mdyn_proj_mod_idx               = num.where(mdyn_proj_mod != 0)[0]
    mdyn_proj_mod_idx_val           = num.log10(mdyn_proj_mod[mdyn_proj_mod_idx])
    mdyn_proj_mod[mdyn_proj_mod_idx] = mdyn_proj_mod_idx_val
    ##
    ## Assigning to `group_mod_pd`
    group_mod_pd.loc[:,'mdyn_rmed' ] = mdyn_med_mod
    group_mod_pd.loc[:,'mdyn_rproj'] = mdyn_proj_mod

    return group_mod_pd

## Distance to nearest cluster
def group_distance_closest_cluster(group_ii_pd, group_mod_pd, mass_factor=10):
    """
    Calculated the distance to the nearest galaxy cluster
    
    Parameters
    ------------
    group_ii_pd: pandas DataFrame
        DataFrame with group properties

    group_mod_pd: pandas DataFrame
        DataFrame, to which to add the group properties

    mass_factor: int, optional (default = 10)
    
    Returns
    ------------
    group_mod_pd: pandas DataFrame
        DataFrame, to which to add the group properties
    """
    # Group Columns
    groups_cols      = ['GG_x', 'GG_y', 'GG_z', 'GG_M_group']
    groups_coords    = group_ii_pd[groups_cols].values
    groups_cart_arr  = groups_coords[:,0:3].copy()
    grups_mgroup_arr = groups_coords[:,  3].copy()
    # Distance to nearest "massive" cluster
    groups_dist_sq_cluster_arr = num.zeros(len(groups_coords))
    # Looping over all galaxy groups
    tq_msg = 'Dist Cluster: '
    for group_zz in tqdm(range(len(groups_coords)), desc=tq_msg):
        ## Group mass
        factor_zz = grups_mgroup_arr[group_zz] + num.log10(mass_factor)
        ## Cartesian Coordinates of the groups 'group_zz'
        cart_zz = groups_cart_arr[group_zz].reshape(1,3)
        ## Selecting next closest "massive cluster"
        mas_clusters_coords = groups_coords[grups_mgroup_arr >= factor_zz][:,0:3]
        if len(mas_clusters_coords) > 0:
            ## Distance squared
            cluster_dists_sq = num.sum((mas_clusters_coords - cart_zz)**2, axis=1)
            ## Minimum distance
            cluster_dist_sq_min = num.min(cluster_dists_sq)
        else:
            cluster_dist_sq_min = 0.
        ## Minimum distance
        groups_dist_sq_cluster_arr[group_zz] = cluster_dist_sq_min
    ##
    ## Square root of `distances_sq`
    groups_dist_cluster_arr = groups_dist_sq_cluster_arr**(.5)
    ## Assigning to 'group_mod_pd'
    group_mod_pd.loc[:,'dist_cluster'] = groups_dist_cluster_arr

    return group_mod_pd

## --------- Galaxy and Group-related functions ------------##

## Merging member galaxy and group DataFrames
def memb_group_merging(memb_mod_pd, group_mod_pd):
    """
    Merges the two datasets into a single dataset

    Parameters
    ------------
    memb_mod_pd: pandas DataFrame
        DataFrame with information on galaxies

    group_mod_pd: pandas DataFrame
        DataFrame with information on galaxy groups

    Returns
    ------------
    memb_group_pd: pandas DataFrame
        DataFrame with the merged info on galaxies and groups
    """
    ## Constants
    memb_merge_key  = 'groupid'
    memb_cols_drop  = ['idx']
    group_cols_drop = ['groupid','mr_brightest_idx']
    ## Removing unnecessary columns
    memb_mod_pd     = memb_mod_pd.drop( memb_cols_drop, axis=1)
    group_mod_pd    = group_mod_pd.drop(group_cols_drop, axis=1)
    ## Renaming `group` M_r column
    group_cols_arr  = dict(zip(group_mod_pd, ['GG_'+xx for xx in group_mod_pd]))
    group_mod_pd    = group_mod_pd.rename(columns=group_cols_arr)
    ## Merging DataFrames
    memb_group_pd   = pd.merge( memb_mod_pd,
                                group_mod_pd, 
                                left_on=memb_merge_key,
                                right_index=True)

    return memb_group_pd

## Saving merged DataFrame
def merging_df_save(merged_vac_filename, memb_group_pd, param_dict, proj_dict):
    """
    Saves merged DataFrame to file on disk

    Parameters
    -------------
    merged_vac_filename: string
        name of the catalogue file being analyzed

    memb_group_pd: pandas DataFrame
        DataFrame with the merged info on galaxies and groups

    param_dict: python dictionary
        dictionary with `project` variables

    proj_dict: python dictionary
        Dictionary with current and new paths to project directories
    """
    ## Saving catalogue
    cfreaders.pandas_df_to_hdf5_file(memb_group_pd, merged_vac_filename,
        key='/gals_groups')
    ## Print message
    if param_dict['verbose']:
        print('{0} Saving `{1}`\n'.format(Prog_msg, merged_vac_filename))
    cfutils.File_Exists(merged_vac_filename)

## Merging all Datasets into a single Dataset
def catl_df_merging(param_dict, proj_dict, ext='hdf5'):
    """
    Merges all of the catalogues into a single dataset

    Parameters
    ------------
    param_dict: python dictionary
        dictionary with `project` variables

    proj_dict: python dictionary
        Dictionary with current and new paths to project directories

    ext: string, optional (default = 'hdf5')
        file extension used when saving catalogues to files
    """
    ## List of catalogues
    files_arr = cfutils.Index(proj_dict['merged_gal_dir'], '.{0}'.format(ext))
    file_key  = '/gals_groups'
    group_key = 'groupid'
    file_str_arr = [param_dict['sample_Mr'],    param_dict['hod_n'],
                        param_dict['clf_method'],   param_dict['cosmo_choice'],
                        param_dict['nmin']      ,   param_dict['halotype'],
                        param_dict['perf_opt']  ,   ext]
    file_str  = '{0}_hodn_{1}_clf_{2}_cosmo_{3}_nmin_{4}_halotype_{5}_perf_{6}'
    file_str += 'merged_vac_all.{7}'
    filename  = file_str.format(*file_str_arr)
    ## Concatenating DataFrames
    group_id_tot = 0
    gals_id_tot  = 0
    ## Looping over catalogues
    tq_msg = 'Catl Merging: '
    for catl_ii in tqdm(range(files_arr.size), desc=tq_msg):
        catl_pd_ii = cfreaders.read_hdf5_file_to_pandas_DF(files_arr[catl_ii])
        if catl_ii == 0:
            catl_pd_main = catl_pd_ii.copy()
        else:
            catl_pd_ii.loc[:,group_key] += group_id_tot
            catl_pd_main = pd.concat([catl_pd_main, catl_pd_ii], 
                ignore_index=True)
        ## Increasing number of groups
        group_id_tot += num.unique(catl_pd_ii[group_key]).size
    ## Saving to file
    filepath = os.path.join(proj_dict['merged_gal_all_dir'],
                            filename)
    ## Saving to file
    cfreaders.pandas_df_to_hdf5_file(catl_pd_main, filepath, key=file_key)
    cfutils.File_Exists(filepath)

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
    ## Index value
    idx_arr  = num.array(range(start_ii, end_ii))
    ## Catalogue array
    catl_arr_ii = catl_arr[start_ii : end_ii]
    ##
    ## Looping the desired catalogues
    for (ii, catl_ii) in zip(idx_arr, catl_arr_ii):
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
    # proj_dict  = directory_skeleton(param_dict, cwpaths.cookiecutter_paths(__file__))
    proj_dict  = directory_skeleton(param_dict, cwpaths.cookiecutter_paths('./'))
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
        n_catls ) = cmcu.extract_catls( catl_kind='mocks',
                                        catl_type=param_dict['catl_type'],
                                        sample_s=param_dict['sample_s'],
                                        halotype=param_dict['halotype'],
                                        clf_method=param_dict['clf_method'],
                                        hod_n=param_dict['hod_n'],
                                        clf_seed=param_dict['clf_seed'],
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
    ## Combining catalogues into a single 'master' file
    print('{0} Merging Catalogues ....'.format(Prog_msg))
    catl_df_merging(param_dict, proj_dict, ext='hdf5')
    print('{0} Merging Catalogues .... Done'.format(Prog_msg))
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
