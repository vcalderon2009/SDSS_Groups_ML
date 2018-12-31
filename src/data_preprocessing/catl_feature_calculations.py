#! /usr/bin/env python
# -*- coding: utf-8 -*-

# Victor Calderon
# Created      : 2018-05-27
# Last Modified: 2018-05-28
# Vanderbilt University
from __future__ import print_function, division, absolute_import
__author__     =['Victor Calderon']
__copyright__  =["Copyright 2017 Victor Calderon, "]
__email__      =['victor.calderon@vanderbilt.edu']
__maintainer__ =['Victor Calderon']
"""
Computes the necessary group and galaxy features for each 
galaxy and galaxy group in the catalogue.
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
from cosmo_utils.mock_catalogues import mags_calculations as cmag

import numpy as num
import math
import os
import sys
import pandas as pd
import pickle
from tqdm import tqdm

# Extra-modules
import argparse
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
from glob import glob, iglob


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
                    galaxy and galaxy group in the catalogue.
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
    ## Factor by which to evaluate the distance to closest cluster.
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
    ## Option for calculating densities or not
    parser.add_argument('-dens_calc',
                        dest='dens_calc',
                        help='Option for calculating densities.',
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
    ## Remove master catalogue
    parser.add_argument('-rm_master',
                        dest='rm_master',
                        help='Option for removing the master catalogue',
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
    ##
    ## Output file for all catalogues
    catl_outdir    = os.path.join(  proj_dict['int_dir'],
                                    'merged_feat_catl',
                                    'SDSS',
                                    'mocks',
                                    'halos_{0}'.format(param_dict['halotype']),
                                    'dv_{0}'.format(param_dict['dv']),
                                    'hod_model_{0}'.format(param_dict['hod_n']),
                                    'clf_seed_{0}'.format(param_dict['clf_seed']),
                                    'clf_method_{0}'.format(param_dict['clf_method']),
                                    param_dict['catl_type'],
                                    param_dict['sample_Mr'],
                                    'dens_{0}'.format(param_dict['dens_calc']))
    ## Creating output folders for the catalogues
    merged_gal_dir          = os.path.join(catl_outdir, 'merged_vac')
    # merged_gal_perf_dir     = os.path.join(catl_outdir, 'merged_vac_perf'    )
    merged_gal_all_dir      = os.path.join(catl_outdir, 'merged_vac_combined')
    # merged_gal_perf_all_dir = os.path.join(catl_outdir, 'merged_vac_perf_all')
    ##
    ## Creating Directories
    cfutils.Path_Folder(catl_outdir)
    cfutils.Path_Folder(merged_gal_dir)
    # cfutils.Path_Folder(merged_gal_perf_dir)
    cfutils.Path_Folder(merged_gal_all_dir)
    # cfutils.Path_Folder(merged_gal_perf_all_dir)
    ## Removing files if necessary
    if param_dict['remove_files']:
        for catl_ii in [merged_gal_dir, merged_gal_all_dir]:
            file_list = glob('{0}/*'.format(catl_ii))
            for f in file_list:
                os.remove(f)
    ##
    ## Adding to `proj_dict`
    proj_dict['catl_outdir'            ] = catl_outdir
    proj_dict['merged_gal_dir'         ] = merged_gal_dir
    # proj_dict['merged_gal_perf_dir'    ] = merged_gal_perf_dir
    proj_dict['merged_gal_all_dir'     ] = merged_gal_all_dir
    # proj_dict['merged_gal_perf_all_dir'] = merged_gal_perf_all_dir

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

def catalogue_analysis(ii, catl_ii_name, box_n, param_dict, proj_dict, 
    ext='hdf5'):
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

    box_n : int
        Box number of the catalogue. This number tells the user from which 
        simulation box the mock catalogue comes.

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
                                                dv=param_dict['dv'],
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
        ## Brightness of brightest galaxy
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
        group_mod_pd, memb_ii_pd = group_radii(memb_ii_pd, group_ii_pd,
            group_mod_pd, group_gals_dict, nmin=param_dict['nmin'])
        ## Abundance matched mass of group
        group_mod_pd = group_general_prop(group_ii_pd, group_mod_pd)
        ## Velocity dispersion
        group_mod_pd = group_sigma_rmed(memb_ii_pd, group_ii_pd, group_mod_pd, 
            group_gals_dict, param_dict, nmin=param_dict['nmin'])
        ## Density of galaxies around group/cluster
        if param_dict['dens_calc']:
            group_mod_pd = group_galaxy_density(memb_ii_pd, group_ii_pd,
                group_mod_pd, group_gals_dict, 
                dist_scales=param_dict['dist_scales'],
                remove_group=param_dict['remove_group'])
        ## Dynamical mass estimates
        group_mod_pd = group_dynamical_mass(group_ii_pd, group_mod_pd)
        ## Distance to nearest cluster
        group_mod_pd = group_distance_closest_cluster(group_ii_pd, group_mod_pd,
            mass_factor=param_dict['mass_factor'])
        ## Pointing Method - Determining 'good' and 'bad' pointings
        group_mod_pd = group_halo_pointing(memb_ii_pd, group_ii_pd,
            group_mod_pd)
        ## Pointing Method - Assignment of halo masses
        group_mod_pd = group_mass_pointing(memb_ii_pd, group_ii_pd,
            group_mod_pd)
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
        ## Assigns the `box_n` variable to the dataframe 
        memb_group_pd.loc[:,'box_n'] = box_n
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
        gal_brightest_idx = (gals_g_lum_arr).idxmin()
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
    Determines the shape of the galaxy group in terms its
    eigenvalues and eigenvectors.

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
    ## Array for group shapes.
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
        coord_dict = cgeom.Coord_Transformation(    memb_gals_kk[0],
                                                    memb_gals_kk[1],
                                                    memb_gals_kk[2],
                                                    group_kk_pd [0],
                                                    group_kk_pd [1],
                                                    group_kk_pd [2],
                                                    trans_opt = 4  )
        ## Joining cartesian coordinates
        x_kk = coord_dict['x'] - num.mean(coord_dict['x'])
        y_kk = coord_dict['y'] - num.mean(coord_dict['y'])
        z_kk = coord_dict['z'] - num.mean(coord_dict['z'])
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
        # Checking type
        if num.isnan(evals_ratio):
            evals_ratio = 0.
        ## Saving elongation
        group_shape_arr[group_kk] = evals_ratio
    ##
    ## Assigning it to DataFrame
    group_mod_pd.loc[:,'shape'] = group_shape_arr

    return group_mod_pd

## Total and median radii of galaxy group
def group_radii(memb_ii_pd, group_ii_pd, group_mod_pd, group_gals_dict, 
    nmin=2, dist_opt='proj'):
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

    dist_opt : {'proj', '3D'} `str`, optional
        Option for which type of distance to use. This variable is set to
        ``proj`` by default.

        Options:
            - ``'proj'``: It computes the *projected* distances
            - ``'3D'`` : Computes the three-dimensional distances.
    
    Returns
    ------------
    group_mod_pd: pandas DataFrame
        DataFrame, to which to add the group properties

    memb_ii_pd : `pandas.DataFrame`
        DataFrame with info about galaxy members
    """
    # Constants
    pi_180 = num.pi / 180.
    ## Array for total and median radii of galaxy group
    gals_rp_arr     = num.zeros(len(memb_ii_pd))
    group_tot_r_arr = num.zeros(len(group_ii_pd))
    group_med_r_arr = num.zeros(len(group_ii_pd))
    group_rms_r_arr = num.zeros(len(group_ii_pd))
    ## Groups with number of galaxies larger than `nmin`
    groups_nmin_arr = group_ii_pd.loc[group_ii_pd['GG_ngals']>=nmin,'groupid']
    groups_nmin_arr = groups_nmin_arr.values
    ## Coordinates
    memb_cart_coords     = memb_ii_pd[['x','y','z']].values
    group_cart_coords    = group_ii_pd[['GG_x','GG_y','GG_z']].values
    ## Angular coordinates
    memb_coords     = memb_ii_pd[['ra', 'dec', 'cz']].values
    group_coords    = group_ii_pd[['GG_cen_ra', 'GG_cen_dec']].values
    ## Looping over all groups
    tq_msg = 'Group Radii: '
    for group_kk in tqdm(groups_nmin_arr, desc=tq_msg):
        ## Group indices
        group_idx    = group_gals_dict[group_kk]
        ## Constants
        inv_members = 1./group_idx.shape[0]
        ##
        ## - Projected distances - ##
        if (dist_opt == 'proj'):
            ## Galaxies in group
            memb_gals_kk = memb_coords [group_idx].T
            group_kk_pd  = group_coords[group_kk ].T
            # SkyCoord objects for member galaxies and group centre
            # c1 = SkyCoord(  ra=memb_gals_kk[0] * u.deg,
            #                 dec=memb_gals_kk[1] * u.deg,
            #                 frame='icrs')
            # c2 = SkyCoord(  ra=group_kk_pd[0] * u.deg,
            #                 dec=group_kk_pd[1] * u.deg,
            #                 frame='icrs')
            # rp_arr = 0.01 * memb_gals_kk[2] * c1.separation(c2).to(u.rad).value
            # # -- RMS    -- #
            # r_rms = (inv_members * num.sum(rp_arr**2))**0.5
            # # -- Median -- #
            # r_med = num.median(rp_arr)
            # # -- Total  -- #
            # r_tot = num.max(rp_arr)
            ## Saving to array
            # group_rms_r_arr[group_kk] = r_rms
            # group_med_r_arr[group_kk] = r_med
            # group_tot_r_arr[group_kk] = r_tot
            ##
            ## Different method
            num1  = num.cos(num.radians(group_kk_pd[1]))
            num1 *= num.cos(num.radians(memb_gals_kk[1]))
            num2  = num.sin(num.radians(group_kk_pd[1]))
            num2 *= num.sin(num.radians(memb_gals_kk[1]))
            num2 *= num.cos(num.radians(group_kk_pd[0] - memb_gals_kk[0]))
            # Projected distance
            cosDps = num1 + num2
            sinDps = (1. - cosDps**2)**0.5
            rp_arr = sinDps * memb_gals_kk[2] * 0.01
            # -- RMS    -- #
            r_rms = (inv_members * num.sum(rp_arr**2))**0.5
            # -- Median -- #
            r_med = num.median(rp_arr)
            # -- Total  -- #
            r_tot = num.max(rp_arr)
            ## Saving to array
            group_rms_r_arr[group_kk] = r_rms
            group_med_r_arr[group_kk] = r_med
            group_tot_r_arr[group_kk] = r_tot
            # Galaxy's projected distances
            gals_rp_arr[group_idx] = rp_arr
        ## - Three-dimensional distances - ##
        if (dist_opt == '3D'):
            ## Galaxies in group
            memb_gals_kk = memb_cart_coords [group_idx]
            group_kk_pd  = group_cart_coords[group_kk ]
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
    # group_mod_pd.loc[:, 'rrms_old'] = group_ii_pd['GG_rproj'].values
    group_mod_pd.loc[:,'r_rms'] = group_rms_r_arr
    group_mod_pd.loc[:,'r_med'] = group_med_r_arr
    group_mod_pd.loc[:,'r_tot'] = group_tot_r_arr
    # Projected distances - Galaxies
    if (dist_opt == 'proj'):
        memb_ii_pd.loc[:, 'rp'] = gals_rp_arr
    
    return group_mod_pd, memb_ii_pd

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
    param_dict, nmin=2, dist_opt='proj'):
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

    dist_opt : {'proj', '3D'} `str`, optional
        Option for which type of distance to use. This variable is set to
        ``proj`` by default.

        Options:
            - ``'proj'``: It computes the *projected* distances
            - ``'3D'`` : Computes the three-dimensional distances.

    Returns
    ------------
    group_mod_pd: pandas DataFrame
        DataFrame, to which to add the group properties
    """
    ## Array for the velocity dispersion at `r_med`
    group_sigma_rmed_arr = num.zeros(len(group_ii_pd))
    ## Groups with number of galaxies larger than `nmin`
    groups_nmin_arr = group_ii_pd.loc[group_ii_pd['GG_ngals']>=nmin,'groupid']
    groups_nmin_arr = groups_nmin_arr.values
    ## Galaxy columns
    if (dist_opt == 'proj'):
        gals_cols  = ['cz', 'rp']
        group_cols = ['GG_cen_cz', 'GG_ngals']
    elif (dist_opt == '3D'):
        gals_cols         = ['cz','x','y','z']
        group_cols        = ['GG_cen_cz', 'GG_ngals', 'GG_x', 'GG_y', 'GG_z']
    ##
    ## Selecting only chosen columns
    group_rmed_arr    = group_mod_pd['r_med'].values
    group_rmed_arr_sq = group_rmed_arr**2
    memb_coords       = memb_ii_pd[gals_cols].values
    group_coords      = group_ii_pd[group_cols].values
    ## Looping over all groups
    tq_msg = 'Group Sigma: '
    for group_kk in tqdm(groups_nmin_arr, desc=tq_msg):
        ## Group indices
        group_idx    = group_gals_dict[group_kk]
        ##
        ## -- Projected distance -- ##
        if (dist_opt == 'proj'):
            # Galaxies in group
            memb_gals_kk = memb_coords [group_idx].T
            group_kk_pd  = group_coords[group_kk ].T
            # If within `r_med`
            memb_rmed_mask = (memb_gals_kk[1] <= group_rmed_arr[group_kk])
            # Calculating velocity dispersion
            memb_rmed_kk = memb_gals_kk.T[memb_rmed_mask].T
            # Checking if more than 1 galaxy is in the `r_med` range
            if len(memb_rmed_kk.T) > 1:
                memb_dz2 = num.sum((memb_rmed_kk[0] - group_kk_pd[0])**2)
                czdisp_num = (memb_dz2/(memb_rmed_kk.T.shape[0] - 1))**(.5)
                czdisp_den = 1. + (group_kk_pd[0]/param_dict['speed_c'])
                czdisp_rmed = czdisp_num / czdisp_den
            else:
                czdisp_rmed = 0.
            ## Saving to array
            group_sigma_rmed_arr[group_kk] = czdisp_rmed
        ## -- 3D distance -- ##
        if (dist_opt == '3D'):
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

## Group Pointing
def group_halo_pointing(memb_ii_pd, group_ii_pd, group_mod_pd):
    """
    Determines the set of Group-Halo "Good Pointing", i.e. it determines the
    group, from which a Halo is mostly comprised, and vice-versa. If they
    both are "pointing" to each other, we consider them "Good" groups.

    Parameters
    -----------
    memb_ii_pd: pandas DataFrame
        DataFrame with info about galaxy members

    group_ii_pd: pandas DataFrame
        DataFrame with group properties

    group_mod_pd: pandas DataFrame
        DataFrame, to which to add the group properties

    Returns
    -----------
    group_mod_pd: pandas DataFrame
        DataFrame with group properties + "Pointing" information
    """
    # Constants
    gal_match    = int(1)
    gal_no_match = int(0)
    #
    # Generating catalogue keys
    (   gm_key,
        id_key,
        galtype_key) = cmcu.catl_keys(  catl_kind='mocks', return_type='list')
    ##
    ## Unique Halo and Group IDs
    haloid_unq = num.unique(memb_ii_pd['haloid'])
    groups_unq = num.unique(memb_ii_pd[id_key])
    n_halos    = len(haloid_unq)
    n_groups   = len(groups_unq)
    #
    # Initializing dictionaries
    g_dict = {}
    h_dict = {}
    ##
    ## -- Groups pointing to Halos -- ##
    for ii, group_ii in enumerate(tqdm(groups_unq)):
        # HaloIDs of the galaxies in the given group
        haloid_arr       = memb_ii_pd.loc[memb_ii_pd[id_key] == group_ii, 'haloid']
        # Most common halo in the group
        haloid_g         = int(Counter(haloid_arr).most_common(1)[0][0])
        # Saving most common HaloID in the galaxy group
        g_dict[group_ii] = haloid_g
    ## -- Halos pointing to Groups -- ##
    for kk, halo_kk in enumerate(tqdm(haloid_unq)):
        # HaloIDs of the galaxies in the given group
        groupid_arr     = memb_ii_pd.loc[memb_ii_pd['haloid'] == halo_kk, id_key]
        # Most common halo in the group
        groupid_g       = int(Counter(groupid_arr).most_common(1)[0][0])
        # Saving most common HaloID in the galaxy group
        h_dict[halo_kk] = groupid_g
    ##
    ## Determining if the Halo and Group "point" to each other
    match_dict = {}
    n_match    = int(0)
    n_nomatch  = int(0)
    # Looping over galaxy groups
    for ii, group_ii in enumerate(tqdm(groups_unq)):
        # Designated haloid
        halo_g  = g_dict[group_ii]
        group_h = h_dict[halo_g]
        if (group_ii == group_h):
            match_dict[group_ii] = gal_match
            n_match += 1
        else:
            match_dict[group_ii] = gal_no_match
            n_nomatch += 1
    #
    # Computing statistics
    match_frac    = 100.* (n_match / n_groups)
    no_match_frac = 100. * (n_nomatch / n_groups)
    # Populating final array for galaxy groups
    match_idx_g = [[] for x in range(n_groups)]
    # Looping over groups
    for ii, group_ii in enumerate(tqdm(groups_unq)):
        match_idx_g[ii] = match_dict[group_ii]
    #
    # Assigning it to main DataFrame
    group_mod_pd.loc[:, 'pointing'] = match_idx_g

    return group_mod_pd

## Group Mass assignment from the 'pointing' method
def group_mass_pointing(memb_ii_pd, group_ii_pd, group_mod_pd):
    """
    Determines the group mass of a galaxy group based on the "Pointing"
    method, i.e. the halo that contributes the most galaxies to the galaxy
    group.

    Parameters
    -----------
    memb_ii_pd: pandas DataFrame
        DataFrame with info about galaxy members

    group_ii_pd: pandas DataFrame
        DataFrame with group properties

    group_mod_pd: pandas DataFrame
        DataFrame, to which to add the group properties

    Returns
    -----------
    group_mod_pd: pandas DataFrame
        DataFrame with group properties + "Pointing" mass information
    """
    # Generating catalogue keys
    (   gm_key,
        id_key,
        galtype_key) = cmcu.catl_keys(  catl_kind='mocks', return_type='list')
    ##
    ## Unique Halo and Group IDs
    haloid_unq = num.unique(memb_ii_pd['haloid'])
    groups_unq = num.unique(memb_ii_pd[id_key])
    n_halos    = len(haloid_unq)
    n_groups   = len(groups_unq)
    # Determine the halo that contributes the most of the halo
    g_dict           = {}
    g_dict['mhalo' ] = {}
    g_dict['haloid'] = {}
    #
    # Looping over galaxy groups
    for ii, group_ii in enumerate(tqdm(groups_unq)):
        # HaloIDs in group
        haloid_group_ii = memb_ii_pd.loc[memb_ii_pd[id_key] == group_ii, 'haloid']
        # Most common halo ID
        haloid_mc   = int(Counter(haloid_group_ii).most_common(1)[0][0])
        mhalo_g     = memb_ii_pd.loc[memb_ii_pd['haloid'] == haloid_mc, 'M_h']
        mhalo_g_unq = float(num.unique(mhalo_g))
        # Saving to dictionaries
        g_dict['haloid'][group_ii] = haloid_mc
        g_dict['mhalo' ][group_ii] = mhalo_g_unq
    ##
    ## Assigning to `group_mod_pd`
    haloid_g = [[] for x in range(n_groups)]
    mhalo_g  = [[] for x in range(n_groups)]
    for ii, group_ii in enumerate(tqdm(groups_unq)):
        haloid_g[ii] = g_dict['haloid'][group_ii]
        mhalo_g [ii] = g_dict['mhalo' ][group_ii]
    # To DataFrame
    group_mod_pd.loc[:, 'haloid_point'] = haloid_g
    group_mod_pd.loc[:, 'mhalo_point' ] = mhalo_g

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
    Prog_msg = param_dict['Prog_msg']
    ## Saving catalogue
    cfreaders.pandas_df_to_hdf5_file(memb_group_pd, merged_vac_filename,
        key='/gals_groups')
    ## Print message
    if param_dict['verbose']:
        print('{0} Saving `{1}`\n'.format(Prog_msg, merged_vac_filename))
    cfutils.File_Exists(merged_vac_filename)

## Testing if `merged` file is complete, i.e. if there are missing 
## files from any catalogue
def test_df_merged_dir(param_dict, proj_dict, n_catls, ext='hdf5'):
    """
    Checks for the completeness of all catalogues

    Parameters
    ------------
    param_dict : `dict`
        Dictionary with `project` variables

    proj_dict : `dict`
        Dictionary with current and new paths to project directories

    n_catls : `int`
        Number of `expected` catalogues in the folder

    ext : `str`, optional (default = 'hdf5')
        File extension used when saving catalogues to files

    Returns
    ------------
    param_dict : `dict`
        Original dictionary with added key `merged_vac_save` On whether 
        or not to create a new merged Value-Added catalogue.
    """
    file_msg = param_dict['Prog_msg']
    ## List of catalogues
    files_arr = cfutils.Index(proj_dict['merged_gal_dir'], '.{0}'.format(ext))
    # Checking agains the expected number of files
    if (param_dict['rm_master'] or param_dict['remove_files']):
        merged_vac_save = True
    else:
        if (files_arr.size == n_catls):
            merged_vac_save = False
        else:
            merged_vac_save = True
    #
    # Saving to dictionary
    param_dict['merged_vac_save'] = merged_vac_save

    return param_dict

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
    file_msg = param_dict['Prog_msg']
    ## List of catalogues
    files_arr = cfutils.Index(proj_dict['merged_gal_dir'], '.{0}'.format(ext))
    file_key  = '/gals_groups'
    group_key = 'groupid'
    file_str_arr = [    param_dict['sample_Mr'],
                        param_dict['halotype'],
                        param_dict['hod_n'],
                        param_dict['dv'],
                        param_dict['clf_method'],
                        param_dict['clf_seed'],
                        param_dict['catl_type'],
                        param_dict['cosmo_choice'],
                        param_dict['nmin'],
                        param_dict['mass_factor'],
                        param_dict['perf_opt'],
                        ext]
    file_str  = '{0}_halo_{1}_hodn_{2}_dv_{3}_clfm_{4}_clfseed_{5}_'
    file_str += 'ctype_{6}_cosmo_{7}_nmin_{8}_massf_{9}_perf_{10}_'
    file_str += 'merged_vac_all.{11}'
    filename  = file_str.format(*file_str_arr)
    ## Saving to file
    filepath = os.path.join(proj_dict['merged_gal_all_dir'],
                            filename)
    # Checking if file exists
    if os.path.exists(filepath):
        if param_dict['rm_master']:
            os.remove(filepath)
    #
    # Only running if necessary
    if param_dict['merged_vac_save']:
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
        cfreaders.pandas_df_to_hdf5_file(catl_pd_main, filepath, key=file_key)
        cfutils.File_Exists(filepath)
    else:
        msg = '{0} `Merged catalogue` found in : {1}'.format(file_msg,
            filepath)
        print(msg)

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
    idx_arr  = num.array(range(start_ii, end_ii), dtype=int)
    ## Catalogue array
    catl_arr_ii = catl_arr[start_ii : end_ii]
    ##
    ## Looping the desired catalogues
    for (ii, catl_ii) in zip(idx_arr, catl_arr_ii):
        ## Converting index to main `int`
        ii = int(ii)
        ## Choosing 1st catalogue
        if param_dict['verbose']:
            print('{0} Analyzing `{1}`\n'.format(Prog_msg, catl_ii))
        ## Extracting `name` of the catalogue
        catl_ii_name = os.path.splitext(os.path.split(catl_ii)[1])[0]
        ## Box Number
        box_n = int(catl_ii_name.split('_')[1])
        ## Analaysis for the Halobias file
        catalogue_analysis(ii, catl_ii_name, box_n, param_dict, proj_dict)

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
                                        dv=param_dict['dv'],
                                        clf_method=param_dict['clf_method'],
                                        hod_n=param_dict['hod_n'],
                                        clf_seed=param_dict['clf_seed'],
                                        return_len=True,
                                        print_filedir=False )
    #
    # Checking if a new merged VAC is needed
    param_dict = test_df_merged_dir(param_dict, proj_dict, n_catls)
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
