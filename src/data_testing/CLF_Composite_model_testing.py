#! /usr/bin/env python
# -*- coding: utf-8 -*-

# Victor Calderon
# Created      : 2019-01-20
# Last Modified: 2019-01-20
# Vanderbilt University
from __future__ import absolute_import, division, print_function 
__author__     = ['Victor Calderon']
__copyright__  = ["Copyright 2018 Victor Calderon, "]
__email__      = ['victor.calderon@vanderbilt.edu']
__maintainer__ = ['Victor Calderon']
"""
Compares different types of scatter using the CLF formalism to compute.

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

from src.ml_tools import ReadML

import numpy as num
import math
import os
import sys
import pandas as pd
import matplotlib
matplotlib.use( 'Agg' )
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
plt.rc('text', usetex=True)
import seaborn as sns
#sns.set()
from tqdm import tqdm

# Halotools
from halotools.empirical_models import PrebuiltHodModelFactory
from halotools.sim_manager import CachedHaloCatalog, DownloadManager



# Extra-modules
import argparse
from argparse import ArgumentParser
from argparse import HelpFormatter
from operator import attrgetter
from tqdm import tqdm

## Functions
def directory_skeleton(proj_dict):
    """
    Creates the directory skeleton for the current project

    Parameters
    ----------
    param_dict : `dict`
        Dictionary with `project` variables

    proj_dict : `dict`
        Dictionary with info of the project that uses the
        `Data Science` Cookiecutter template.

    Returns
    ---------
    proj_dict : `dict`
        Dictionary with current and new paths to project directories
    """
    ## Output data - directory
    # outdir  = os.path.join( proj_dict['proc_dir'],
    #                         'clf_models')
    ## Figure directory
    fig_dir = os.path.join( proj_dict['plot_dir'],
                            'clf_models')
    # Creating directory
    dir_arr = [fig_dir]
    for dir_ii in dir_arr:
        if not (os.path.exists(dir_ii)):
            cfutils.Path_Folder(dir_ii)
    #
    # Adding to main project dictionary
    proj_dict['fig_dir'] = fig_dir
    # proj_dict['outdir' ] = outdir

    return proj_dict

## --------- CLF Models ------------##
def clf_models_analysis_main(proj_dict, sigma_c_init=0.1, sigma_c_final=2,
    sigma_c_int=0.1):
    """
    Computes the various CLF models that have different values of ``sigma_c``
    for the CLF (conditional luminosity function)

    Parameters
    -------------
    proj_dict : `dict`
        Dictionary with the paths of the project.

    sigma_c_init : `float`, optional
        Initial value for ``sigma_c`` to test. This variable is set to
        ``0.1`` by default.

    sigma_c_final : `float`, optional
        Final value for ``sigma_c`` to test. This variable is set to
        ``0.3`` by default.

    sigma_c_int : `float`, optional
        Value for the interval for ``sigma_c``. This variable is set to
        ``0.1`` by default.

    Returns
    ---------
    clf_models_dict : `dict`
        Dictionary with the final results for each of the various CLF
        models.
    """
    ## Array of ``sigma_c`` values
    sigma_c_arr = num.arange(   sigma_c_init,
                                sigma_c_final + 0.5*sigma_c_int,
                                sigma_c_int)
    sigma_c_arr = num.round(sigma_c_arr, 3)
    # Initialize dictionary
    clf_models_dict = {}
    # Looping over CLF models
    for ii, sigma_c_ii in enumerate(tqdm(sigma_c_arr)):
        # Running CLF analysis
        clf_models_dict[sigma_c_ii] = clf_models_analysis_calc(sigma_c_ii)

    return clf_models_dict

def clf_models_analysis_calc(sigma_c_ii, log_lum_cut=9):
    """
    Computes the CLF models and computes relevant statistics for such model

    Parameters
    -----------
    sigma_c_ii : `float`
        Value of the scatter in ``log(L)`` for central galaxies in the
        ``CLF`` formalism.

    log_lum_cut : `float`, optional
        Log value of the luminosity cut for the given CLF model.

    Returns
    ----------
    clf_ii_model_dict : `dict`
        Dictionary with the set of measurements and statistics for the given
        CLF model.
    """
    # Initializing `clf_ii_model_dict` dictionary
    clf_ii_model_dict = {}
    # Initializing model
    clf_ii_model = PrebuiltHodModelFactory('cacciato09', threshold=9)
    # Modifying `sigma_c` value
    clf_ii_model.param_dict['sigma'] = sigma_c_ii
    # Modifying model with LasDamas values
    clf_ii_model.param_dict['log_L_0'] = 9.952
    clf_ii_model.param_dict['log_M_1'] = 11.08
    clf_ii_model.param_dict['gamma_1'] = 3.510
    clf_ii_model.param_dict['gamma_2'] = 0.245
    clf_ii_model.param_dict['a_1']     = 0.5858
    clf_ii_model.param_dict['a_2']     = 0.8795
    clf_ii_model.param_dict['log_M_2'] = 14.30
    clf_ii_model.param_dict['b_0']     = -0.6954
    clf_ii_model.param_dict['b_1']     = 0.8672
    clf_ii_model.param_dict['b_2']     = -0.05834
    clf_ii_model.param_dict['delta_1'] = 0.0
    clf_ii_model.param_dict['delta_2'] = 0.0
    # Array of halo masses
    mhalo_arr = num.logspace(9, 15, 1000)
    # Mean number of centrals and satellites
    mean_cens = clf_ii_model.mean_occupation_centrals(prim_haloprop=mhalo_arr)
    mean_sats = clf_ii_model.mean_occupation_satellites(prim_haloprop=mhalo_arr)
    # CLF for centrals and satellites
    clf_cens  = clf_ii_model.clf_centrals(mhalo_arr)
    clf_sats  = clf_ii_model.clf_satellites(mhalo_arr)
    ##
    ## Saving values to dictionary
    clf_ii_model_dict['mhalo_arr'] = mhalo_arr
    clf_ii_model_dict['mean_cens'] = mean_cens
    clf_ii_model_dict['mean_sats'] = mean_sats
    clf_ii_model_dict['clf_cens' ] = clf_cens
    clf_ii_model_dict['clf_sats' ] = clf_sats

    return clf_ii_model_dict

## Plotting HOD's
def hod_models_plotting(clf_models_dict, proj_dict, fig_fmt='pdf',
    figsize=(15,6), fig_number=1):
    """
    Plots the HOD for `centrals` and `satellite` galaxies.

    Parameters
    -----------
    clf_models_dict : `dict`
        Dictionary with the final results for each of the various CLF
        models.

    proj_dict : `dict`
        Dictionary with the paths of the project.

    fig_fmt : `str`, optional (default = 'pdf')
        extension used to save the figure

    figsize : `tuple`, optional
        Size of the output figure. This variable is set to `(10,6)` by
        default.

    fig_number : `int`, optional
        Number of figure in the workflow. This variable is set to `1`
        by default.
    """
    file_msg = cfutils.Program_Msg('./')
    # Matplotlib option
    matplotlib.rcParams['axes.linewidth'] = 2.5
    matplotlib.rcParams['axes.edgecolor'] = 'black'
    # Figure figure
    fname = os.path.join(   proj_dict['fig_dir'],
                            'Figure_{0}_clf_models.{1}'.format(fig_number,
                                fig_fmt))
    # Number of models
    clf_models = num.sort(list(clf_models_dict.keys()))
    n_models   = len(clf_models)
    # Labels
    xlabel = r'\boldmath$\log M_{\textrm{vir}}\left[h^{-1} M_{\odot}\right]$'
    ylabel = r'\boldmath$\langle N \rangle$'
    # Fontsize
    label_size = 14
    ##
    ## Figure details
    plt.clf()
    plt.close()
    fig, axes = plt.subplots(ncols=2, nrows=1, figsize=figsize, sharex=True,
        sharey=True, facecolor='white')
    try:
        axes = axes.flatten()
    except:
        axes = [axes]
    # Limits
    mhalo_lim = (10.5, 14.9)
    ylim      = (0.1, 10)
    # Colors
    cm       = plt.cm.get_cmap('viridis')
    cm_arr   = [cm(kk/float(n_models)) for kk in range(n_models)]
    propssfr = dict(boxstyle='round', facecolor='white', alpha=0.7)
    # Looping over CLF models
    for ii, clf_ii in enumerate(clf_models):
        if (ii == 0):
            for ax_ii in axes:
                ax_ii.set_facecolor('white')
        cen_ax = axes[0]
        sat_ax = axes[1]
        ## CLF Model
        clf_model_ii = clf_models_dict[clf_ii]
        # Halo mass
        mhalo_arr = clf_model_ii['mhalo_arr']
        log_mhalo_arr = num.log10(mhalo_arr)
        ## ------ Centrals ------ ##
        # Mean occupation numbers
        mean_cens_ii = clf_model_ii['mean_cens']
        # Plotting occupation numbers
        cen_ax.plot(log_mhalo_arr, mean_cens_ii, color=cm_arr[ii],
            label=r'\boldmath$\sigma_{\textrm{c}}$ :' + '{0}'.format(clf_ii),
            linestyle='-', marker='o')
        ## ------ Satellites ------ ##
        # Mean occupation numbers
        mean_sats_ii = clf_model_ii['mean_sats']
        # Plotting occupation numbers
        sat_ax.plot(log_mhalo_arr, mean_sats_ii, color=cm_arr[ii],
            label=r'\boldmath$\sigma_{\textrm{c}}$ :' + '{0}'.format(clf_ii),
            linestyle='-')
        # Axes limits
        cen_ax.set_xlim(mhalo_lim)
        cen_ax.set_ylim(ylim)
        # Axes labels
        if (ii == 0):
            cen_ax.set_xlabel(xlabel, fontsize=label_size)
            sat_ax.set_xlabel(xlabel, fontsize=label_size)
            cen_ax.set_ylabel(ylabel, fontsize=label_size)
            plt.setp(sat_ax.get_yticklabels(), visible=False)
        # Logspace
        cen_ax.set_yscale('log')
        sat_ax.set_yscale('log')
    # Galaxy type text
    cen_ax.text(0.60, 0.10, 'Centrals', transform=cen_ax.transAxes,
        verticalalignment='top', color='black',
        bbox=propssfr, weight='bold', fontsize=10)
    sat_ax.text(0.60, 0.10, 'Satellites', transform=sat_ax.transAxes,
        verticalalignment='top', color='black',
        bbox=propssfr, weight='bold', fontsize=10)
    # Legend
    axes[0].legend(loc='upper left', ncol=3, numpoints=1, frameon=False,
        prop={'size': 14})
    axes[1].legend(loc='upper left', ncol=3, numpoints=1, frameon=False,
        prop={'size': 14})
    # Horizontal line
    cen_ax.axhline(y=0.5, color='black', linestyle='--', zorder=10)
    sat_ax.axhline(y=0.5, color='black', linestyle='--', zorder=10)
    # Removing any spacings
    plt.subplots_adjust(wspace=0., hspace=0.)
    ##
    ## Saving figure
    if (fig_fmt == 'pdf'):
        plt.savefig(fname, bbox_inches='tight', rasterize=True)
    else:
        plt.savefig(fname, bbox_inches='tight', dpi=400)
    ##
    ##
    # print('{0} Figure saved as: {1}'.format(file_msg, fname))
    print('{0} CLF Models Figure saved as: {1}'.format(file_msg, fname))
    plt.clf()
    plt.close()

## --------- Main Function ------------##

def main():
    """
    Compares various CLF models
    """
    # Initializing project dictionary
    param_dict = {}
    ## Creating Folder Structure
    proj_dict  = cwpaths.cookiecutter_paths('./')
    proj_dict  = directory_skeleton(proj_dict)
    ##
    ## Computing various CLF models
    clf_models_dict = clf_models_analysis_main(proj_dict)
    # HOD Plotting
    hod_models_plotting(clf_models_dict, proj_dict)


# Main function
if __name__=='__main__':
    # Main Function
    main()
