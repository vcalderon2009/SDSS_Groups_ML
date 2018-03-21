#! /usr/bin/env python
# -*- coding: utf-8 -*-

# Victor Calderon
# Created      : 12/13/2017
# Last Modified: 12/13/2017
# Vanderbilt University
from __future__ import print_function, division, absolute_import
__author__     =['Victor Calderon']
__copyright__  =["Copyright 2017 Victor Calderon, SDSS Mocks Create - Make"]
__email__      =['victor.calderon@vanderbilt.edu']
__maintainer__ =['Victor Calderon']
"""
Script that runs the `SDSS Mocks Create` script
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
import os
import sys
import pandas as pd

# Extra-modules
from argparse import ArgumentParser
from argparse import HelpFormatter
from operator import attrgetter
import datetime

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
    description_msg = 'Script that creates the set of synthetic SDSS catalogues'
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
                        type=str,
                        default='2 5 10')
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
    ## Option for removing file
    parser.add_argument('-remove',
                        dest='remove_files',
                        help='Delete pickle files containing pair counts',
                        type=_str2bool,
                        default=False)
    ## CPU to use
    parser.add_argument('-cpu_frac',
                        dest='cpu_frac',
                        help='Fraction of CPUs to use',
                        type=float,
                        default=0.7)
    ## Verbose
    parser.add_argument('-v','--verbose',
                        dest='verbose',
                        help='Option to print out project parameters',
                        type=_str2bool,
                        default=False)
    ## Parsing Objects
    args = parser.parse_args()

    return args

def get_analysis_params(param_dict):
    """
    Parameters for the 1-halo conformity analysis

    Parameters
    -----------
    param_dict: python dictionary
        dictionary with project variables

    Returns
    --------
    params_pd: pandas DataFrame
        DataFrame with necessary parameters to run `sdss mocks create`
    """
    ##
    ## Array of values used for the analysis.
    ## Format: (name of variable, flag, value)
    #
    # For `mocks`
    params_arr = num.array([('hod_n'       ,'-hod_model_n' ,0         ),
                            ('halotype'    ,'-halotype'    ,'so'      ),
                            ('clf_method'  ,'-clf_method'  ,3         ),
                            ('sample'      ,'-sample'      ,'19'      ),
                            ('catl_type'   ,'-abopt'       ,'mr'      ),
                            ('cosmo_choice','-cosmo'       ,'LasDamas'),
                            ('nmin'        ,'-nmin'        ,1         ),
                            ('mass_factor' ,'-mass_factor' ,10        ),
                            ('remove_group','-remove_group',True      ),
                            ('dist_scales' ,'-dist_scales' ,'2 5 10'  ),
                            ('perf_opt'    ,'-perf'        ,False     ),
                            ('seed'        ,'-seed'        ,1         ),
                            ('cpu_frac'    ,'-cpu'         ,0.75      ),
                            ('remove_files','-remove'      ,'False'   ),
                            ('verbose'     ,'-v'           ,'False'   )])
    ##
    ## Converting to pandas DataFrame
    colnames = ['Name','Flag','Value']
    params_pd = pd.DataFrame(params_arr, columns=colnames)
    ##
    ## Options for `mocks`
    ##
    ## Sorting out DataFrame by `name`
    params_pd = params_pd.sort_values(by='Name').reset_index(drop=True)
    ##
    ## Number of distinct HOD model to use
    params_pd.loc[params_pd['Name']=='hod_n','Value'] = param_dict['hod_n']
    ##
    ## Type of the DM halo.
    params_pd.loc[params_pd['Name']=='halotype','Value'] = param_dict['halotype']
    ##
    ## CLF Method for assigning galaxy properties
    params_pd.loc[params_pd['Name']=='clf_method','Value'] = param_dict['clf_method']
    ##
    ## Choosing luminosity sample
    params_pd.loc[params_pd['Name']=='sample','Value'] = param_dict['sample']
    ##
    ## Choosing type of catalogue
    params_pd.loc[params_pd['Name']=='catl_type','Value'] = param_dict['catl_type']
    ##
    ## Cosmological model
    params_pd.loc[params_pd['Name']=='cosmo_choice','Value'] = param_dict['cosmo_choice']
    ##
    ## Minimum number of galaxies in galaxy group
    params_pd.loc[params_pd['Name']=='nmin','Value'] = param_dict['nmin']
    ##
    ## Mass factor for distance to closest cluster
    params_pd.loc[params_pd['Name']=='mass_factor','Value'] = param_dict['mass_factor']
    ##
    ## Option for removing group when calculating densities
    params_pd.loc[params_pd['Name']=='remove_group','Value'] = param_dict['remove_group']
    ##
    ## Distance scales used for calculating densities
    params_pd.loc[params_pd['Name']=='dist_scales','Value'] = param_dict['dist_scales']
    ##
    ## Option for using 'perfect' catalogues
    params_pd.loc[params_pd['Name']=='perf_opt','Value'] = param_dict['perf_opt']
    ##
    ## Option for setting the 'random seed'
    params_pd.loc[params_pd['Name']=='seed','Value'] = param_dict['seed']
    ##
    ## Choosing if to delete files
    if param_dict['remove_files']:
        ## Overwriting `remove_files` from `params_pd`
        params_pd.loc[params_pd['Name']=='remove_files','Value'] = 'True'
    ##
    ## Choosing the amount of CPUs
    params_pd.loc[params_pd['Name']=='cpu_frac','Value'] = param_dict['cpu_frac']
    ##
    ## Option for verbosity
    params_pd.loc[params_pd['Name']=='verbose','Value'] = param_dict['verbose']

    return params_pd

def get_exec_string(params_pd, param_dict):
    """
    Produces string be executed in the bash file

    Parameters
    -----------
    params_pd: pandas DataFrame
        DataFrame with necessary parameters to run `sdss mocks create`

    param_dict: python dictionary
        dictionary with project variables

    Returns
    -----------
    string_dict: python dictionary
        dictionary containing strings for `mocks`
    """
    ## Current directory
    working_dir = os.path.abspath(os.path.dirname(__file__))
    ## Choosing which file to run
    CATL_MAKE_file = 'catl_properties_calculations.py'
    ##
    ## Getting path to `CATL_MAKE_file`
    file_path = os.path.join(working_dir, CATL_MAKE_file)
    ##
    ## Check if File exists
    if os.path.isfile(file_path):
        pass
    else:
        msg = '{0} `CATL_MAKE_file` ({1}) not found!! Exiting!'.format(
            param_dict['Prog_msg'], file_path)
        raise ValueError(msg)
    ##
    ## Constructing string
    CATL_MAKE_mocks = 'python {0} '.format(file_path)
    for ii in range(params_pd.shape[0]):
        ## Appending to string
        CATL_MAKE_mocks += ' {0} {1}'.format(  params_pd['Flag' ][ii],
                                                params_pd['Value'][ii])
    ##
    ##
    string_dict = {'mocks':CATL_MAKE_mocks}

    return string_dict

def file_construction_and_execution(params_pd, param_dict):
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
    string_dict = get_exec_string(params_pd, param_dict)
    ##
    ## Parsing text that will go in file
    # Working directory
    working_dir = os.path.abspath(os.path.dirname(__file__))
    ## Obtaining path to file
    outfile_name = 'catl_properties_calculations_run.sh'
    outfile_path = os.path.join(working_dir, outfile_name)
    ##
    ## Opening file
    with open(outfile_path, 'wb') as out_f:
        out_f.write(b"""#!/usr/bin/env bash\n\n""")
        out_f.write(b"""## Author: Victor Calderon\n\n""")
        out_f.write( """## Last Edited: {0}\n\n""".format(now_str).encode())
        out_f.write(b"""### --- Variables\n""")
        out_f.write(b"""ENV_NAME="sdss_groups_ml"\n""")
        out_f.write( """WINDOW_NAME="SDSS_ML_Groups_Catls_Create"\n""".encode())
        out_f.write(b"""WINDOW_MOCKS="mocks"\n""")
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
        out_f.write(b"""activate=`which activate`\n""")
        out_f.write(b"""source $activate ${ENV_NAME}\n""")
        out_f.write(b"""###\n""")
        out_f.write(b"""### --- Python Strings\n""")
        out_f.write( """SDSS_mocks="{0}"\n""".format(string_dict['mocks']).encode())
        out_f.write(b"""###\n""")
        out_f.write(b"""### --- Deleting previous Screen Session\n""")
        out_f.write(b"""screen -S ${WINDOW_NAME} -X quit\n""")
        out_f.write(b"""###\n""")
        out_f.write(b"""### --- Screen Session\n""")
        out_f.write(b"""screen -mdS ${WINDOW_NAME}\n""")
        out_f.write(b"""## Mocks\n""")
        out_f.write(b"""screen -S ${WINDOW_NAME} -X screen -t ${WINDOW_MOCKS}\n""")
        out_f.write(b"""screen -S ${WINDOW_NAME} -p ${WINDOW_MOCKS} -X stuff $"source $activate ${ENV_NAME};"\n""")
        out_f.write(b"""screen -S ${WINDOW_NAME} -p ${WINDOW_MOCKS} -X stuff $"${SDSS_mocks}"\n""")
        out_f.write(b"""screen -S ${WINDOW_NAME} -p ${WINDOW_MOCKS} -X stuff $'\\n'\n""")
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
    ## Parameters for the analysis
    params_pd = get_analysis_params(param_dict)
    ##
    ## Running analysis
    file_construction_and_execution(params_pd, param_dict)

# Main function
if __name__=='__main__':
    ## Input arguments
    args = get_parser()
    # Main Function
    main(args)
