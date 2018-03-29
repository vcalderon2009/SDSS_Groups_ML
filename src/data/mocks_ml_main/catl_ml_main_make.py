#! /usr/bin/env python
# -*- coding: utf-8 -*-

# Victor Calderon
# Created      : 03/24/2018
# Last Modified: 03/24/2018
# Vanderbilt University
from __future__ import print_function, division, absolute_import
__author__     =['Victor Calderon']
__copyright__  =["Copyright 2017 Victor Calderon, SDSS Mocks Create - Make"]
__email__      =['victor.calderon@vanderbilt.edu']
__maintainer__ =['Victor Calderon']
"""
Script that performs the ML learning of group and galaxy catalogues
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
    ## Type of analysis to perform
    parser.add_argument('-a',
                        dest='analysis_type',
                        help='Type of analysis to make',
                        type=str,
                        choices=['training', 'plots'],
                        default='training')
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
    ## Testing size for ML
    parser.add_argument('-test_size',
                        dest='test_size',
                        help='Percentage size of the catalogue used for testing',
                        type=_check_pos_val,
                        default=0.25)
    ## Total number of K-folds, i.e. 'kf_splits'
    parser.add_argument('-kf_splits',
                        dest='kf_splits',
                        help="""
                        Total number of K-folds to perform. Must be larger 
                        than 2""",
                        type=int,
                        default=3)
    ## Number of hidden layers to use
    parser.add_argument('-hidden_layers',
                        dest='hidden_layers',
                        help='Number of hidden layers to use for neural network',
                        type=int,
                        default=1000)
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
    if param_dict['analysis_type'] == 'training':
        params_arr = num.array([('hod_n'        ,'-hod_model_n'  ,0          ),
                                ('halotype'     ,'-halotype'     ,'so'       ),
                                ('clf_method'   ,'-clf_method'   ,3          ),
                                ('sample'       ,'-sample'       ,'19'       ),
                                ('catl_type'    ,'-abopt'        ,'mr'       ),
                                ('cosmo_choice' ,'-cosmo'        ,'LasDamas' ),
                                ('nmin'         ,'-nmin'         ,1          ),
                                ('sample_frac'  ,'-sample_frac'  ,0.01       ),
                                ('perf_opt'     ,'-perf'         ,False      ),
                                ('test_size'    ,'-test_size'    ,0.25       ),
                                ('kf_splits'    ,'-kf_splits'    ,3          ),
                                ('kf_splits'    ,'-kf_splits'    ,3          ),
                                ('hidden_layers','-hidden_layers',1000       ),
                                ('shuffle_opt'  ,'-shuffle_opt'  ,'True'     ),
                                ('dropna_opt'   ,'-dropna_opt'   ,'True'     ),
                                ('pre_opt'      ,'-pre_opt'      ,'normalize'),
                                ('score_method' ,'-score_method' ,'threshold'),
                                ('seed'         ,'-seed'         ,1          ),
                                ('cpu_frac'     ,'-cpu'          ,0.75       ),
                                ('remove_files' ,'-remove'       ,'False'    ),
                                ('verbose'      ,'-v'            ,'False'    )])
    elif param_dict['analysis_type'] == 'plots':
        params_arr = num.array([('hod_n'       ,'-hod_model_n' ,0          ),
                                ('halotype'    ,'-halotype'    ,'so'       ),
                                ('clf_method'  ,'-clf_method'  ,3          ),
                                ('sample'      ,'-sample'      ,'19'       ),
                                ('catl_type'   ,'-abopt'       ,'mr'       ),
                                ('cosmo_choice','-cosmo'       ,'LasDamas' ),
                                ('nmin'        ,'-nmin'        ,1          ),
                                ('sample_frac' ,'-sample_frac' ,0.01       ),
                                ('perf_opt'    ,'-perf'        ,False      ),
                                ('n_predict'   ,'-n_predict'   ,1          ),
                                ('pre_opt'     ,'-pre_opt'     ,'normalize'),
                                ('score_method','-score_method','threshold'),
                                ('cpu_frac'    ,'-cpu'         ,0.75       ),
                                ('remove_files','-remove'      ,'False'    ),
                                ('verbose'     ,'-v'           ,'False'    )])
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
    ## Option for using 'perfect' catalogues
    params_pd.loc[params_pd['Name']=='perf_opt','Value'] = param_dict['perf_opt']
    ##
    ## Number of properties to predict. Default = 1
    params_pd.loc[params_pd['Name']=='n_predict','Value'] = param_dict['n_predict']
    ##
    ## Fraction of the total sample to use / read
    params_pd.loc[params_pd['Name']=='sample_frac','Value'] = param_dict['sample_frac']
    ##
    ## Choosing the amount of CPUs
    params_pd.loc[params_pd['Name']=='cpu_frac','Value'] = param_dict['cpu_frac']
    ##
    ## Option for verbosity
    params_pd.loc[params_pd['Name']=='verbose','Value'] = param_dict['verbose']
    ##
    ## Option for which preprocessing of the data to use
    params_pd.loc[params_pd['Name']=='pre_opt','Value'] = param_dict['pre_opt']
    ##
    ## Option for which preprocessing of the data to use
    params_pd.loc[params_pd['Name']=='score_method','Value'] = param_dict['score_method']
    ## Choosing if to delete files
    if param_dict['remove_files']:
        ## Overwriting `remove_files` from `params_pd`
        params_pd.loc[params_pd['Name']=='remove_files','Value'] = 'True'
    ##
    ## Only for `training`
    if param_dict['analysis_type'] == 'training':
        ##
        ## Option for setting the 'random seed'
        params_pd.loc[params_pd['Name']=='seed','Value'] = param_dict['seed']
        ##
        ## Option for shuffling the testing and training dataset when splitting
        params_pd.loc[params_pd['Name']=='shuffle_opt','Value'] = param_dict['shuffle_opt']
        ##
        ## Option for dropping any NaN's in the training/testing datasets
        params_pd.loc[params_pd['Name']=='dropna_opt','Value'] = param_dict['dropna_opt']
        ##
        ## Fraction of the sample to use for `Testing`
        params_pd.loc[params_pd['Name']=='test_size','Value'] = param_dict['test_size']
        ##
        ## Number of K-Folds to use when estimating the score of the model
        params_pd.loc[params_pd['Name']=='kf_splits','Value'] = param_dict['kf_splits']
        ##
        ## Number of K-Folds to use when estimating the score of the model
        params_pd.loc[params_pd['Name']=='hidden_layers','Value'] = param_dict['hidden_layers']
    ##
    ## Only for `plots`
    ##


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
    CATL_MAKE_file = param_dict['run_file_name']
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
    string_dict = {'catl_mocks':CATL_MAKE_mocks}

    return string_dict

def project_const(param_dict):
    """
    Contains the name of variables used in the project

    Parameters
    -----------
    param_dict: python dictionary
        dictionary with project variables

    Returns
    ----------
    param_dict: python dictionary
        dictionary with project variables
    """
    ## Constants
    # Environment name
    env_name        = 'sdss_groups_ml'
    ## Choosing script that will be ran
    if param_dict['analysis_type'] == 'training':
        window_name     = 'SDSS_ML_TRAINING'
        sub_window_name = 'ML_Training'
        run_file_name   = 'catl_ml_main.py'
    elif param_dict['analysis_type'] == 'plots':
        window_name     = 'SDSS_ML_PLOTTING'
        sub_window_name = 'ML_Plotting'
        run_file_name   = 'catl_ml_main_plots.py'
    ##
    ## Saving to dictionary
    param_dict['window_name'    ] = window_name
    param_dict['sub_window_name'] = sub_window_name
    param_dict['env_name'       ] = env_name
    param_dict['run_file_name'  ] = run_file_name

    return param_dict

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
    outfile_name = 'catl_properties_calculations_{0}_run.sh'.format(
        param_dict['analysis_type'])
    outfile_path = os.path.join(working_dir, outfile_name)
    ##
    ## Opening file
    with open(outfile_path, 'wb') as out_f:
        out_f.write(b"""#!/usr/bin/env bash\n\n""")
        out_f.write(b"""## Author: Victor Calderon\n\n""")
        out_f.write( """## Last Edited: {0}\n\n""".format(now_str).encode())
        out_f.write(b"""### --- Variables\n""")
        out_f.write( """ENV_NAME="{0}"\n""".format(param_dict['env_name']).encode())
        out_f.write( """WINDOW_NAME="{0}"\n""".format(param_dict['window_name']).encode())
        out_f.write( """SUB_WINDOW="{0}"\n""".format(param_dict['sub_window_name']).encode())
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
        out_f.write( """SCRIPT_CMD="{0}"\n""".format(string_dict['catl_mocks']).encode())
        out_f.write(b"""###\n""")
        out_f.write(b"""### --- Deleting previous Screen Session\n""")
        out_f.write(b"""screen -S ${WINDOW_NAME} -X quit\n""")
        out_f.write(b"""###\n""")
        out_f.write(b"""### --- Screen Session\n""")
        out_f.write(b"""screen -mdS ${WINDOW_NAME}\n""")
        out_f.write(b"""## Mocks\n""")
        out_f.write(b"""screen -S ${WINDOW_NAME} -X screen -t ${SUB_WINDOW}\n""")
        out_f.write(b"""screen -S ${WINDOW_NAME} -p ${SUB_WINDOW} -X stuff $"source $activate ${ENV_NAME};"\n""")
        out_f.write(b"""screen -S ${WINDOW_NAME} -p ${SUB_WINDOW} -X stuff $"${SCRIPT_CMD}"\n""")
        out_f.write(b"""screen -S ${WINDOW_NAME} -p ${SUB_WINDOW} -X stuff $'\\n'\n""")
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
    ## Project Constants
    param_dict = project_const(param_dict)
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
