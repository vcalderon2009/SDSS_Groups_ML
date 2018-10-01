#! /usr/bin/env python
# -*- coding: utf-8 -*-

# Victor Calderon
# Created      : 2018-06-01
# Last Modified: 2018-06-01
# Vanderbilt University
from __future__ import absolute_import, division, print_function
__author__     = ['Victor Calderon']
__copyright__  = ["Copyright 2018 Victor Calderon, "]
__email__      = ['victor.calderon@vanderbilt.edu']
__maintainer__ = ['Victor Calderon']
__all__        = ["ReadML"]
"""
Utilities for reading in the ML outputs for this project.
"""
# Importing Modules
from cosmo_utils       import mock_catalogues as cm
from cosmo_utils       import utils           as cu
from cosmo_utils.utils import file_utils      as cfutils
from cosmo_utils.utils import file_readers    as cfreaders
from cosmo_utils.utils import work_paths      as cwpaths
from cosmo_utils.ml    import ml_utils        as cmlu
from cosmo_utils.mock_catalogues import catls_utils as cmcu
from cosmo_utils.utils import gen_utils    as cgu

import numpy as num
import os
import pandas as pd
import pickle

# Functions

class ReadML(object):
    """
    Reads in the multiple outputs of the ML data preprocessing and
    analysis steps for this project. This class is mainly for handling
    the aspects of reading/writing output files for this project.
    """
    def __init__(self, **kwargs):
        """
        Parameters
        -----------
        hod_n : `int`, optional
            Number of the HOD model to use. This value is set to `0` by default

        halotype : {'so', 'fof'}, `str`
            Type of dark matter halo to analyze. This value is set to `so`
            by default.

        clf_method : {1, 2, 3}, `int`
            Method for assigning galaxy properties to mock galaxies.
            This variable is set to `1` by default.

            Options:
                - 1 :   Independent assignment of (g-r), sersic, logssfr
                - 2 :   (g-r) decides active/passive designation and draws values independently.
                - 3 :   (g-r) decides active/passive designation, and assigns other galaxy properties for that given galaxy.

        clf_seed : `int`
            Random seed to be used for CLF.
            This variable is set to `1235` by default.

        dv : `float`
            Difference between galaxy and mass velocity profiles
            (v_g-v_c)/(v_m-v_c).

        sample : {'19', '20', '21'}, `str`, optional
            SDSS Luminosity sample to analyze. This variable is set to '19' by
            default.

        catl_type : {'mr', 'mstar'}, `str`, optional
            Type of Abund. Matching used in catalogue. This variable is set to
            'mr' by default.

        cosmo_choice : {'LasDamas', 'Planck'}, `str`, optional
            Choice of Cosmology. This variable is set to `LasDamas` by default.

        nmin : {2-1000}, `int`
            Minimum number of galaxies in a galaxy group. The range for this
            variable is from `2` to `1000`. This variable is set to `2` by
            default.

        mass_factor : {2 - 100}, `int`
            Factor by which to evaluate the distance to closest cluster.
            This variable is set to `10` by default.

        n_predict : {1, 3}, `int`, optional
            Number of properties to predict. This variable is set to `1` by
            default. See the documentation for more details.

        shuffle_opt : `bool`
            If True, the data will be shuffled before being splitted.
            If True, `stratify` must be set to `None`.

        dropna_opt : `bool`
            If True, all the `NaN` will be dropped from the dataset.

        pre_opt : {'min_max', 'standard', 'normalize', 'no'} `str`, optional
            Type of preprocessing to do on `feat_arr`.

            Options:
                - 'min_max' : Turns `feat_arr` to values between (0,1)
                - 'standard' : Uses `~sklearn.preprocessing.StandardScaler` method
                - 'normalize' : Uses the `~sklearn.preprocessing.Normalizer` method
                - 'no' : No preprocessing on `feat_arr`

        test_train_opt : {'sample_frac', 'boxes_n', 'box_sample_frac'} `str`
            Option for which kind of separation to use for the training/testing
            splitting. This variable is set to 'boxes_n' by default.

            Options:
                - 'sample_frac' : Selects a fraction of the total sample
                - 'boxes_n' : Uses a set of the simulation boxes for the `training` and `testing`

        box_idx : `str`
            Initial and final indices of the simulation boxes to
            use for the `training` datasets. And the index of the boxes used
            for `testing`.
            Example: 0_4_5 >>> This will use from 0th to 4th box for training, and
            the 5th box for testing.

        box_test : `int`, optional
            Index of the simulation box to use for the `training` and
            `testing` datasets. This index represents the simulation box
            from which both the `training` and `testing` datasets will be
            produced. It uses the `test_size` variable to determine
            the fraction of the sample used for the `testing` dataset.
            This variable is set to `0` by default.

        sample_frac : `float`, optional
            Fraction of the total dataset ot use. This variable is set
            to `0.01` by default.

        test_size : float, optional
            Percentage of the catalogue that represents the `test` size of
            the testing dataset. This variable must be between (0,1).
            This variable is set to `0.25` by default.

        n_feat_use : {'all', 'sub'}, `str`, optional
            Option for which features to use for the ML training dataset.
            This variable is set to `sub` by default.

            Options:
                - 'all' : It uses all of the galaxy-group features to train.
                - 'sub' : It only uses a subclass of galaxy-group features to train

        dens_calc : `bool`, optional
            If True, densities at different distances will be calculated.
            This variable is set to `False` by default.

        perf_opt : `bool`, optional
            If True, a `perfect` version of the catalogues is used.
            This variable is set to `False` by default.

        seed : {0 - 4294967295}, `int`, optional
            Random seed to be used for the analysis. This variable is set to
            `1` by default.

        kf_splits : `int`, optional 
            Total number of K-folds to perform. Must be larger than 2.
            This variable is set to `3` by default.

        hidden_layers : `int`, optional
            Number of `hidden` layers to use for the Neural Network.
            This variable is set to `3` by default.

        unit_layer : `int`, optional
            Number of `units` per layer. This is used by the neural network.
            This variable is set to `100` by default.

        score_method : {'perc', 'threshold', 'model_score', 'r2'} `str`, optional
            Type of scoring to use when determining how well an algorithm
            is performing.

            Options:
                - 'perc' : Use percentage and rank-ordering of the values
                - 'threshold' : Score based on diffs of `threshold` or less from true value.
                - 'model_score' : Out-of-the-box method from `sklearn` to determine success.
                - 'r2': R-squared statistic for error calcuation.

        sample_method : {'normal', binning', 'subsample', 'weights'}, `str`, optional
            Method for binning or subsample the array of the estimated 
            group mass. This variable set to `binning` by default.

            Options:
                - 'normal' : Applied no special sampling method.
                - 'binning' : It bins the estimated group mass
                - 'subsample' : It subsamples the group mass to be equally 
                    representative at all masses
                - 'weights' : Applies larger weights to more massive systems.

        bin_val : {'fixed', 'nbins'} `str`, optional
            If ``sample_method == 'binning'``, `bin_val` determines the
            type of binning to use. This variable is set to 'fixed' by
            default.

            Options:
                - 'fixed' : Uses a fixed set of bins, evenly spaced by 0.4 dex.
                - 'nbins' : Splits the data into `2` bins, for low- and high-mass systems.

        ml_analysis : {'hod_fixed', 'dv_fixed', 'hod_dv_fixed'}, `str`, optional
            Type of analysis to perform. This variable is set to 
            `hod_dv_fixed`.

            Options:
                - 'hod_fixed' : Keeps the `hod_n` fixes, but varies `dv`.
                - 'dv_fixed' : Keeps `dv` fixed, but varies `hod_n`.
                - 'hod_dv_fixed' : Keeps both `dv` and `hod_n` fixed.
        """
        super().__init__()
        # Assigning variables
        self.hod_n          = kwargs.get('hod_n', 0)
        self.halotype       = kwargs.get('halotype', 'so')
        self.clf_method     = kwargs.get('clf_method', 1)
        self.clf_seed       = kwargs.get('clf_seed', 1235)
        self.dv             = kwargs.get('dv', 1.0)
        self.sample         = kwargs.get('sample', '19')
        self.catl_type      = kwargs.get('catl_type', 'mr')
        self.cosmo_choice   = kwargs.get('cosmo_choice', 'LasDamas')
        self.nmin           = kwargs.get('nmin', 2)
        self.mass_factor    = kwargs.get('mass_factor', 10)
        self.n_predict      = kwargs.get('n_predict', 1)
        self.shuffle_opt    = kwargs.get('shuffle_opt', True)
        self.dropna_opt     = kwargs.get('dropna_opt', False)
        self.pre_opt        = kwargs.get('pre_opt', 'standard')
        self.test_train_opt = kwargs.get('test_train_opt', 'boxes_n')
        self.box_idx        = kwargs.get('box_idx', '0_4_5')
        self.box_test       = kwargs.get('box_test', 0)
        self.sample_frac    = kwargs.get('sample_frac', 0.1)
        self.test_size      = kwargs.get('test_size', 0.25)
        self.n_feat_use     = kwargs.get('n_feat_use', 'sub')
        self.dens_calc      = kwargs.get('dens_calc', False)
        self.perf_opt       = kwargs.get('perf_opt', False)
        self.seed           = kwargs.get('seed', 1)
        self.kf_splits      = kwargs.get('kdf_splits', 3)
        self.hidden_layers  = kwargs.get('hidden_layers', 1)
        self.unit_layer     = kwargs.get('unit_layer', 100)
        self.score_method   = kwargs.get('score_method', 'threshold')
        self.threshold      = kwargs.get('threshold', 0.1)
        self.perc_val       = kwargs.get('perc_val', 0.68)
        self.sample_method  = kwargs.get('sample_method', 'binning')
        self.bin_val        = kwargs.get('bin_val', 'fixed')
        self.ml_analysis    = kwargs.get('ml_analysis', 'hod_dv_fixed')
        self.resample_opt   = kwargs.get('resample_opt', 'under')
        self.hod_models_n   = kwargs.get('hod_models_n', '0_1_2_3_4_5_6_7_8')
        self.include_nn     = kwargs.get('include_nn', False)
        self.dv_models_n    = kwargs.get('dv_models_n', '0.9_0.925_0.95_0.975_1.025_1.05_1.10')
        self.chosen_ml_alg  = kwargs.get('chosen_ml_alg', 'xgboost')
        #
        # Extra variables
        self.sample_Mr      = 'Mr{0}'.format(self.sample)
        self.sample_s       = str(self.sample)
        self.proj_dict      = cwpaths.cookiecutter_paths(__file__)
        self.mass_bin_width = 0.4
        self.nbins          = 2
        # self.proj_dict      = cwpaths.cookiecutter_paths('./')

    ## Prefix path based on initial parameters - Mock catalogues
    def catl_prefix_path(self, catl_kind='mocks'):
        """
        Prefix path used for saving catalogues and other catalogue-related
        objects.

        Parameters
        -----------
        catl_kind : {'mocks', 'data'} `str`
            Option for which kind of catalogue to analyze

        Returns
        -----------
        catl_pre_path : `str`
            Prefix path used for saving catalogue-related objects.
        """
        # Catalogue prefix string
        catl_pre_path = os.path.join('SDSS',
                                    catl_kind,
                                    'halos_{0}'.format(self.halotype),
                                    'dv_{0}'.format(self.dv),
                                    'hod_model_{0}'.format(self.hod_n),
                                    'clf_seed_{0}'.format(self.clf_seed),
                                    'clf_method_{0}'.format(self.clf_method),
                                    self.catl_type,
                                    self.sample_Mr,
                                    'dens_{0}'.format(self.dens_calc))

        return catl_pre_path

    ## Directory for the merged catalogues
    def catl_merged_dir(self, opt='catl', catl_kind='mocks', check_exist=True,
        create_dir=False):
        """
        Directory for the `merged` catalogues with the features for the
        ML analysis.

        Parameters
        ------------
        opt : {'catl', 'combined'} `str`, optional
            Option for returning which directory to return.

            Options:
                - 'catl' : Directory of the individuals merged catls
                - 'combined' : Directory of all the catalogues combined.

        catl_kind : {'mocks', 'data'} `str`
            Option for which kind of catalogue to analyze

        check_exist : `bool`, optional
            If `True`, it checks for whether or not the file exists.
            This variable is set to `True` by default.

        Returns
        ------------
        merged_feat_dir : `str`
            Path to the directory with the `merged` catalogues with features.
        """
        # Checking input variables
        if not (opt in ['catl', 'combined']):
            msg = '`opt` ({0}) is not a valid input variable!'.format(opt)
            raise ValueError(msg)
        # Catalogue prefix string
        catl_pre_path = self.catl_prefix_path(catl_kind=catl_kind)
        # `Merged` catalogues
        if (opt == 'catl'):
            merged_feat_dir = os.path.join( self.proj_dict['int_dir'],
                                            'merged_feat_catl',
                                            catl_pre_path,
                                            'merged_vac')
        # `All` catalogues merged
        if (opt == 'combined'):
            merged_feat_dir = os.path.join( self.proj_dict['int_dir'],
                                            'merged_feat_catl',
                                            catl_pre_path,
                                            'merged_vac_combined')
        # Creating directory if necessary
        if create_dir:
            cfutils.Path_Folder(merged_feat_dir)
        # Checking that folders exist
        if check_exist:
            if not (os.path.exists(merged_feat_dir)):
                msg = '`merged_feat_dir` ({0}) does not exist!'.format(
                    merged_feat_dir)
                raise FileNotFoundError(msg)

        return merged_feat_dir

    ## File prefix
    def _catl_pre_str(self):
        """
        String used as the prefix of files

        Returns
        ---------
        catl_pre_str : `str`
            String used for the prefix of a file.
        """
        # Prefix string
        catl_pre_arr = [self.sample_Mr,
                        self.halotype,
                        self.hod_n,
                        self.dv,
                        self.clf_method,
                        self.clf_seed,
                        self.catl_type,
                        self.cosmo_choice,
                        self.nmin,
                        self.mass_factor,
                        self.perf_opt]
        # Combining string
        catl_pre_str = '{0}_halo_{1}_hodn_{2}_dv_{3}_clfm_{4}_clfseed_{5}_'
        catl_pre_str += 'ctype_{6}_cosmo_{7}_nmin_{8}_massf_{9}_perf_{10}'
        catl_pre_str = catl_pre_str.format(*catl_pre_arr)

        return catl_pre_str

    ## Info for merged catalogue
    def extract_merged_catl_info(self, catl_kind='mocks', opt='catl', idx=0,
        ext='hdf5', return_path=False):
        """
        Extracts the information from a catalogue and returns it as
        pandas DataFrame

        Parameters
        ------------
        opt : {'catls', 'combined'} `str`, optional
            Option for returning which directory to return.
            This variable is set to `combined` by default.

            Options:
                - 'catls' : Directory of the individuals merged catls
                - 'all' : Directory of all the catalogues combined.

        catl_kind : {'mocks', 'data'} `str`
            Option for which kind of catalogue to analyze

        idx : `NoneType` or `int`, optional
            Index of the catalogue, from which to extract the information.
            This variable is set to `0` by default.

        ext : `str`, optional
            String for the extension of the file(s). This variable is set
            to `hdf5` by default.

        return_path : `bool`, optional
            If `True`, the function returns the path to the file being
            read. This variable is set to `False` by default

        Returns
        ------------
        merged_feat_pd : `pandas.DataFrame`
            DataFrame with the information from the `combined` file or
            `single` file.

        merged_feat_path : `str`, optional
            Path to the file being read. This object is only returned when
            ``return_path == True``.
        """
        # Check input parameters
        # `opt`
        if not (opt in ['catl', 'combined']):
            msg = '`opt` ({0}) is not a valid input variable!'.format(opt)
            raise ValueError(msg)
        # `idx`
        if not (isinstance(idx, (int)) or (idx is None)):
            msg = '`idx` ({0}) must be an integer or `None`!'.format(
                type(idx))
            raise TypeError(msg)
        if (opt == 'catl'):
            # Type of `idx`
            if not (isinstance(idx, (int))):
                msg = '`idx` ({0}) must be an integer!'.format(type(idx))
                raise TypeError(msg)
            # Value of `idx`
            if not (idx >= 0):
                msg = '`idx` ({0}) must be larger or equal to 0'.format(idx)
                raise ValueError(msg)
        # Find file
        merged_dir   = self.catl_merged_dir(opt=opt, catl_kind=catl_kind)
        catl_pre_str = self._catl_pre_str()
        # `catl` case
        if (opt == 'catl'):
            files_arr = cfutils.Index(merged_dir, ext)
            # Selecting index
            if not (idx <= (len(files_arr) - 1)):
                msg = '`idx` ({0}) is larger than the number of files '
                msg += 'in the directory ({1}).'
                msg = msg.format(idx, len(files_arr) - 1)
                raise ValueError(msg)
            # Reading in file and converting it to DataFrame
            merged_feat_path = files_arr[idx]
            # Making sure file exists
            if not (os.path.exists(merged_feat_path)):
                msg = '`merged_feat_path` ({0}) was not found!'.format(
                    merged_feat_path)
                raise FileNotFoundError(msg)
        # `combined` case
        if (opt == 'combined'):
            merged_feat_path = os.path.join(merged_dir,
                                            '{0}_merged_vac_all.{1}'.format(
                                                catl_pre_str, ext))
            # Checking file exists
            if not (os.path.exists(merged_feat_path)):
                msg = '`merged_feat_path` ({0}) was not found!'.format(
                    merged_feat_path)
                raise FileNotFoundError(msg)
        #
        # Reading in file
        merged_feat_pd = cfreaders.read_hdf5_file_to_pandas_DF(
            merged_feat_path)

        if return_path:
            return merged_feat_pd, merged_feat_path
        else:
            return merged_feat_pd

    ## -- Feature preprocessing --
    def _feat_proc_pre_str(self):
        """
        String used as the prefix of files during the `feature processing`
        step.

        Returns
        --------
        feat_proc_pre_str : `str`
            String used as the prefix of files during the `feature processing`
            step.
        """
        # `catl_pre_str`
        catl_pre_str = self._catl_pre_str()
        # File Prefix - Features
        feat_pre_str_arr = [catl_pre_str,
                            self.shuffle_opt,
                            self.n_predict,
                            self.pre_opt,
                            self.n_feat_use,
                            self.test_train_opt]
        feat_pre_str  = '{0}_sh_{1}_np_{2}_preopt_{3}_nf_{4}_'
        feat_pre_str += 'tt_{5}'
        feat_pre_str  = feat_pre_str.format(*feat_pre_str_arr)
        # `sample_frac`
        if (self.test_train_opt == 'sample_frac'):
            # String array
            file_str_arr = [feat_pre_str,
                            self.test_size,
                            self.sample_frac]
            # Main string
            feat_proc_pre_str = '{0}_tsize_{1}_sf_{2}'
            feat_proc_pre_str = feat_proc_pre_str.format(*file_str_arr)
        # `boxes_n`
        if (self.test_train_opt == 'boxes_n'):
            # String array
            file_str_arr = [feat_pre_str,
                            self.box_idx]
            # Main string
            feat_proc_pre_str = '{0}_bidx_{1}'.format(*file_str_arr)
        # `box_sample_frac`
        if (self.test_train_opt == 'box_sample_frac'):
            # String array
            file_str_arr = [feat_pre_str,
                            self.box_test]
            # Main string
            feat_proc_pre_str = '{0}_btest_{1}'.format(*file_str_arr)

        return feat_proc_pre_str

    def catl_feat_dir(self, catl_kind='mocks', check_exist=True,
        create_dir=False):
        """
        Directory for the `features` dictionaries for the ML analysic.

        Parameters
        -----------
        catl_kind : {'mocks', 'data'} `str`
            Option for which kind of catalogue to analyze

        check_exist : `bool`, optional
            If `True`, it checks for whether or not the file exists.
            This variable is set to `True` by default.

        create_dir : `bool`, optional
            If `True`, it creates the directory if it does not exist.
            This variable is set to `False` by default.

        Returns
        --------
        catl_feat_dir : `str`
            Path to the directory with the `features` for the `combined`
            versions of the catalogues.
        """
        # Check input parameters
        # `check_exist`
        if not (isinstance(check_exist, bool)):
            msg = '`check_exist` ({0}) must be of `boolean` type!'.format(
                type(check_exist))
            raise TypeError(msg)
        #
        # `create_dir`
        if not (isinstance(create_dir, bool)):
            msg = '`create_dir` ({0}) must be of `boolean` type!'.format(
                type(create_dir))
            raise TypeError(msg)
        # Catalogue prefix string
        catl_pre_path = self.catl_prefix_path(catl_kind=catl_kind)
        # Feature catalogue
        catl_feat_dir = os.path.join(   self.proj_dict['int_dir'],
                                        'catl_features',
                                        catl_pre_path,
                                        'feat_processing')
        # Creating directory if necessary
        if create_dir:
            cfutils.Path_Folder(catl_feat_dir)
        # Check for its existence
        if check_exist:
            if not (os.path.exists(catl_feat_dir)):
                msg = '`catl_feat_dir` ({0}) was not found!'.format(
                    catl_feat_dir)
                raise FileNotFoundError(msg)

        return catl_feat_dir

    def catl_feat_file(self, catl_kind='mocks', ext='p', check_exist=True):
        """
        Path to the file that contains the `training` and `testing`
        dictionaries for the `combined` catalogue.

        Parameters
        -----------
        catl_kind : {'mocks', 'data'} `str`
            Option for which kind of catalogue to analyze. This variable is
            set to 'mocks' by default'.

        ext : `str`, optional
            Extension of the file being analyzed. This variable is set to
            `p` by default.

        check_exist : `bool`, optional
            If `True`, it checks for whether or not the file exists.
            This variable is set to `True` by default.

        Returns
        ---------
        catl_feat_filepath : `str`
            Path to the file with the `training` and `testing` dictionaries
            for the `combined` version of the catalogues.
        """
        # Feature directory
        catl_feat_dir = self.catl_feat_dir(catl_kind=catl_kind)
        # Feat Pre-processing string
        feat_proc_pre_str = self._feat_proc_pre_str()
        # `catl_feat_filepath`
        catl_feat_filepath = os.path.join(catl_feat_dir,
            '{0}_feat_pro_out.{1}'.format(feat_proc_pre_str, ext))
        # Checking if file exists
        if check_exist:
            if not (os.path.exists(catl_feat_filepath)):
                msg = '`catl_feat_filepath` ({0}) was not found!'.format(
                    catl_feat_filepath)
                raise FileNotFoundError(msg)

        return catl_feat_filepath

    def extract_feat_file_info(self, catl_kind='mocks', ext='p',
        return_path=False):
        """
        Extracts the information from the `features` catalogues, and returns
        a set of dictionaries.

        Parameters
        -----------
        catl_kind : {'mocks', 'data'} `str`
            Option for which kind of catalogue to analyze.

        ext : `str`, optional
            Extension of the file being analyzed. This variable is set to
            `p` by default.

        return_path : `bool`, optional
            If True, the function also returns the path to the file being read.

        Returns
        ---------
        feat_dict : `dict`
            Dictionary with the `features` arrays for the given parameters.
            This variable is only returned if ``catl_kind == 'data'``.

        train_dict : `dict`
            Dictionary with the `training` arrays for the given parameters
            This variable is only returned if ``catl_kind == 'mocks'``.

        test_dict : `dict`
            Dictionary with the `training` arrays for the given parameters.
            This variable is only returned if ``catl_kind == 'mocks'``.

        catl_feat_filepath : `str`, optional
            Path to the dictionaries being read. This is only returned when
            ``return_path == True``.

        Notes
        ---------
        The objects returned by these functions depend on the choice of
        `catl_kind`.
        If:
            - ``catl_kind == 'data'``: `feat_dict`, `catl_feat_filepath` are
                returned.
            - ``catl_kind == 'data'``: `train_dict`, `test_dict`, and 
                `catl_feat_filepath` are returned.
        """
        # File containing the dictionaries
        catl_feat_filepath = self.catl_feat_file(ext=ext, catl_kind=catl_kind,
            check_exist=True)
        # Extracting information
        with open(catl_feat_filepath, 'rb') as feat_f:
            train_test_dict = pickle.load(feat_f)
        # Check for number of elements in the pickle file
        if (catl_kind == 'data'):
            if not (len(train_test_dict) == 1):
                msg = '`train_test_dict` must have 1 elements, but it has {1}!'
                msg = msg.format(len(train_test_dict))
                raise ValueError(msg)
            else:
                feat_dict = train_test_dict[0]
        elif (catl_kind == 'mocks'):
            if not (len(train_test_dict) == 2):
                msg = '`train_test_dict` must have 1 elements, but it has {1}!'
                msg = msg.format(len(train_test_dict))
                raise ValueError(msg)
            else:
                (train_dict, test_dict) = train_test_dict
        # Returning items
        if (catl_kind == 'data'):
            if return_path:
                return feat_dict, catl_feat_filepath
            else:
                return feat_dict
        elif (catl_kind == 'mocks'):
            if return_path:
                return train_dict, test_dict, catl_feat_filepath
            else:
                return train_dict, test_dict

    def _predicted_cols(self):
        """
        Determines the list of `predicted` columsn to use for the ML analysis.

        Returns
        ---------
        pred_cols : `list`
            List of columns used for the `prediction` in the ML analysis.
        """
        # Choosing columns
        if (self.n_predict == 1):
            pred_cols = ['M_h']
        elif (self.n_predict == 2):
            pred_cols = ['M_h', 'galtype']
        # Converting to numpy array

        return pred_cols

    def _feature_cols(self):
        """
        Determines the list of features used for the ML analysis

        Returns
        --------
        features_cols : `list`
            List of feature names
        """
        # Choosing names for the different features
        if (self.n_feat_use == 'all'):
            # Opening up a catalogue
            catl_pd = self.extract_merged_catl_info()
            # List of columns in `catl_pd`
            catl_cols = catl_pd.columns.values
            # List of predicted columns
            pred_cols = self._predicted_cols()
            # List of features used
            features_cols = [s for s in catl_cols if s not in pred_cols]
        # A subsample of columns
        if (self.n_feat_use == 'sub'):
            features_cols = [   'M_r',
                                'GG_mr_brightest',
                                'g_r',
                                'GG_rproj',
                                'GG_sigma_v',
                                'GG_M_r',
                                'GG_ngals',
                                'GG_M_group',
                                'GG_mdyn_rproj']

        return features_cols

    ## -- Traditional mass estimates extraction --
    def extract_trad_masses_alt(self, mass_opt='ham', score_method='perc',
        threshold=None, perc=None, return_score=False, return_frac_diff=False,
        nlim_min=5, nlim_threshold=False):
        """
        Extracts the `training` and `testing` datasets for the
        traditional methods of estimating masses.

        Parameters
        -----------
        mass_opt : {'ham', 'dyn'} `str`, optional
            Option for which kind of `estimated` mass to render the
            `training` and `testing` dictionary.

        score_method : {'perc', 'threshold', 'r2'} `str`, optional
            Type of scoring to use when determining how well an algorithm
            is performing. This variable is set to `perc` by default.

            Options:
                - 'perc' : Use percentage and rank-ordering of the values
                - 'threshold' : Score based on diffs of `threshold` or less from true value.
                - 'r2': R-squared statistic for error calcuation.

        threshold : `float`, optional
            Value to use when calculating error within some `threshold` value
            from the truth. This variable is set to `0.1` by default.

        perc : `float`, optional
            Value used when determining score within some `perc`
            percentile. The range for `perc` is [0, 1]. This variable
            is set to `0.9` by default.

        return_score : `bool`, optional
            If True, the function returns a `score` for the given
            `predicted` array. This variable is set to `False` by default.

        return_frac_diff : `bool`, optional
            If True, the function returns an array of the fractional
            differences between the `predicted` array and the `true`
            array. This variable is set to `False` by default.

        nlim_min : `int`, optional
            Minimum number of elements in a group, to show as part of the
            catalogue. This variable is set to `5` by default.

        nlim_threshold: `bool`, optional
            If `True`, only groups with number of members larger than
            `nlim_min` are included as part of the catalogue.

        Returns
        ---------
        pred_mass_arr : `numpy.ndarray`
            Array with the estimated `mass_opt` mass.

        true_mhalo_arr : `numpy.ndarray`
            Array with the `true` halo mass.

        frac_diff_arr : `numpy.ndarray`
            Array of the fractional difference between `predicted` and
            `true` arrays. This array is only returned when
            ``return_score == True``.

        score : `numpy.ndarray`, optional
            Array with the overall score. This value is returned only
            when ``return_score == True``.
        """
        # Check for inputs
        # `perc`
        if (perc is None):
            perc = self.perc_val
        # `threshold`
        if (threshold is None):
            threshold = self.threshold
        #
        # Loading datafile
        catl_pd_tot = self.extract_merged_catl_info(opt='combined')
        ##
        ## Only selecting groups with `nlim_min` galaxies or larger
        if nlim_threshold:
            catl_pd_tot = catl_pd_tot.loc[(catl_pd_tot['GG_ngals'] >= nlim_min)]
        ##
        ## Temporarily fixing `GG_mdyn_rproj`
        catl_pd_tot.loc[:, 'GG_mdyn_rproj'] /= 0.96
        ##
        ## List of column names
        catl_cols = catl_pd_tot.columns.values
        ##
        ## Dropping NaN's
        catl_pd_tot.dropna(how='any', inplace=True)
        ##
        ## Fraction of elements
        catl_pd = catl_pd_tot.sample(   frac=self.sample_frac,
                                        random_state=self.seed)
        catl_pd_cols = catl_pd.columns.values
        #
        # Choosing which type array to compute
        if (mass_opt == 'ham') and ('GG_M_group' in catl_pd_cols):
            mass_idx = 'GG_M_group'
        if (mass_opt == 'dyn') and ('GG_mdyn_rproj' in catl_pd_cols):
            mass_idx = 'GG_mdyn_rproj'
        #
        # Predicted and expected arrays
        pred_mass_arr  = catl_pd[mass_idx].values
        true_mhalo_arr = catl_pd['M_h'].values
        #
        # Cleaning up `dynamical mass`
        if (mass_opt == 'dyn'):
            pred_mass_arr_clean = num.where(pred_mass_arr != 0.)[0]
            pred_mass_arr       = pred_mass_arr[pred_mass_arr_clean]
            true_mhalo_arr      = true_mhalo_arr[pred_mass_arr_clean]
        # Return object list
        return_obj_list = [pred_mass_arr, true_mhalo_arr]
        ##
        ## Fractional Difference
        if return_frac_diff:
            # Calculating fractional difference
            frac_diff_arr  = 100 * (pred_mass_arr - true_mhalo_arr)
            frac_diff_arr /= true_mhalo_arr
            # Appending to return_obj_list`
            return_obj_list.append(frac_diff_arr)
        ##
        ## Score
        if return_score:
            # Computing general score
            score = cmlu.scoring_methods(true_mhalo_arr,
                                        pred_arr=pred_mass_arr,
                                        score_method=score_method,
                                        threshold=threshold,
                                        perc=perc)
            # Appending to return_obj_list`
            return_obj_list.append(score)

        return return_obj_list
    
    def extract_trad_masses(self, mass_opt='ham', score_method='perc',
        threshold=None, perc=None, return_score=False, return_frac_diff=False):
        """
        Extracts the `training` and `testing` datasets for the
        traditional methods of estimating masses.

        Parameters
        -----------
        mass_opt : {'ham', 'dyn'} `str`, optional
            Option for which kind of `estimated` mass to render the
            `training` and `testing` dictionary.

        score_method : {'perc', 'threshold', 'r2'} `str`, optional
            Type of scoring to use when determining how well an algorithm
            is performing. This variable is set to `perc` by default.

            Options:
                - 'perc' : Use percentage and rank-ordering of the values
                - 'threshold' : Score based on diffs of `threshold` or less from true value.
                - 'r2': R-squared statistic for error calcuation.

        threshold : `float`, optional
            Value to use when calculating error within some `threshold` value
            from the truth. This variable is set to `0.1` by default.

        perc : `float`, optional
            Value used when determining score within some `perc`
            percentile. The range for `perc` is [0, 1]. This variable
            is set to `0.9` by default.

        return_score : `bool`, optional
            If True, the function returns a `score` for the given
            `predicted` array. This variable is set to `False` by default.

        return_frac_diff : `bool`, optional
            If True, the function returns an array of the fractional
            differences between the `predicted` array and the `true`
            array. This variable is set to `False` by default.

        Returns
        ---------
        pred_mass_arr : `numpy.ndarray`
            Array with the estimated `mass_opt` mass.

        true_mhalo_arr : `numpy.ndarray`
            Array with the `true` halo mass.

        frac_diff_arr : `numpy.ndarray`
            Array of the fractional difference between `predicted` and
            `true` arrays. This array is only returned when
            ``return_score == True``.

        score : `numpy.ndarray`, optional
            Array with the overall score. This value is returned only
            when ``return_score == True``.
        """
        # Check for inputs
        # `perc`
        if (perc is None):
            perc = self.perc_val
        # `threshold`
        if (threshold is None):
            threshold = self.threshold
        # List of features
        feat_cols = num.array(self._feature_cols())
        pred_cols = num.array(self._predicted_cols())
        # Extracting `training` and `testing` dictionaries from main catalogue
        train_dict, test_dict = self.extract_feat_file_info()
        # Dictionary with feature columsn for each option of `mass_opt`
        mass_opt_dict = {'ham': 'GG_M_group', 'dyn': 'GG_mdyn_rproj'}
        #
        # Array for the `estimated` mass for the given `mass_opt` option.
        if (len(feat_cols) == 1):
            if (mass_opt_dict[mass_opt] in feat_cols):
                # Array of estimated mass
                pred_mass_arr = train_dict['X_train_ns']
            else:
                msg = '`{0}` was not found in the list of features!'.format(
                    mass_opt[mass_opt])
                raise ValueError(msg)
        else:
            if (mass_opt_dict[mass_opt] in feat_cols):
                # Index of the mass string
                mass_idx = num.where(feat_cols == mass_opt_dict[mass_opt])[0]
                # Array of estimated mass
                pred_mass_arr = train_dict['X_train_ns'].T[mass_idx].flatten()
            else:
                msg = '`{0}` was not found in the list of features!'.format(
                    mass_opt[mass_opt])
                raise ValueError(msg)
        #
        # Array for the `true` mass, i.e. Halo mass
        if ((self.n_predict == 1) and ('M_h' in pred_cols)):
            # Array of `true` halo mass
            true_mhalo_arr = train_dict['Y_train_ns']
        elif ((self.n_predict > 1) and ('M_h' in pred_cols)):
            # `True` mass Index
            true_mhalo_idx = num.where(pred_cols == 'M_h')[0]
            # `True` mass array
            true_mhalo_arr = train_dict['Y_train_ns'].T[true_mhalo_idx].flatten()
        elif ('M_h' not in pred_cols):
            msg = '`M_h` (True mass) was not part of the predicted values! '
            msg += 'The predicted values were: ({1})'
            msg = msg.format(pred_cols)
            raise ValueError(msg)
        #
        # Cleaning up `dynamical mass`
        if (mass_opt == 'dyn'):
            pred_mass_arr_clean = num.where(pred_mass_arr != 0.)[0]
            pred_mass_arr       = pred_mass_arr[pred_mass_arr_clean]
            true_mhalo_arr      = true_mhalo_arr[pred_mass_arr_clean]
        # Return object list
        return_obj_list = [pred_mass_arr, true_mhalo_arr]
        ##
        ## Fractional Difference
        if return_frac_diff:
            # Calculating fractional difference
            frac_diff_arr  = 100 * (pred_mass_arr - true_mhalo_arr)
            frac_diff_arr /= true_mhalo_arr
            # Appending to return_obj_list`
            return_obj_list.append(frac_diff_arr)
        ##
        ## Score
        if return_score:
            # Computing general score
            score = cmlu.scoring_methods(true_mhalo_arr,
                                        pred_arr=pred_mass_arr,
                                        score_method=score_method,
                                        threshold=threshold,
                                        perc=perc)
            # Appending to return_obj_list`
            return_obj_list.append(score)

        return return_obj_list

    ## -- Training of algorithms --
    def main_catl_train_dir(self, catl_kind='mocks', check_exist=True,
        create_dir=False):
        """
        Directory for the main training of the ML algorithms. This directory
        is mainly for the training and testing of the algorithms.

        Parameters
        -----------
        catl_kind : {'mocks', 'data'} `str`
            Option for which kind of catalogue to analyze

        check_exist : `bool`, optional
            If `True`, it checks for whether or not the file exists.
            This variable is set to `True` by default.

        create_dir : `bool`, optional
            If `True`, it creates the directory if it does not exist.

        Returns
        --------
        main_catl_train_dir : `str`
            Output directory for the main ML analysis.
        """
        # Check input parameters
        # `check_exist`
        if not (isinstance(check_exist, bool)):
            msg = '`check_exist` ({0}) must be of `boolean` type!'.format(
                type(check_exist))
            raise TypeError(msg)
        #
        # `create_dir`
        if not (isinstance(create_dir, bool)):
            msg = '`create_dir` ({0}) must be of `boolean` type!'.format(
                type(create_dir))
            raise TypeError(msg)
        # Catalogue Prefix
        catl_prefix_path = self.catl_prefix_path(catl_kind=catl_kind)
        # Output directory
        main_catl_train_dir = os.path.join( self.proj_dict['int_dir'],
                                            'train_test_dir',
                                            self.ml_analysis,
                                            catl_prefix_path)
        # Creating directory if necessary
        if create_dir:
            cfutils.Path_Folder(main_catl_train_dir)
        # Check that folder exists
        if check_exist:
            if not (os.path.exists(main_catl_train_dir)):
                msg = '`main_catl_train_dir` ({0}) was not found! '
                msg += 'Check your path!'
                msg = msg.format(main_catl_train_dir)
                raise FileNotFoundError(msg)

        return main_catl_train_dir

    def catl_train_alg_comp_dir(self, check_exist=True,
        create_dir=False):
        """
        Directory for the `algorithm comparison` section of the ML analysis.

        Parameters
        -----------
        check_exist : `bool`, optional
            If `True`, it checks for whether or not the file exists.
            This variable is set to `True` by default.

        create_dir : `bool`, optional
            If `True`, it creates the directory if it does not exist.

        Returns
        --------
        catl_train_alg_comp_dir : `str`
            Output directory for the `algorithm comparison` ML analysis.
        """
        # Check input parameters
        # `check_exist`
        if not (isinstance(check_exist, bool)):
            msg = '`check_exist` ({0}) must be of `boolean` type!'.format(
                type(check_exist))
            raise TypeError(msg)
        #
        # `create_dir`
        if not (isinstance(create_dir, bool)):
            msg = '`create_dir` ({0}) must be of `boolean` type!'.format(
                type(create_dir))
            raise TypeError(msg)
        #
        # Output directory
        main_catl_train_dir = self.main_catl_train_dir(check_exist=False,
            create_dir=False)
        # Appending to main directory
        catl_train_alg_comp_dir = os.path.join(main_catl_train_dir,
                                    'ml_alg_comparison')
        # Creating directory if necessary
        if create_dir:
            cfutils.Path_Folder(catl_train_alg_comp_dir)
        # Check that folder exists
        if check_exist:
            if not (os.path.exists(catl_train_alg_comp_dir)):
                msg = '`catl_train_alg_comp_dir` ({0}) was not found! '
                msg += 'Check your path!'
                msg = msg.format(catl_train_alg_comp_dir)
                raise FileNotFoundError(msg)

        return catl_train_alg_comp_dir

    def _catl_train_prefix_str(self):
        """
        String used as the prefix of files for the ML analysis.

        Returns
        --------
        catl_train_str : `str`
            String used as the prefix of files during the `data_analysis`
            step.
        """
        # Feature string - Prefix
        feat_proc_pre_str = self._feat_proc_pre_str()
        # Score Method
        # `perc`
        if (self.score_method == 'perc'):
            score_str = 'p_{0}'.format(self.perc_val)
        # `threshold`
        if (self.score_method == 'threshold'):
            score_str = 't_{0}'.format(self.threshold)
        # `model_score`
        if (self.score_method == 'model_score'):
            score_str = 'ms'
        # `r2`
        if (self.score_method == 'r2'):
            score_str = 'r2'
        #
        # Sample Method
        #
        if (self.sample_method in ['subsample', 'weights', 'normal']):
            sm_str = '{0}'.format(self.sample_method)
        elif (self.sample_method == 'binning'):
            sm_str = 'sm_{0}_{1}'.format(self.sample_method, self.bin_val)
        elif (self.sample_method == 'subsample'):
            sm_str = 'sm_{0}_{1}'.format(self.sample_method, self.resample_opt)
        #
        # File Prefix - ML Analysis
        catl_train_str_arr = [  feat_proc_pre_str,
                                self.hidden_layers,
                                self.unit_layer,
                                self.score_method,
                                score_str,
                                sm_str]
        catl_train_str = '{0}_{1}_{2}_{3}_{4}_{5}'
        catl_train_str = catl_train_str.format(*catl_train_str_arr)

        return catl_train_str

    ## -- Training algorithms - Algorithm comparison --
    def catl_train_alg_comp_file(self, ext='p', check_exist=True):
        """
        Path to the file that contains the outputs from the
        `algorithm comparison` stage.

        Parameters
        -----------
        ext : `str`, optional
            Extension of the file being analyzed. This variable is set to
            `p` by default.

        check_exist : `bool`, optional
            If `True`, it checks for whether or not the file exists.
            This variable is set to `True` by default.

        Returns
        ---------
        catl_alg_comp_path : `str`
            Path to the file with the outputs from the `algorithm comparison`
            stage of the ML analysis.
        """
        # `Algorithm comparison` directory
        catl_train_alg_comp_dir = self.catl_train_alg_comp_dir(
                                    check_exist=True,
                                    create_dir=False)
        # `Alg. Compr` Prefix string
        filename_str = '{0}_md.{1}'.format(self._catl_train_prefix_str(), ext)
        # `catl_alg_comp_path`
        catl_alg_comp_path = os.path.join(catl_train_alg_comp_dir,
                                filename_str)
        # Checking if file exists
        if check_exist:
            if not (os.path.exists(catl_alg_comp_path)):
                msg = '`catl_alg_comp_path` ({0}) was not found!'.format(
                    catl_alg_comp_path)
                raise FileNotFoundError(msg)

        return catl_alg_comp_path

    def extract_catl_alg_comp_info(self, ext='p', return_path=False):
        """
        Extracts the information from the `algorithm comparison`, and
        returns a set of dictionaries.

        Parameters
        -----------
        ext : `str`, optional
            Extension of the file being analyzed. This variable is set to
            `p` by default.

        return_path : `bool`, optional
            If True, the function also returns the path to the file being read.

        Returns
        ---------
        models_dict : `dict`
            Dictionary with the output results from the `algorithm comparison`
            stage of the ML analysis.
        """
        # File containing the dictionaries
        catl_alg_comp_path = self.catl_train_alg_comp_file(ext=ext,
                                check_exist=True)
        # Extracting information
        with open(catl_alg_comp_path, 'rb') as file_p:
            obj_arr = pickle.load(file_p)
        # Unpacking objects
        if (len(obj_arr) == 1):
            models_dict = obj_arr[0]
        else:
            msg = '`obj` ({0}) must be of length `1`'.format(len(obj_arr))

        if return_path:
            return models_dict, catl_alg_comp_path
        else:
            return models_dict

    ## -- Alternate names for different features
    def feat_cols_names_dict(self, return_all=False):
        """
        Substitutes for the column names in the list of `features`.

        Parameters
        ------------
        return_all : `bool`, optional
            If `True`, it returns the entire dictionary of galaxy properties.
            This variable is set to `False` by default.

        Returns
        ---------
        feat_cols_dict : `dict`
            Dictionary with column names for each of the `features` in 
            `feat_cols`.
        """
        # Column names
        feat_cols_names = { 'GG_r_tot':"Total Radius (G)",
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
                            'sersic': "Galaxy's morphology",
                            'M_h':"Galaxy's Halo mass"}
        #
        # Cross matching with the list of `features` for the project
        feat_cols = self._feature_cols()
        #
        # Intersection between column names
        feat_cols_intersect = num.intersect1d(  list(feat_cols_names.keys()),
                                                feat_cols)
        #
        # Creating new dictionary
        feat_cols_dict = {key: feat_cols_names[key] for key in
                            feat_cols_intersect}

        if return_all:
            return feat_cols_names
        else:
            return feat_cols_dict

    def catl_alg_comp_fig_str(self):
        """
        Prefix string for the figure of `algorithm comparison` ML stage.

        Returns
        ---------
        catl_alg_comp_fig_str : `str`
            Prefix string for the figure of `algorithm comparison`.
        """
        # ML Training prefix
        catl_train_prefix_str = self._catl_train_prefix_str()
        # Adding to main string
        catl_train_prefix_str += '_alg_comp'

        return catl_train_prefix_str

    ## -- Training algorithms - HOD Comparison
    def catl_train_hod_diff_dir(self, check_exist=True,
        create_dir=False):
        """
        Directory for the `HOD comparison` section of the ML analysis.

        Parameters
        -----------
        check_exist : `bool`, optional
            If `True`, it checks for whether or not the file exists.
            This variable is set to `True` by default.

        create_dir : `bool`, optional
            If `True`, it creates the directory if it does not exist.

        Returns
        --------
        catl_train_hod_diff_dir : `str`
            Output directory for the `HOD comparison` ML analysis.
        """
        # Check input parameters
        # `check_exist`
        if not (isinstance(check_exist, bool)):
            msg = '`check_exist` ({0}) must be of `boolean` type!'.format(
                type(check_exist))
            raise TypeError(msg)
        #
        # `create_dir`
        if not (isinstance(create_dir, bool)):
            msg = '`create_dir` ({0}) must be of `boolean` type!'.format(
                type(create_dir))
            raise TypeError(msg)
        #
        # Output directory
        main_catl_train_dir = self.main_catl_train_dir(check_exist=False,
            create_dir=False)
        # Appending to main directory
        catl_train_alg_comp_dir = os.path.join(main_catl_train_dir,
                                    'ml_hod_diff',
                                    self.hod_models_n)
        # Creating directory if necessary
        if create_dir:
            cfutils.Path_Folder(catl_train_alg_comp_dir)
        # Check that folder exists
        if check_exist:
            if not (os.path.exists(catl_train_alg_comp_dir)):
                msg = '`catl_train_alg_comp_dir` ({0}) was not found! '
                msg += 'Check your path!'
                msg = msg.format(catl_train_alg_comp_dir)
                raise FileNotFoundError(msg)

        return catl_train_alg_comp_dir

    def catl_train_hod_diff_file(self, ext='p', check_exist=True):
        """
        Path to the file that contains the outputs from the
        `HOD comparison` stage.

        Parameters
        -----------
        ext : `str`, optional
            Extension of the file being analyzed. This variable is set to
            `p` by default.

        check_exist : `bool`, optional
            If `True`, it checks for whether or not the file exists.
            This variable is set to `True` by default.

        Returns
        ---------
        catl_train_hod_diff_path : `str`
            Path to the file with the outputs from the `HOD comparison`
            stage of the ML analysis.
        """
        # `Algorithm comparison` directory
        catl_train_hod_diff_dir = self.catl_train_hod_diff_dir(
                                    check_exist=True,
                                    create_dir=False)
        # `HOD Comparison` Prefix string
        filename_str = '{0}_nn_{1}_md.{2}'.format(
                            self._catl_train_prefix_str(),
                            self.include_nn,
                            ext)
        # `catl_train_hod_diff_path`
        catl_train_hod_diff_path = os.path.join(catl_train_hod_diff_dir,
                                filename_str)
        # Checking if file exists
        if check_exist:
            if not (os.path.exists(catl_train_hod_diff_path)):
                msg = '`catl_train_hod_diff_path` ({0}) was not found!'.format(
                    catl_train_hod_diff_path)
                raise FileNotFoundError(msg)

        return catl_train_hod_diff_path

    def extract_catl_hod_diff_info(self, ext='p', return_path=False):
        """
        Extracts the information from the `algorithm comparison`, and
        returns a set of dictionaries.

        Parameters
        -----------
        ext : `str`, optional
            Extension of the file being analyzed. This variable is set to
            `p` by default.

        return_path : `bool`, optional
            If True, the function also returns the path to the file being read.

        Returns
        ---------
        models_dict : `dict`
            Dictionary with the output results from the `algorithm comparison`
            stage of the ML analysis.
        """
        # File containing the dictionaries
        catl_hod_diff_path = self.catl_train_hod_diff_file(ext=ext,
                                check_exist=True)
        # Extracting information
        with open(catl_hod_diff_path, 'rb') as file_p:
            obj_arr = pickle.load(file_p)
        # Unpacking objects
        if (len(obj_arr) == 1):
            models_dict = obj_arr[0]
        else:
            msg = '`obj` ({0}) must be of length `1`'.format(len(obj_arr))

        if return_path:
            return models_dict, catl_hod_diff_path
        else:
            return models_dict

    def catl_hod_diff_fig_str(self):
        """
        Prefix string for the figure of `algorithm comparison` ML stage.

        Returns
        ---------
        catl_alg_comp_fig_str : `str`
            Prefix string for the figure of `algorithm comparison`.
        """
        # ML Training prefix
        catl_train_prefix_str = self._catl_train_prefix_str()
        # Adding to main string
        catl_train_prefix_str += '_hod_diff'

        return catl_train_prefix_str

    ## -- Training algorithms - DV Comparison
    def catl_train_dv_diff_dir(self, check_exist=True,
        create_dir=False):
        """
        Directory for the `HOD comparison` section of the ML analysis.

        Parameters
        -----------
        check_exist : `bool`, optional
            If `True`, it checks for whether or not the file exists.
            This variable is set to `True` by default.

        create_dir : `bool`, optional
            If `True`, it creates the directory if it does not exist.

        Returns
        --------
        catl_train_hod_diff_dir : `str`
            Output directory for the `HOD comparison` ML analysis.
        """
        # Check input parameters
        # `check_exist`
        if not (isinstance(check_exist, bool)):
            msg = '`check_exist` ({0}) must be of `boolean` type!'.format(
                type(check_exist))
            raise TypeError(msg)
        #
        # `create_dir`
        if not (isinstance(create_dir, bool)):
            msg = '`create_dir` ({0}) must be of `boolean` type!'.format(
                type(create_dir))
            raise TypeError(msg)
        #
        # Output directory
        main_catl_train_dir = self.main_catl_train_dir(check_exist=False,
            create_dir=False)
        # Appending to main directory
        catl_train_dv_diff_comp_dir = os.path.join(main_catl_train_dir,
                                    'ml_dv_diff',
                                    self.dv_models_n)
        # Creating directory if necessary
        if create_dir:
            cfutils.Path_Folder(catl_train_dv_diff_comp_dir)
        # Check that folder exists
        if check_exist:
            if not (os.path.exists(catl_train_dv_diff_comp_dir)):
                msg = '`catl_train_dv_diff_comp_dir` ({0}) was not found! '
                msg += 'Check your path!'
                msg = msg.format(catl_train_dv_diff_comp_dir)
                raise FileNotFoundError(msg)

        return catl_train_dv_diff_comp_dir

    def catl_train_dv_diff_file(self, ext='p', check_exist=True):
        """
        Path to the file that contains the outputs from the
        `HOD comparison` stage.

        Parameters
        -----------
        ext : `str`, optional
            Extension of the file being analyzed. This variable is set to
            `p` by default.

        check_exist : `bool`, optional
            If `True`, it checks for whether or not the file exists.
            This variable is set to `True` by default.

        Returns
        ---------
        catl_train_hod_diff_path : `str`
            Path to the file with the outputs from the `HOD comparison`
            stage of the ML analysis.
        """
        # `Algorithm comparison` directory
        catl_train_dv_diff_dir = self.catl_train_dv_diff_dir(
                                    check_exist=True,
                                    create_dir=False)
        # `HOD Comparison` Prefix string
        filename_str = '{0}_nn_{1}_md.{2}'.format(
                            self._catl_train_prefix_str(),
                            self.include_nn,
                            ext)
        # `catl_train_hod_diff_path`
        catl_train_dv_diff_path = os.path.join(catl_train_dv_diff_dir,
                                filename_str)
        # Checking if file exists
        if check_exist:
            if not (os.path.exists(catl_train_dv_diff_path)):
                msg = '`catl_train_dv_diff_path` ({0}) was not found!'.format(
                    catl_train_dv_diff_path)
                raise FileNotFoundError(msg)

        return catl_train_dv_diff_path

    def extract_catl_dv_diff_info(self, ext='p', return_path=False):
        """
        Extracts the information from the `DV Difference`, and
        returns a set of dictionaries.

        Parameters
        -----------
        ext : `str`, optional
            Extension of the file being analyzed. This variable is set to
            `p` by default.

        return_path : `bool`, optional
            If True, the function also returns the path to the file being read.

        Returns
        ---------
        models_dict : `dict`
            Dictionary with the output results from the `DV difference`
            stage of the ML analysis.
        """
        # File containing the dictionaries
        catl_dv_diff_path = self.catl_train_dv_diff_file(ext=ext,
                                check_exist=True)
        # Extracting information
        with open(catl_dv_diff_path, 'rb') as file_p:
            obj_arr = pickle.load(file_p)
        # Unpacking objects
        if (len(obj_arr) == 1):
            models_dict = obj_arr[0]
        else:
            msg = '`obj` ({0}) must be of length `1`'.format(len(obj_arr))

        if return_path:
            return models_dict, catl_dv_diff_path
        else:
            return models_dict

    def catl_dv_diff_fig_str(self):
        """
        Prefix string for the figure of `algorithm comparison` ML stage.

        Returns
        ---------
        catl_alg_comp_fig_str : `str`
            Prefix string for the figure of `algorithm comparison`.
        """
        # ML Training prefix
        catl_train_prefix_str = self._catl_train_prefix_str()
        # Adding to main string
        catl_train_prefix_str += '_dv_diff'

        return catl_train_prefix_str

    ## -- Model Application to Data - SDSS
    def catl_model_app_data_main(self, check_exist=True, create_dir=False):
        """
        Directory for the main training of the ML algorithms. This directory
        is mainly for the training and testing of the algorithms.

        Parameters
        -----------
        check_exist : `bool`, optional
            If `True`, it checks for whether or not the file exists.
            This variable is set to `True` by default.

        create_dir : `bool`, optional
            If `True`, it creates the directory if it does not exist.

        Returns
        --------
        main_catl_data_outdir : `str`
            Output directory for the main catalogue output
        """
        # Check input parameters
        # `check_exist`
        if not (isinstance(check_exist, bool)):
            msg = '`check_exist` ({0}) must be of `boolean` type!'.format(
                type(check_exist))
            raise TypeError(msg)
        #
        # `create_dir`
        if not (isinstance(create_dir, bool)):
            msg = '`create_dir` ({0}) must be of `boolean` type!'.format(
                type(create_dir))
            raise TypeError(msg)
        # Catalogue Prefix
        catl_prefix_path_data = self.catl_prefix_path(catl_kind='data')
        # Output directory
        main_catl_data_outdir = os.path.join(   self.proj_dict['proc_dir'],
                                                'catl_data',
                                                self.ml_analysis,
                                                catl_prefix_path_data)
        # Creating directory if necessary
        if create_dir:
            cfutils.Path_Folder(main_catl_data_outdir)
        # Check that folder exists
        if check_exist:
            if not (os.path.exists(main_catl_data_outdir)):
                msg = '`main_catl_data_outdir` ({0}) was not found! '
                msg += 'Check your path!'
                msg = msg.format(main_catl_data_outdir)
                raise FileNotFoundError(msg)

        return main_catl_data_outdir

    def catl_model_app_data_file_prefix_str(self):
        """
        Prefix of the output data catalogue of SDSS. This file serves as
        the prefix for the training

        Returns
        ---------
        catl_app_data_str : `str`
            Prefix string of the output file for SDSS data.
        """
        # ML Training prefix
        catl_app_data_str = self._catl_train_prefix_str()
        # Adding to main string
        catl_app_data_str += '_catl_data_out'

        return catl_app_data_str

    def catl_model_application_data_file(self, ext='hdf5', check_exist=True):
        """
        Path to the output file containing the necessary information
        about the predicted ML masses, etc.

        Parameters
        -----------
        ext : `str`, optional
            Extension of the file being analyzed. This variable is set to
            `hdf5` by default.

        check_exist : `bool`, optional
            If `True`, it checks for whether or not the file exists.
            This variable is set to `True` by default.

        Returns
        -----------
        catl_outfile_path : `str`
            Path to the output file for the SDSS catalogue
        """
        # Catl Filename prefix
        catl_app_data_str = self.catl_model_app_data_file_prefix_str()
        # Output directory
        outdir = self.catl_model_app_data_main( create_dir=True,
                                                check_exist=False)
        # Filename
        filename_str = '{0}.{1}'.format(catl_app_data_str, ext)
        # Output directory
        catl_outfile_path = os.path.join(outdir, filename_str)
        # Checking if file exists
        if check_exist:
            if not (os.path.exists(catl_outfile_path)):
                msg = '`catl_outfile_path` ({0}) was not found!'.format(
                    catl_outfile_path)
                raise FileNotFoundError(msg)

        return catl_outfile_path

    def catl_model_mass_predictions(self):
        """
        Extracts the necessary information from the desired ML model, and
        computes the `predicted` array given the correct input. It
        predicts the `predicted` values for the `real` data sample.

        Returns
        ---------
        pred_arr : `numpy.ndarray`
            Array of predicted elements 
        """
        # Extracting `feature` arrays from `real` data sample.
        feat_dict = self.extract_feat_file_info(catl_kind='data',
                                                return_path=False)
        # Selecting `non-scaled` features
        feat_arr    = feat_dict['pred_X']
        feat_ns_arr = feat_dict['pred_X_ns']
        catl_pd_tot = feat_dict['catl_pd_tot']
        # List of features used
        feat_colnames = num.array(self._feature_cols())
        # Extracting columns corresponding to `GG_M_group`
        mgroup_data_arr = feat_ns_arr.T[num.where(
                                feat_colnames == 'GG_M_group')[0]].flatten()
        mgroup_data_idx_arr = num.arange(len(mgroup_data_arr))
        # Selecting algorithms
        if (self.chosen_ml_alg == 'xgboost'):
            ml_alg_str = 'XGBoost'
        elif (self.chosen_ml_alg == 'rf'):
            ml_alg_str = 'random_forest'
        elif (self.chosen_ml_alg == 'nn'):
            ml_alg_str = 'neural_network'
        # Extracting model from file
        models_dict = self.extract_catl_alg_comp_info()[ml_alg_str]
        # Extracting bins of mass for the `training` set
        if (self.sample_method in ['subsample', 'normal']):
            # Trained-ML model from synthetic catalogues
            model_obj = models_dict['model_ii']
            # Reshaping feature if necessary
            feat_arr_rs = feat_arr.reshape(len(feat_arr), len(feat_arr.T))
            # Computing the predicted array(s)
            pred_arr  = model_obj.predict(feat_arr_rs)
        elif (self.sample_method in ['binning']):
            # List of models for each bin of mass
            models_arr      = models_dict['model_ii']
            # Bins of mass from the `training` dataset.
            train_mass_bins = models_dict['train_mass_bins']
            # Total number of bin indices
            train_mass_idx  = num.arange(1, len(train_mass_bins))
            # Figuring out to which mass bin each mass corresponds
            mass_digits_bins = num.digitize(mgroup_data_arr, train_mass_bins)
            # Computing `predicted` array
            pred_arr = num.zeros(len(mgroup_data_arr))
            # Looping over models for each mass bin
            for kk, model_kk in enumerate(models_arr):
                # Mask for given `kk`
                mask_kk      = mass_digits_bins == (kk + 1)
                # Predicted masses and their corresponding indices
                mass_kk_pred = mgroup_data_arr    [mask_kk]
                mass_kk_idx  = mgroup_data_idx_arr[mask_kk]
                # Reshaping arrays
                feat_arr_rs = feat_arr[mask_kk].reshape(len(mass_kk_pred),
                                                        len(feat_arr.T))
                # Computing `predicted` arrays
                pred_mass_kk = models_arr[kk].predict(feat_arr_rs)
                # Populating array
                pred_arr[mass_kk_idx] = pred_mass_kk
        ##
        ## Joining predicted array(s) to DataFrame
        pred_colnames  = num.array(self._predicted_cols())
        pred_colnames  = num.array([xx + '_pred' for xx in pred_colnames])
        pred_pd        = pd.DataFrame(pred_arr, columns=pred_colnames)
        catl_pd_merged = pd.merge(  catl_pd_tot,
                                    pred_pd,
                                    left_index=True,
                                    right_index=True)

        return catl_pd_merged





