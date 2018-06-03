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

import numpy as num
import os
import pandas as pd
import pickle

# Functions

class ReadML(object):
    """
    Reads in the training and testing dictionaries for a given
    set of variables. This is mainly used for reading in the output
    of the `data preprocessing`.
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

        sample_frac : `float`
            Fraction of the total dataset ot use.

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
        self.sample_frac    = kwargs.get('sample_frac', 0.01)
        self.test_size      = kwargs.get('test_size', 0.25)
        self.n_feat_use     = kwargs.get('n_feat_use', 'sub')
        self.dens_calc      = kwargs.get('dens_calc', False)
        self.perf_opt       = kwargs.get('perf_opt', False)
        self.seed           = kwargs.get('seed', 1)
        # Extra variables
        self.sample_Mr      = 'Mr{0}'.format(self.sample)
        self.sample_s       = str(self.sample)
        self.proj_dict      = cwpaths.cookiecutter_paths('./')

    def catl_prefix_path(self):
        """
        Prefix path used for saving catalogues and other catalogue-related
        objects.

        Returns
        ---------
        catl_pre_path : `str`
            Prefix path used for saving catalogue-related objects.
        """
        # Catalogue prefix string
        catl_pre_path = os.path.join('SDSS',
                                    'mocks',
                                    'halos_{0}'.format(self.halotype),
                                    'dv_{0}'.format(self.dv),
                                    'hod_model_{0}'.format(self.hod_n),
                                    'clf_seed_{0}'.format(self.clf_seed),
                                    'clf_method_{0}'.format(self.clf_method),
                                    self.catl_type,
                                    self.sample_Mr,
                                    'dens_{0}'.format(self.dens_calc))

        return catl_pre_path

    def catl_merged_dir(self, opt='catl'):
        """
        Directory for the `merged` catalogues with the features for the
        ML analysis.

        Parameters
        ------------
        opt : {'catls', 'combined'} `str`, optional
            Option for returning which directory to return.

            Options:
                - 'catls' : Directory of the individuals merged catls
                - 'all' : Directory of all the catalogues combined.

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
        catl_pre_path = self.catl_prefix_path()
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
        # Checking that folders exist
        if not (os.path.exists(merged_feat_dir)):
            msg = '`merged_feat_dir` ({0}) does not exist!'.format(
                merged_feat_dir)
            raise FileNotFoundError(msg)

        return merged_feat_dir

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

    def extract_merged_catl_info(self, opt='catl', idx=0, ext='hdf5',
        return_path=False):
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
        merged_dir   = self.catl_merged_dir(opt=opt)
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
        feat_pre_str  = '{0}_sh_{1}_npredict_{2}_preopt_{3}_nfeat_{4}_'
        feat_pre_str += 'testtrain_{5}'
        feat_pre_str  = feat_pre_str.format(*feat_pre_str_arr)
        # `sample_frac`
        if (self.test_train_opt == 'sample_frac'):
            # String array
            file_str_arr = [feat_pre_str,
                            self.test_size,
                            self.sample_frac,
                            self.dens_calc]
            # Main string
            feat_proc_pre_str = '{0}_testsize_{1}_samplefrac_{2}_dens_{3}'
            feat_proc_pre_str = feat_proc_pre_str.format(*file_str_arr)
        # `boxes_n`
        if (self.test_train_opt == 'boxes_n'):
            # String array
            file_str_arr = [feat_pre_str,
                            self.box_idx,
                            self.dens_calc]
            # Main string
            feat_proc_pre_str = '{0}_boxidx_{1}_dens_{2}'.format(*file_str_arr)
        # `box_sample_frac`
        if (self.test_train_opt == 'box_sample_frac'):
            # String array
            file_str_arr = [feat_pre_str,
                            self.box_test,
                            self.dens_calc]
            # Main string
            feat_proc_pre_str = '{0}_boxtest_{1}_dens_{2}'.format(*file_str_arr)

        return feat_proc_pre_str

    def catl_feat_dir(self):
        """
        Directory for the `features` dictionaries for the ML analysic.

        Returns
        --------
        catl_feat_dir : `str`
            Path to the directory with the `features` for the `combined`
            versions of the catalogues.
        """
        # Catalogue prefix string
        catl_pre_path = self.catl_prefix_path()
        # Feature catalogue
        catl_feat_dir = os.path.join(   self.proj_dict['int_dir'],
                                        'catl_features',
                                        catl_pre_path,
                                        'feat_processing')
        # Check for its existence
        if not (os.path.exists(catl_feat_dir)):
            msg = '`catl_feat_dir` ({0}) was not found!'.format(catl_feat_dir)
            raise FileNotFoundError(msg)

        return catl_feat_dir

    def catl_feat_file(self, ext='p', check_exist=True):
        """
        Path to the file that contains the `training` and `testing`
        dictionaries for the `combined` catalogue.

        Parameters
        -----------
        ext : `str`, optional
            Extension of the file being analyzed. This variable is set to
            `p` by default.

        check_exist : `bool`, optional
            If `True`, it check for whether the file exists or not.
            This variable is set to `False`.

        Returns
        ---------
        catl_feat_filepath : `str`
            Path to the file with the `training` and `testing` dictionaries
            for the `combined` version of the catalogues.
        """
        # Feature directory
        catl_feat_dir = self.catl_feat_dir()
        # Feat Pre-processing string
        feat_proc_pre_str = self._feat_proc_pre_str()
        # `catl_feat_filepath`
        catl_feat_filepath = os.path.join(catl_feat_dir,
            '{0}_feature_processing_out.{1}'.format(feat_proc_pre_str, ext))
        # Checking if file exists
        if check_exist:
            if not (os.path.exists(catl_feat_filepath)):
                msg = '`catl_feat_filepath` ({0}) was not found!'.format(
                    catl_feat_filepath)
                raise FileNotFoundError(msg)

        return catl_feat_filepath

    def extract_feat_file_info(self, ext='p', return_path=False):
        """
        Extracts the information from the `features` catalogues, and returns
        a set of dictionaries.

        Parameters
        -----------
        ext : `str`, optional
            Extension of the file being analyzed. This variable is set to
            `p` by default.

        return_path : `bool`, optional
            If True, the function also returns the path to the file being read.

        check_exist : `bool`, optional
            If `True`, it check for whether the file exists or not.
            This variable is set to `False`.

        Returns
        ---------
        train_dict : `dict`
            Dictionary with the `training` arrays for the given parameters

        test_dict : `dict`
            Dictionary with the `training` arrays for the given parameters

        train_test_path : `str`, optional
            Path to the dictionaries being read. This is only returned when
            ``return_path == True``.
        """
        # File containing the dictionaries
        catl_feat_filepath = self.catl_feat_file(ext=ext,
            check_exist=True)
        # Extracting information
        with open(catl_feat_filepath, 'rb') as feat_f:
            train_test_dict = pickle.load(feat_f)
        # Check for number of elements in the pickle file
        if not (len(train_test_dict) == 2):
            msg = '`train_test_dict` must have 2 elements, but it has {1}!'
            msg = msg.format(len(train_test_dict))
            raise ValueError(msg)
        else:
            (train_dict, test_dict) = train_test_dict

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

    def extract_trad_masses(self, mass_opt='ham', score_method='perc',
        threshold=0.1, perc=0.9, return_score=False, return_frac_diff=False):
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
























