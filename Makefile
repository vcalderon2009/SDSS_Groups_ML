.PHONY: clean lint test_environment environment update_environment
	remove_environment catl_props test_files 
	delete_mock_catls delete_data_catls delete_all_but_raw clean_data_dir

#################################################################################
# GLOBALS                                                                       #
#################################################################################

PROJECT_DIR := $(shell dirname $(realpath $(lastword $(MAKEFILE_LIST))))
BUCKET = [OPTIONAL] your-bucket-for-syncing-data (do not include 's3://')
PROFILE = default
PROJECT_NAME = SDSS_Groups_ML
PYTHON_INTERPRETER = python3
ENVIRONMENT_FILE = environment.yml
ENVIRONMENT_NAME = sdss_groups_ml

DATA_DIR           = $(PROJECT_DIR)/data
SRC_DIR            = $(PROJECT_DIR)/src
SRC_DATA_DIR       = $(SRC_DIR)/data
SRC_PREPROC_DIR    = $(SRC_DIR)/data_preprocessing
MOCKS_CATL_DIR     = $(DATA_DIR)/external/SDSS/mocks
DATA_CATL_DIR      = $(DATA_DIR)/external/SDSS/data

# INPUT VARIABLES
# -- General
CPU_FRAC       = 0.75
REMOVE_FILES   = "True"
HOD_N          = 0
HALOTYPE       = 'so'
CLF_METHOD     = 3
CLF_SEED       = 1235
DV             = 1.0
SAMPLE         = "19"
CATL_TYPE      = "mr"
COSMO          = "LasDamas"
NMIN           = 2
VERBOSE        = "False"
# -- Data Preprocessing -- #
MASS_FACTOR    = 10
REMOVE_GROUP   = "True"
N_PREDICT      = 1
SHUFFLE_OPT    = "True"
DROP_NA        = "True"
PRE_OPT        = "standard"
TEST_TRAIN_OPT = "boxes_n"
BOX_IDX        = "0_4_5"
SAMPLE_FRAC    = 0.01
TEST_SIZE      = 0.25
N_FEAT_USE     = "sub"
PERF_OPT       = "False"
SEED           = 1235
# -- Training
KF_SPLITS    = 3
# SHUFFLE_OPT  = "True"
# TEST_SIZE    = 0.25
# SAMPLE_FRAC  = 0.05
# DROP_NA      = "True"
# N_PREDICT    = 1
# PRE_OPT      = 'standard'
SCORE_METHOD = 'perc'
HIDDEN_LAYERS= 100
THRESHOLD    = 0.1
PERC_VAL     = 0.68

ifeq (,$(shell which conda))
HAS_CONDA=False
else
HAS_CONDA=True
endif

#################################################################################
# COMMANDS                                                                      #
#################################################################################

## Deletes all build, test, coverage, and Python artifacts
clean: clean-build clean-pyc clean-test

## Removes Python file artifacts
clean-pyc:
	find . -name '*.pyc' -exec rm -f {} +
	find . -name '*.pyo' -exec rm -f {} +
	find . -name '*~' -exec rm -f {} +
	find . -name '__pycache__' -exec rm -fr {} +

## Remove build artifacts
clean-build:
	rm -fr build/
	rm -fr dist/
	rm -fr .eggs/
	find . -name '*.egg-info' -exec rm -fr {} +
	find . -name '*.egg' -exec rm -f {} +

## Remove test and coverage artifacts
clean-test:
	rm -fr .tox/
	rm -f .coverage
	rm -fr htmlcov/
	rm -fr .pytest_cache

## Lint using flake8
lint:
	flake8 --exclude=lib/,bin/,docs/conf.py .

## Test python environment is setup correctly
test_environment:
	$(PYTHON_INTERPRETER) test_environment.py

## Set up python interpreter environment - Using environment.yml
environment:
ifeq (True,$(HAS_CONDA))
		@echo ">>> Detected conda, creating conda environment."
		# conda config --add channels conda-forge
		conda env create -f $(ENVIRONMENT_FILE)
endif

## Update python interpreter environment
update_environment:
ifeq (True,$(HAS_CONDA))
		@echo ">>> Detected conda, creating conda environment."
		conda env update -f $(ENVIRONMENT_FILE)
		pip install --upgrade cosmo-utils
endif

## Delete python interpreter environment
remove_environment:
ifeq (True,$(HAS_CONDA))
		@echo ">>> Detected conda, removing conda environment"
		conda env remove -n $(ENVIRONMENT_NAME)
endif

#################################################################################
# PROJECT RULES                                                                 #
#################################################################################

## Preprocesses the datasets and transforms them into user-friendly versions
data_preprocess: download_dataset
	@python $(SRC_PREPROC_DIR)/data_preprocessing_main.py \
	-hod_model_n $(HOD_N) -halotype $(HALOTYPE) -clf_method $(CLF_METHOD) \
	-dv $(DV) -clf_seed $(CLF_SEED) -sample $(SAMPLE) -abopt $(CATL_TYPE) \
	-cosmo $(COSMO) -nmin $(NMIN) -mass_factor $(MASS_FACTOR) \
	-remove_group $(REMOVE_GROUP) -n_predict $(N_PREDICT) \
	 -shuffle_opt $(SHUFFLE_OPT) -dropna_opt $(dropna_opt) \
	 -pre_opt $(PRE_OPT) -test_train_opt $(TEST_TRAIN_OPT) -box_idx $(BOX_IDX) \
	 -sample_frac $(SAMPLE_FRAC) -test_size $(TEST_SIZE) \
	 -n_feat_use $(N_FEAT_USE) -cpu $(CPU_FRAC) -remove $(REMOVE_FILES) \
	 -v $(VERBOSE) -perf $(PERF_OPT) -seed $(SEED)

## Create set of `merged` catalogues, i.e. galaxy + group information
catl_props:
	@python $(SRC_DATA_DIR)/mocks_ml_main/catl_properties_calculations_make.py \
	-cpu $(CPU_FRAC) -remove $(REMOVE_FILES) -halotype $(HALOTYPE) \
	-clf_method $(CLF_METHOD) -hod_model_n $(HOD_N) -sample $(SAMPLE) \
	-nmin $(NMIN) -v $(VERBOSE) -clf_seed $(CLF_SEED)

## Plots the figures of the set of `merged` catalogues
catl_props_plots:
	@python $(SRC_DATA_DIR)/mocks_ml_main/catl_properties_plots.py -cpu $(CPU_FRAC) \
	-remove $(REMOVE_FILES) -halotype $(HALOTYPE) -clf_method $(CLF_METHOD) \
	-hod_model_n $(HOD_N) -sample $(SAMPLE) -nmin $(NMIN) -v $(VERBOSE) \
	-clf_seed $(CLF_SEED)

## Trains ML algorithms on the `merged` dataset
ml_train:
	@python $(SRC_DATA_DIR)/mocks_ml_main/catl_ml_main_make.py -a 'training' \
	-cpu $(CPU_FRAC) -remove $(REMOVE_FILES) -halotype $(HALOTYPE) \
	-clf_method $(CLF_METHOD) -hod_model_n $(HOD_N) -sample $(SAMPLE) \
	-nmin $(NMIN) -shuffle_opt $(SHUFFLE_OPT) -kf_splits $(KF_SPLITS) \
	-n_predict $(N_PREDICT) -test_size $(TEST_SIZE) -sample_frac $(SAMPLE_FRAC) \
	-dropna_opt $(DROP_NA) -v $(VERBOSE) -pre_opt $(PRE_OPT) \
	-score_method $(SCORE_METHOD) -hidden_layers $(HIDDEN_LAYERS) \
	-threshold $(THRESHOLD) -perc_val $(PERC_VAL) -clf_seed $(CLF_SEED)

## Plots the ML figures of the `trained` dataset
ml_plots:
	@python $(SRC_DATA_DIR)/mocks_ml_main/catl_ml_main_make.py -a 'plots' \
	-cpu $(CPU_FRAC) -remove $(REMOVE_FILES) -halotype $(HALOTYPE) \
	-clf_method $(CLF_METHOD) -hod_model_n $(HOD_N) -sample $(SAMPLE) \
	-nmin $(NMIN) -v $(VERBOSE) -pre_opt $(PRE_OPT) -sample_frac $(SAMPLE_FRAC)\
	-score_method $(SCORE_METHOD) -clf_seed $(CLF_SEED)

## Run tests to see if all files (Halobias, catalogues) are in order
test_files:
	@pytest

## Downloads Dataset
download_dataset:
	@python $(SRC_PREPROC_DIR)/download_dataset.py \
	-hod_model_n $(HOD_N) -halotype $(HALOTYPE) -clf_method $(CLF_METHOD) \
	-dv $(DV) -sample $(SAMPLE) -abopt $(CATL_TYPE) -clf_seed $(CLF_SEED) \
	-perf $(PERF_OPT) -v $(VERBOSE)

## Delete existing `mock` catalogues
delete_mock_catls:
	find $(MOCKS_CATL_DIR) -type f -name '*.hdf5' -delete

## Delete existing `data` catalogues
delete_data_catls:
	find $(DATA_CATL_DIR) -type f -name '*.hdf5' -delete
	find $(DATA_CATL_DIR) -type f -name '*.tex' -delete
	find $(DATA_CATL_DIR) -type f -name '*.csv' -delete

## Delete all catalogues `mocks` and `data`
delete_all_catls: delete_mock_catls delete_data_catls
	@echo ""

## Delete all files, except for `raw` files
delete_all_but_raw:
	@rm -rf $(DATA_DIR)/external/*
	@rm -rf $(DATA_DIR)/interim/*
	@rm -rf $(DATA_DIR)/processed/*

## Clean the `./data` folder and remove all of the files
clean_data_dir:
	@rm -rf $(DATA_DIR)/external/*
	@rm -rf $(DATA_DIR)/interim/*
	@rm -rf $(DATA_DIR)/processed/*
	@rm -rf $(DATA_DIR)/raw/*

## Delete screens from creating catalogues
delete_catl_screens:
	screen -S "SDSS_ML_Groups_Catls_Create" -X quit || echo ""
	screen -S "SDSS_ML_TRAINING" -X quit || echo ""
	# screen -S "SDSS_Data_Mocks_create" -X quit || echo ""

#################################################################################
# Self Documenting Commands                                                     #
#################################################################################

.DEFAULT_GOAL := show-help

# Inspired by <http://marmelab.com/blog/2016/02/29/auto-documented-makefile.html>
# sed script explained:
# /^##/:
# 	* save line in hold space
# 	* purge line
# 	* Loop:
# 		* append newline + line to hold space
# 		* go to next line
# 		* if line starts with doc comment, strip comment character off and loop
# 	* remove target prerequisites
# 	* append hold space (+ newline) to line
# 	* replace newline plus comments by `---`
# 	* print line
# Separate expressions are necessary because labels cannot be delimited by
# semicolon; see <http://stackoverflow.com/a/11799865/1968>
.PHONY: show-help
show-help:
	@echo "$$(tput bold)Available rules:$$(tput sgr0)"
	@echo
	@sed -n -e "/^## / { \
		h; \
		s/.*//; \
		:doc" \
		-e "H; \
		n; \
		s/^## //; \
		t doc" \
		-e "s/:.*//; \
		G; \
		s/\\n## /---/; \
		s/\\n/ /g; \
		p; \
	}" ${MAKEFILE_LIST} \
	| LC_ALL='C' sort --ignore-case \
	| awk -F '---' \
		-v ncol=$$(tput cols) \
		-v indent=19 \
		-v col_on="$$(tput setaf 6)" \
		-v col_off="$$(tput sgr0)" \
	'{ \
		printf "%s%*s%s ", col_on, -indent, $$1, col_off; \
		n = split($$2, words, " "); \
		line_length = ncol - indent; \
		for (i = 1; i <= n; i++) { \
			line_length -= length(words[i]) + 1; \
			if (line_length <= 0) { \
				line_length = ncol - indent - length(words[i]) - 1; \
				printf "\n%*s ", -indent, " "; \
			} \
			printf "%s ", words[i]; \
		} \
		printf "\n"; \
	}' \
	| more $(shell test $(shell uname) = Darwin && echo '--no-init --raw-control-chars')
