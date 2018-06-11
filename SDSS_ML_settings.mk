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