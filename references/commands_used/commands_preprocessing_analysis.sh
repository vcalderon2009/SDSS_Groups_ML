#!/usr/bin/env bash

##############################################################################
# ---- Author ----:
# - Victor Calderon (victor.calderon@vanderbilt.edu)
#
# ---- Description ----:
# List of commands used for data preprocesssing and analysis for this
# ML project.
#
# ---- Date Modified ----:
# 2019-01-14
#
##############################################################################

##### --------------------------- 2019-01-14 --------------------------- #####

##############################################################################
##################### -------- PREPROCESSING -------- ########################
##############################################################################

# --- 1 Box --- #
# Run all at the same time for 32 cores
make HALOTYPE="so" CLF_METHOD=1 CLF_SEED=1235 DENS_CALC="False" TEST_TRAIN_OPT="boxes_n" BOX_IDX="0_3_4" REMOVE_MASTER="True" REMOVE_FILES="True" CPU_FRAC=0.15 DV=0.9   data_preprocess
make HALOTYPE="so" CLF_METHOD=1 CLF_SEED=1235 DENS_CALC="False" TEST_TRAIN_OPT="boxes_n" BOX_IDX="0_3_4" REMOVE_MASTER="True" REMOVE_FILES="True" CPU_FRAC=0.15 DV=0.925 data_preprocess
make HALOTYPE="so" CLF_METHOD=1 CLF_SEED=1235 DENS_CALC="False" TEST_TRAIN_OPT="boxes_n" BOX_IDX="0_3_4" REMOVE_MASTER="True" REMOVE_FILES="True" CPU_FRAC=0.15 DV=0.95  data_preprocess
make HALOTYPE="so" CLF_METHOD=1 CLF_SEED=1235 DENS_CALC="False" TEST_TRAIN_OPT="boxes_n" BOX_IDX="0_3_4" REMOVE_MASTER="True" REMOVE_FILES="True" CPU_FRAC=0.15 DV=0.975 data_preprocess
make HALOTYPE="so" CLF_METHOD=1 CLF_SEED=1235 DENS_CALC="False" TEST_TRAIN_OPT="boxes_n" BOX_IDX="0_3_4" REMOVE_MASTER="True" REMOVE_FILES="True" CPU_FRAC=0.15 DV=1.0   data_preprocess
make HALOTYPE="so" CLF_METHOD=1 CLF_SEED=1235 DENS_CALC="False" TEST_TRAIN_OPT="boxes_n" BOX_IDX="0_3_4" REMOVE_MASTER="True" REMOVE_FILES="True" CPU_FRAC=0.15 DV=1.025 data_preprocess
make HALOTYPE="so" CLF_METHOD=1 CLF_SEED=1235 DENS_CALC="False" TEST_TRAIN_OPT="boxes_n" BOX_IDX="0_3_4" REMOVE_MASTER="True" REMOVE_FILES="True" CPU_FRAC=0.15 DV=1.05  data_preprocess
make HALOTYPE="so" CLF_METHOD=1 CLF_SEED=1235 DENS_CALC="False" TEST_TRAIN_OPT="boxes_n" BOX_IDX="0_3_4" REMOVE_MASTER="True" REMOVE_FILES="True" CPU_FRAC=0.10 DV=1.075 data_preprocess
make HALOTYPE="so" CLF_METHOD=1 CLF_SEED=1235 DENS_CALC="False" TEST_TRAIN_OPT="boxes_n" BOX_IDX="0_3_4" REMOVE_MASTER="True" REMOVE_FILES="True" CPU_FRAC=0.10 DV=1.10  data_preprocess

# --- Multiple boxes --- #
# Run all at the same time for 32 cores
make HALOTYPE="so" CLF_METHOD=1 CLF_SEED=1235 DENS_CALC="False" TEST_TRAIN_OPT="boxes_n" BOX_IDX="0_3_4" REMOVE_MASTER="True" REMOVE_FILES="True" CPU_FRAC=0.15 DV=1.0 HOD_N=1 data_preprocess
make HALOTYPE="so" CLF_METHOD=1 CLF_SEED=1235 DENS_CALC="False" TEST_TRAIN_OPT="boxes_n" BOX_IDX="0_3_4" REMOVE_MASTER="True" REMOVE_FILES="True" CPU_FRAC=0.15 DV=1.0 HOD_N=2 data_preprocess
make HALOTYPE="so" CLF_METHOD=1 CLF_SEED=1235 DENS_CALC="False" TEST_TRAIN_OPT="boxes_n" BOX_IDX="0_3_4" REMOVE_MASTER="True" REMOVE_FILES="True" CPU_FRAC=0.15 DV=1.0 HOD_N=3 data_preprocess
make HALOTYPE="so" CLF_METHOD=1 CLF_SEED=1235 DENS_CALC="False" TEST_TRAIN_OPT="boxes_n" BOX_IDX="0_3_4" REMOVE_MASTER="True" REMOVE_FILES="True" CPU_FRAC=0.15 DV=1.0 HOD_N=4 data_preprocess
make HALOTYPE="so" CLF_METHOD=1 CLF_SEED=1235 DENS_CALC="False" TEST_TRAIN_OPT="boxes_n" BOX_IDX="0_3_4" REMOVE_MASTER="True" REMOVE_FILES="True" CPU_FRAC=0.15 DV=1.0 HOD_N=5 data_preprocess
make HALOTYPE="so" CLF_METHOD=1 CLF_SEED=1235 DENS_CALC="False" TEST_TRAIN_OPT="boxes_n" BOX_IDX="0_3_4" REMOVE_MASTER="True" REMOVE_FILES="True" CPU_FRAC=0.15 DV=1.0 HOD_N=6 data_preprocess
make HALOTYPE="so" CLF_METHOD=1 CLF_SEED=1235 DENS_CALC="False" TEST_TRAIN_OPT="boxes_n" BOX_IDX="0_3_4" REMOVE_MASTER="True" REMOVE_FILES="True" CPU_FRAC=0.15 DV=1.0 HOD_N=7 data_preprocess
make HALOTYPE="so" CLF_METHOD=1 CLF_SEED=1235 DENS_CALC="False" TEST_TRAIN_OPT="boxes_n" BOX_IDX="0_3_4" REMOVE_MASTER="True" REMOVE_FILES="True" CPU_FRAC=0.15 DV=1.0 HOD_N=8 data_preprocess

# --- Different CLF Scatter Values --- #
# Run all at the same time for `32` cores
make HALOTYPE="so" CLF_METHOD=1 CLF_SEED=1235 DENS_CALC="False" TEST_TRAIN_OPT="boxes_n" BOX_IDX="0_3_4" REMOVE_MASTER="True" REMOVE_FILES="True" CPU_FRAC=0.15 DV=1.0 SIGMA_CLF_C=0.10 data_preprocess
make HALOTYPE="so" CLF_METHOD=1 CLF_SEED=1235 DENS_CALC="False" TEST_TRAIN_OPT="boxes_n" BOX_IDX="0_3_4" REMOVE_MASTER="True" REMOVE_FILES="True" CPU_FRAC=0.15 DV=1.0 SIGMA_CLF_C=0.12 data_preprocess
make HALOTYPE="so" CLF_METHOD=1 CLF_SEED=1235 DENS_CALC="False" TEST_TRAIN_OPT="boxes_n" BOX_IDX="0_3_4" REMOVE_MASTER="True" REMOVE_FILES="True" CPU_FRAC=0.15 DV=1.0 SIGMA_CLF_C=0.14 data_preprocess
make HALOTYPE="so" CLF_METHOD=1 CLF_SEED=1235 DENS_CALC="False" TEST_TRAIN_OPT="boxes_n" BOX_IDX="0_3_4" REMOVE_MASTER="True" REMOVE_FILES="True" CPU_FRAC=0.15 DV=1.0 SIGMA_CLF_C=0.15 data_preprocess
make HALOTYPE="so" CLF_METHOD=1 CLF_SEED=1235 DENS_CALC="False" TEST_TRAIN_OPT="boxes_n" BOX_IDX="0_3_4" REMOVE_MASTER="True" REMOVE_FILES="True" CPU_FRAC=0.15 DV=1.0 SIGMA_CLF_C=0.16 data_preprocess
make HALOTYPE="so" CLF_METHOD=1 CLF_SEED=1235 DENS_CALC="False" TEST_TRAIN_OPT="boxes_n" BOX_IDX="0_3_4" REMOVE_MASTER="True" REMOVE_FILES="True" CPU_FRAC=0.15 DV=1.0 SIGMA_CLF_C=0.18 data_preprocess
make HALOTYPE="so" CLF_METHOD=1 CLF_SEED=1235 DENS_CALC="False" TEST_TRAIN_OPT="boxes_n" BOX_IDX="0_3_4" REMOVE_MASTER="True" REMOVE_FILES="True" CPU_FRAC=0.15 DV=1.0 SIGMA_CLF_C=0.20 data_preprocess
make HALOTYPE="so" CLF_METHOD=1 CLF_SEED=1235 DENS_CALC="False" TEST_TRAIN_OPT="boxes_n" BOX_IDX="0_3_4" REMOVE_MASTER="True" REMOVE_FILES="True" CPU_FRAC=0.15 DV=1.0 SIGMA_CLF_C=0.22 data_preprocess
make HALOTYPE="so" CLF_METHOD=1 CLF_SEED=1235 DENS_CALC="False" TEST_TRAIN_OPT="boxes_n" BOX_IDX="0_3_4" REMOVE_MASTER="True" REMOVE_FILES="True" CPU_FRAC=0.10 DV=1.0 SIGMA_CLF_C=0.24 data_preprocess
make HALOTYPE="so" CLF_METHOD=1 CLF_SEED=1235 DENS_CALC="False" TEST_TRAIN_OPT="boxes_n" BOX_IDX="0_3_4" REMOVE_MASTER="True" REMOVE_FILES="True" CPU_FRAC=0.10 DV=1.0 SIGMA_CLF_C=0.25 data_preprocess
make HALOTYPE="so" CLF_METHOD=1 CLF_SEED=1235 DENS_CALC="False" TEST_TRAIN_OPT="boxes_n" BOX_IDX="0_3_4" REMOVE_MASTER="True" REMOVE_FILES="True" CPU_FRAC=0.10 DV=1.0 SIGMA_CLF_C=0.26 data_preprocess
make HALOTYPE="so" CLF_METHOD=1 CLF_SEED=1235 DENS_CALC="False" TEST_TRAIN_OPT="boxes_n" BOX_IDX="0_3_4" REMOVE_MASTER="True" REMOVE_FILES="True" CPU_FRAC=0.10 DV=1.0 SIGMA_CLF_C=0.28 data_preprocess
make HALOTYPE="so" CLF_METHOD=1 CLF_SEED=1235 DENS_CALC="False" TEST_TRAIN_OPT="boxes_n" BOX_IDX="0_3_4" REMOVE_MASTER="True" REMOVE_FILES="True" CPU_FRAC=0.20 DV=1.0 SIGMA_CLF_C=0.30 data_preprocess

# --- Total Sample for DV=1.0 and HOD_N = 0 --- #
# Run it separately
make HALOTYPE="so" N_FEAT_USE="all" CLF_METHOD=1 CLF_SEED=1235 DENS_CALC="False" TEST_TRAIN_OPT="boxes_n" BOX_IDX="0_3_4" REMOVE_MASTER="True" REMOVE_FILES="True" CPU_FRAC=0.80 DV=1.0 HOD_N=0 data_preprocess
make HALOTYPE="so" N_FEAT_USE="sub" CLF_METHOD=1 CLF_SEED=1235 DENS_CALC="False" TEST_TRAIN_OPT="boxes_n" BOX_IDX="0_3_4" REMOVE_MASTER="True" REMOVE_FILES="True" CPU_FRAC=0.80 DV=1.0 HOD_N=0 data_preprocess

# --- REAL DATA - Total Sample ---
# Run it separately
make HALOTYPE="so" CLF_METHOD=1 CLF_SEED=1235 DENS_CALC="False" TEST_TRAIN_OPT="boxes_n" BOX_IDX="0_3_4" REMOVE_MASTER="True" REMOVE_FILES="True" CPU_FRAC=0.75 DV=1.0 HOD_N=0 data_real_preprocess

# --- General Figures ---
# Run it separately
make HALOTYPE="so" CLF_METHOD=1 CLF_SEED=1235 DENS_CALC="False" TEST_TRAIN_OPT="boxes_n" BOX_IDX="0_3_4" CPU_FRAC=0.75 DV=1.0 HOD_N=0 catl_main_props

##############################################################################
##################### -------- ANALYSIS -------- #############################
##############################################################################

## --- Fixed HOD and DV ---
make HALOTYPE="so" N_FEAT_USE="all" CLF_METHOD=1 CLF_SEED=1235 DENS_CALC="False" TEST_TRAIN_OPT="boxes_n" BOX_IDX="0_3_4" REMOVE_FILES="True" CPU_FRAC=0.75 DV=1.0 HOD_N=0 ML_ANALYSIS="hod_dv_fixed" SAMPLE_METHOD="binning" BIN_VAL="nbins" ml_analysis
make HALOTYPE="so" N_FEAT_USE="sub" CLF_METHOD=1 CLF_SEED=1235 DENS_CALC="False" TEST_TRAIN_OPT="boxes_n" BOX_IDX="0_3_4" REMOVE_FILES="True" CPU_FRAC=0.75 DV=1.0 HOD_N=0 ML_ANALYSIS="hod_dv_fixed" SAMPLE_METHOD="binning" BIN_VAL="nbins" ml_analysis

## --- HOD Comparison - Same DV ---
make HALOTYPE="so" CLF_METHOD=1 CLF_SEED=1235 DENS_CALC="False" TEST_TRAIN_OPT="boxes_n" BOX_IDX="0_3_4" REMOVE_FILES="True" CPU_FRAC=0.75 DV=1.0 HOD_N=0 ML_ANALYSIS="dv_fixed" HOD_MODELS_N='0_1_2_3_4_5_6_7_8' SAMPLE_METHOD="binning" BIN_VAL="nbins" ml_analysis

## --- DV Comparison - Same HOD ---
make HALOTYPE="so" CLF_METHOD=1 CLF_SEED=1235 DENS_CALC="False" TEST_TRAIN_OPT="boxes_n" BOX_IDX="0_3_4" REMOVE_FILES="True" CPU_FRAC=0.75 DV=1.0 HOD_N=0 ML_ANALYSIS="hod_fixed" DV_MODELS_N='0.9_0.925_0.95_0.975_1.0_1.025_1.05_1.10' SAMPLE_METHOD="binning" BIN_VAL="nbins" ml_analysis

## --- Sigma_C Comparison - Same HOD ---
make HALOTYPE="so" CLF_METHOD=1 CLF_SEED=1235 DENS_CALC="False" TEST_TRAIN_OPT="boxes_n" BOX_IDX="0_3_4" REMOVE_FILES="False" CPU_FRAC=0.75 DV=1.0 HOD_N=0 ML_ANALYSIS="hod_dv_fixed_sigma_c" SIGMA_C_MOD_N='0.10_0.14_0.1417_0.16_0.18_0.20_0.22_0.24_0.26_0.28_0.30' SAMPLE_METHOD="binning" BIN_VAL="nbins" ml_analysis

##############################################################################
##################### -------- CATALOGUES -------- ###########################
##############################################################################

## --- Creating Catalogues for Mr19 --- ###
make HALOTYPE="so" CLF_METHOD=1 CLF_SEED=1235 DENS_CALC="False" TEST_TRAIN_OPT="boxes_n" BOX_IDX="0_3_4" REMOVE_FILES="True" CPU_FRAC=0.75 DV=1.0 HOD_N=0 ML_ANALYSIS="hod_dv_fixed" SAMPLE_METHOD="binning" BIN_VAL="nbins" CHOSEN_ML_ALG="xgboost"  SAMPLE_METHOD="binning" BIN_VAL="nbins" data_real_catl_create

## --- Creating plots for the REAL catalogues for Mr19 --- ###
make HALOTYPE="so" CLF_METHOD=1 CLF_SEED=1235 DENS_CALC="False" TEST_TRAIN_OPT="boxes_n" BOX_IDX="0_3_4" REMOVE_FILES="True" CPU_FRAC=0.75 DV=1.0 HOD_N=0 ML_ANALYSIS="hod_dv_fixed" SAMPLE_METHOD="binning" BIN_VAL="nbins" CHOSEN_ML_ALG="xgboost"  SAMPLE_METHOD="binning" BIN_VAL="nbins" data_real_catl_plots


##### --------------------------- 2018-09-03 --------------------------- #####

##############################################################################
##################### -------- PREPROCESSING -------- ########################
##############################################################################

# --- 1 Box --- #
make HALOTYPE="so" CLF_METHOD=1 DENS_CALC="False" TEST_TRAIN_OPT="box_sample_frac" BOX_TEST=0 SAMPLE_FRAC=0.1 REMOVE_MASTER="False" CPU_FRAC=0.25 REMOVE_FILES="True" DV=0.9   data_preprocess
make HALOTYPE="so" CLF_METHOD=1 DENS_CALC="False" TEST_TRAIN_OPT="box_sample_frac" BOX_TEST=0 SAMPLE_FRAC=0.1 REMOVE_MASTER="False" CPU_FRAC=0.25 REMOVE_FILES="True" DV=0.925 data_preprocess
make HALOTYPE="so" CLF_METHOD=1 DENS_CALC="False" TEST_TRAIN_OPT="box_sample_frac" BOX_TEST=0 SAMPLE_FRAC=0.1 REMOVE_MASTER="False" CPU_FRAC=0.25 REMOVE_FILES="True" DV=0.95  data_preprocess
make HALOTYPE="so" CLF_METHOD=1 DENS_CALC="False" TEST_TRAIN_OPT="box_sample_frac" BOX_TEST=0 SAMPLE_FRAC=0.1 REMOVE_MASTER="False" CPU_FRAC=0.25 REMOVE_FILES="True" DV=0.975 data_preprocess
make HALOTYPE="so" CLF_METHOD=1 DENS_CALC="False" TEST_TRAIN_OPT="box_sample_frac" BOX_TEST=0 SAMPLE_FRAC=0.1 REMOVE_MASTER="False" CPU_FRAC=0.25 REMOVE_FILES="True" DV=1.0   data_preprocess
make HALOTYPE="so" CLF_METHOD=1 DENS_CALC="False" TEST_TRAIN_OPT="box_sample_frac" BOX_TEST=0 SAMPLE_FRAC=0.1 REMOVE_MASTER="False" CPU_FRAC=0.25 REMOVE_FILES="True" DV=1.025 data_preprocess
make HALOTYPE="so" CLF_METHOD=1 DENS_CALC="False" TEST_TRAIN_OPT="box_sample_frac" BOX_TEST=0 SAMPLE_FRAC=0.1 REMOVE_MASTER="False" CPU_FRAC=0.25 REMOVE_FILES="True" DV=1.05  data_preprocess
make HALOTYPE="so" CLF_METHOD=1 DENS_CALC="False" TEST_TRAIN_OPT="box_sample_frac" BOX_TEST=0 SAMPLE_FRAC=0.1 REMOVE_MASTER="False" CPU_FRAC=0.25 REMOVE_FILES="True" DV=1.075 data_preprocess
make HALOTYPE="so" CLF_METHOD=1 DENS_CALC="False" TEST_TRAIN_OPT="box_sample_frac" BOX_TEST=0 SAMPLE_FRAC=0.1 REMOVE_MASTER="False" CPU_FRAC=0.25 REMOVE_FILES="True" DV=1.10  data_preprocess
# make HALOTYPE="so" CLF_METHOD=1 DENS_CALC="False" TEST_TRAIN_OPT="box_sample_frac" BOX_TEST=0 SAMPLE_FRAC=0.1 CPU_FRAC=0.75 REMOVE_FILES="True" DV=1.0   data_preprocess
#
# Multiple boxes
make HALOTYPE="so" CLF_METHOD=1 DENS_CALC="False" CPU_FRAC=0.10 TEST_TRAIN_OPT="boxes_n" BOX_IDX="0_4_5" REMOVE_MASTER="False" REMOVE_FILES="True" DV=1.0 HOD_N=0 data_preprocess
make HALOTYPE="so" CLF_METHOD=1 DENS_CALC="False" CPU_FRAC=0.10 TEST_TRAIN_OPT="boxes_n" BOX_IDX="0_4_5" REMOVE_MASTER="False" REMOVE_FILES="True" DV=1.0 HOD_N=1 data_preprocess
make HALOTYPE="so" CLF_METHOD=1 DENS_CALC="False" CPU_FRAC=0.10 TEST_TRAIN_OPT="boxes_n" BOX_IDX="0_4_5" REMOVE_MASTER="False" REMOVE_FILES="True" DV=1.0 HOD_N=2 data_preprocess
make HALOTYPE="so" CLF_METHOD=1 DENS_CALC="False" CPU_FRAC=0.10 TEST_TRAIN_OPT="boxes_n" BOX_IDX="0_4_5" REMOVE_MASTER="False" REMOVE_FILES="True" DV=1.0 HOD_N=3 data_preprocess
make HALOTYPE="so" CLF_METHOD=1 DENS_CALC="False" CPU_FRAC=0.10 TEST_TRAIN_OPT="boxes_n" BOX_IDX="0_4_5" REMOVE_MASTER="False" REMOVE_FILES="True" DV=1.0 HOD_N=4 data_preprocess
make HALOTYPE="so" CLF_METHOD=1 DENS_CALC="False" CPU_FRAC=0.10 TEST_TRAIN_OPT="boxes_n" BOX_IDX="0_4_5" REMOVE_MASTER="False" REMOVE_FILES="True" DV=1.0 HOD_N=5 data_preprocess
make HALOTYPE="so" CLF_METHOD=1 DENS_CALC="False" CPU_FRAC=0.10 TEST_TRAIN_OPT="boxes_n" BOX_IDX="0_4_5" REMOVE_MASTER="False" REMOVE_FILES="True" DV=1.0 HOD_N=6 data_preprocess
make HALOTYPE="so" CLF_METHOD=1 DENS_CALC="False" CPU_FRAC=0.10 TEST_TRAIN_OPT="boxes_n" BOX_IDX="0_4_5" REMOVE_MASTER="False" REMOVE_FILES="True" DV=1.0 HOD_N=7 data_preprocess
make HALOTYPE="so" CLF_METHOD=1 DENS_CALC="False" CPU_FRAC=0.10 TEST_TRAIN_OPT="boxes_n" BOX_IDX="0_4_5" REMOVE_MASTER="False" REMOVE_FILES="True" DV=1.0 HOD_N=8 data_preprocess
#
# --- Total Sample for DV=1.0 and HOD_N = 0 ---
make HALOTYPE="so" CLF_METHOD=1 DENS_CALC="False" CPU_FRAC=0.75 TEST_TRAIN_OPT="sample_frac" SAMPLE_FRAC=0.1 REMOVE_FILES="True" DV=1.0 HOD_N=0 REMOVE_MASTER="False" data_preprocess
#
# --- Original work - Data Science Symposium - Optional ---
make HALOTYPE="so" CLF_METHOD=1 DENS_CALC="False" CPU_FRAC=0.75 TEST_TRAIN_OPT="sample_frac" SAMPLE_FRAC=0.05 REMOVE_FILES="True" DV=1.0 HOD_N=0 REMOVE_MASTER="False" data_preprocess
#
# --- REAL DATA - Total Sample ---
# make HALOTYPE="so" CLF_METHOD=1 DENS_CALC="False" CPU_FRAC=0.75 TEST_TRAIN_OPT="sample_frac" SAMPLE_FRAC=0.1 REMOVE_FILES="True" DV=1.0 HOD_N=0 REMOVE_MASTER="True" data_real_preprocess
make HALOTYPE="so" CLF_METHOD=1 DENS_CALC="False" CPU_FRAC=0.75 TEST_TRAIN_OPT="boxes_n" SAMPLE_FRAC=0.1 REMOVE_FILES="True" DV=1.0 HOD_N=0 REMOVE_MASTER="True" data_real_preprocess

#
# --- General Figures ---
make HALOTYPE="so" CLF_METHOD=1 DENS_CALC="False" CPU_FRAC=0.75 TEST_TRAIN_OPT="sample_frac" SAMPLE_FRAC=0.1 REMOVE_FILES="True" DV=1.0 HOD_N=0 REMOVE_MASTER="False" catl_main_props

##############################################################################
##################### -------- ANALYSIS -------- #############################
##############################################################################

## --- Fixed HOD and DV
# make HALOTYPE="so" CLF_METHOD=1 DENS_CALC="False" CPU_FRAC=0.75 TEST_TRAIN_OPT="sample_frac" SAMPLE_FRAC=0.2 REMOVE_FILES="False" DV=1.0 HOD_N=0 SAMPLE_METHOD="binning" BIN_VAL="nbins" ML_ANALYSIS="hod_dv_fixed" PLOT_OPT="mhalo" ml_analysis
# make HALOTYPE="so" CLF_METHOD=1 DENS_CALC="False" CPU_FRAC=0.75 TEST_TRAIN_OPT="sample_frac" SAMPLE_FRAC=0.2 REMOVE_FILES="False" DV=1.0 HOD_N=0 SAMPLE_METHOD="subsample" RESAMPLE_OPT="under" BIN_VAL="nbins" ML_ANALYSIS="hod_dv_fixed" PLOT_OPT="mhalo" ml_analysis
# make HALOTYPE="so" CLF_METHOD=1 DENS_CALC="False" CPU_FRAC=0.75 TEST_TRAIN_OPT="sample_frac" SAMPLE_FRAC=0.2 REMOVE_FILES="False" DV=1.0 HOD_N=0 SAMPLE_METHOD="subsample" RESAMPLE_OPT="under" BIN_VAL="nbins" ML_ANALYSIS="hod_dv_fixed" PLOT_OPT="mgroup" ml_analysis
# make HALOTYPE="so" CLF_METHOD=1 DENS_CALC="False" CPU_FRAC=0.75 TEST_TRAIN_OPT="sample_frac" SAMPLE_FRAC=0.2 REMOVE_FILES="False" DV=1.0 HOD_N=0 SAMPLE_METHOD="subsample" RESAMPLE_OPT="over" BIN_VAL="nbins" ML_ANALYSIS="hod_dv_fixed" PLOT_OPT="mhalo" ml_analysis

# make HALOTYPE="so" CLF_METHOD=1 DENS_CALC="False" CPU_FRAC=0.75 TEST_TRAIN_OPT="sample_frac" SAMPLE_FRAC=0.1 REMOVE_FILES="False" DV=1.0 HOD_N=0  PLOT_OPT="mhalo" ml_analysis
# make HALOTYPE="so" CLF_METHOD=1 DENS_CALC="False" CPU_FRAC=0.75 TEST_TRAIN_OPT="sample_frac" SAMPLE_FRAC=0.1 REMOVE_FILES="False" DV=1.0 HOD_N=0  PLOT_OPT="mgroup" ml_analysis

##
## --- HOD Comparison - Same DV ---
# make ML_ANALYSIS="dv_fixed" HALOTYPE="so" CLF_METHOD=1 DENS_CALC="False" CPU_FRAC=0.75 TEST_TRAIN_OPT="boxes_n" REMOVE_FILES="False" DV=1.0 HOD_N=0 HOD_MODELS_N='0_1_2_3_4_5_6_7_8' SAMPLE_METHOD="subsample" RESAMPLE_OPT="under" BIN_VAL="nbins" ml_analysis
make ML_ANALYSIS="dv_fixed" HALOTYPE="so" CLF_METHOD=1 DENS_CALC="False" CPU_FRAC=0.75 TEST_TRAIN_OPT="boxes_n" REMOVE_FILES="True" DV=1.0 HOD_N=0 HOD_MODELS_N='0_1_2_3_4_5_6_7_8' SAMPLE_METHOD="binning" BIN_VAL="nbins" ml_analysis
##
## --- DV Comparison - Same HOD ---
# make ML_ANALYSIS="hod_fixed" HALOTYPE="so" CLF_METHOD=1 DENS_CALC="False" CPU_FRAC=0.75 TEST_TRAIN_OPT="box_sample_frac" REMOVE_FILES="False" DV=1.0 HOD_N=0 DV_MODELS_N='0.9_0.925_0.95_0.975_1.0_1.025_1.05_1.10' SAMPLE_METHOD="subsample" BIN_VAL="nbins" ml_analysis
make ML_ANALYSIS="hod_fixed" HALOTYPE="so" CLF_METHOD=1 DENS_CALC="False" CPU_FRAC=0.75 TEST_TRAIN_OPT="box_sample_frac" REMOVE_FILES="True" DV=1.0 HOD_N=0 DV_MODELS_N='0.9_0.925_0.95_0.975_1.0_1.025_1.05_1.10' SAMPLE_METHOD="binning" BIN_VAL="nbins" ml_analysis
##
## --- Fixed HOD and DV ---
# make ML_ANALYSIS='hod_dv_fixed' HALOTYPE='so' CLF_METHOD=1 DENS_CALC="False" CPU_FRAC=0.75 TEST_TRAIN_OPT="sample_frac" SAMPLE_FRAC=0.1 REMOVE_FILES="True" DV=1.0 HOD_N=0 SAMPLE_METHOD="normal" RESAMPLE_OPT="under" BIN_VAL="nbins" ml_analysis
# make ML_ANALYSIS='hod_dv_fixed' HALOTYPE='so' CLF_METHOD=1 DENS_CALC="False" CPU_FRAC=0.75 TEST_TRAIN_OPT="sample_frac" SAMPLE_FRAC=0.1 REMOVE_FILES="False" DV=1.0 HOD_N=0 SAMPLE_METHOD="binning" BIN_VAL="nbins" ml_analysis
make ML_ANALYSIS='hod_dv_fixed' HALOTYPE='so' CLF_METHOD=1 DENS_CALC="False" CPU_FRAC=0.75 TEST_TRAIN_OPT="boxes_n" SAMPLE_FRAC=0.1 REMOVE_FILES="True" DV=1.0 HOD_N=0 SAMPLE_METHOD="binning" BIN_VAL="nbins" ml_analysis

##############################################################################
##################### -------- CATALOGUES -------- ###########################
##############################################################################
##
## --- Creating Catalogues for Mr19 --- ###
make ML_ANALYSIS='hod_dv_fixed' HALOTYPE='so' CLF_METHOD=1 DENS_CALC="False" CPU_FRAC=0.75 TEST_TRAIN_OPT="boxes_n" SAMPLE_FRAC=0.1 REMOVE_FILES="True" DV=1.0 HOD_N=0 SAMPLE_METHOD="binning" BIN_VAL="nbins" CHOSEN_ML_ALG="xgboost"  data_real_catl_create
##
## --- Creating plots for the REAL catalogues for Mr19 --- ###
make ML_ANALYSIS='hod_dv_fixed' HALOTYPE='so' CLF_METHOD=1 DENS_CALC="False" CPU_FRAC=0.75 TEST_TRAIN_OPT="boxes_n" SAMPLE_FRAC=0.1 REMOVE_FILES="True" DV=1.0 HOD_N=0 SAMPLE_METHOD="binning" BIN_VAL="nbins" CHOSEN_ML_ALG="xgboost"  data_real_catl_plots



