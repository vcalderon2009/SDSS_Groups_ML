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
# 2018-09-03
#
##############################################################################

##############################################################################
##################### -------- PREPROCESSING -------- ########################
##############################################################################

# --- 1 Box --- #
make HALOTYPE="so" CLF_METHOD=1 DENS_CALC="False" TEST_TRAIN_OPT="box_sample_frac" BOX_TEST=0 SAMPLE_FRAC=0.1 CPU_FRAC=0.25 REMOVE_FILES="False" DV=0.9   data_preprocess
make HALOTYPE="so" CLF_METHOD=1 DENS_CALC="False" TEST_TRAIN_OPT="box_sample_frac" BOX_TEST=0 SAMPLE_FRAC=0.1 CPU_FRAC=0.25 REMOVE_FILES="False" DV=0.925 data_preprocess
make HALOTYPE="so" CLF_METHOD=1 DENS_CALC="False" TEST_TRAIN_OPT="box_sample_frac" BOX_TEST=0 SAMPLE_FRAC=0.1 CPU_FRAC=0.25 REMOVE_FILES="False" DV=0.95  data_preprocess
make HALOTYPE="so" CLF_METHOD=1 DENS_CALC="False" TEST_TRAIN_OPT="box_sample_frac" BOX_TEST=0 SAMPLE_FRAC=0.1 CPU_FRAC=0.25 REMOVE_FILES="False" DV=0.975 data_preprocess
make HALOTYPE="so" CLF_METHOD=1 DENS_CALC="False" TEST_TRAIN_OPT="box_sample_frac" BOX_TEST=0 SAMPLE_FRAC=0.1 CPU_FRAC=0.25 REMOVE_FILES="False" DV=1.0   data_preprocess
make HALOTYPE="so" CLF_METHOD=1 DENS_CALC="False" TEST_TRAIN_OPT="box_sample_frac" BOX_TEST=0 SAMPLE_FRAC=0.1 CPU_FRAC=0.25 REMOVE_FILES="False" DV=1.025 data_preprocess
make HALOTYPE="so" CLF_METHOD=1 DENS_CALC="False" TEST_TRAIN_OPT="box_sample_frac" BOX_TEST=0 SAMPLE_FRAC=0.1 CPU_FRAC=0.25 REMOVE_FILES="False" DV=1.05  data_preprocess
make HALOTYPE="so" CLF_METHOD=1 DENS_CALC="False" TEST_TRAIN_OPT="box_sample_frac" BOX_TEST=0 SAMPLE_FRAC=0.1 CPU_FRAC=0.25 REMOVE_FILES="False" DV=1.075 data_preprocess
make HALOTYPE="so" CLF_METHOD=1 DENS_CALC="False" TEST_TRAIN_OPT="box_sample_frac" BOX_TEST=0 SAMPLE_FRAC=0.1 CPU_FRAC=0.25 REMOVE_FILES="False" DV=1.10  data_preprocess
make HALOTYPE="so" CLF_METHOD=1 DENS_CALC="False" TEST_TRAIN_OPT="box_sample_frac" BOX_TEST=0 SAMPLE_FRAC=0.1 CPU_FRAC=0.75 REMOVE_FILES="False" DV=1.0 data_preprocess
#
# Multiple boxes
make HALOTYPE="so" CLF_METHOD=1 DENS_CALC="False" CPU_FRAC=0.10 TEST_TRAIN_OPT="boxes_n" BOX_IDX="0_4_5" REMOVE_FILES="False" DV=1.0 HOD_N=0 data_preprocess
make HALOTYPE="so" CLF_METHOD=1 DENS_CALC="False" CPU_FRAC=0.10 TEST_TRAIN_OPT="boxes_n" BOX_IDX="0_4_5" REMOVE_FILES="False" DV=1.0 HOD_N=1 data_preprocess
make HALOTYPE="so" CLF_METHOD=1 DENS_CALC="False" CPU_FRAC=0.10 TEST_TRAIN_OPT="boxes_n" BOX_IDX="0_4_5" REMOVE_FILES="False" DV=1.0 HOD_N=2 data_preprocess
make HALOTYPE="so" CLF_METHOD=1 DENS_CALC="False" CPU_FRAC=0.10 TEST_TRAIN_OPT="boxes_n" BOX_IDX="0_4_5" REMOVE_FILES="False" DV=1.0 HOD_N=3 data_preprocess
make HALOTYPE="so" CLF_METHOD=1 DENS_CALC="False" CPU_FRAC=0.10 TEST_TRAIN_OPT="boxes_n" BOX_IDX="0_4_5" REMOVE_FILES="False" DV=1.0 HOD_N=4 data_preprocess
make HALOTYPE="so" CLF_METHOD=1 DENS_CALC="False" CPU_FRAC=0.10 TEST_TRAIN_OPT="boxes_n" BOX_IDX="0_4_5" REMOVE_FILES="False" DV=1.0 HOD_N=5 data_preprocess
make HALOTYPE="so" CLF_METHOD=1 DENS_CALC="False" CPU_FRAC=0.10 TEST_TRAIN_OPT="boxes_n" BOX_IDX="0_4_5" REMOVE_FILES="False" DV=1.0 HOD_N=6 data_preprocess
make HALOTYPE="so" CLF_METHOD=1 DENS_CALC="False" CPU_FRAC=0.10 TEST_TRAIN_OPT="boxes_n" BOX_IDX="0_4_5" REMOVE_FILES="False" DV=1.0 HOD_N=7 data_preprocess
make HALOTYPE="so" CLF_METHOD=1 DENS_CALC="False" CPU_FRAC=0.10 TEST_TRAIN_OPT="boxes_n" BOX_IDX="0_4_5" REMOVE_FILES="False" DV=1.0 HOD_N=8 data_preprocess
#
# --- Total Sample for DV=1.0 and HOD_N = 0 ---
make HALOTYPE="so" CLF_METHOD=1 DENS_CALC="False" CPU_FRAC=0.75 TEST_TRAIN_OPT="sample_frac" SAMPLE_FRAC=0.1 REMOVE_FILES="False" DV=1.0 HOD_N=0 data_preprocess
#
# --- Original work - Data Science Symposium ---
make HALOTYPE="so" CLF_METHOD=1 DENS_CALC="False" CPU_FRAC=0.75 TEST_TRAIN_OPT="sample_frac" SAMPLE_FRAC=0.05 REMOVE_FILES="False" DV=1.0 HOD_N=0 data_preprocess
#
# --- General Figures ---
make HALOTYPE="so" CLF_METHOD=1 DENS_CALC="False" CPU_FRAC=0.75 TEST_TRAIN_OPT="sample_frac" SAMPLE_FRAC=0.1 REMOVE_FILES="False" DV=1.0 HOD_N=0 catl_main_props

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
make ML_ANALYSIS="dv_fixed" HALOTYPE="so" CLF_METHOD=1 DENS_CALC="False" CPU_FRAC=0.75 TrueEST_TRAIN_OPT="boxes_n" REMOVE_FILES="False" DV=1.0 HOD_N=0 HOD_MODELS_N='0_1_2_3_4_5_6_7_8' SAMPLE_METHOD="binning" BIN_VAL="nbins" ml_analysis
##
## --- DV Comparison - Same HOD ---
# make ML_ANALYSIS="hod_fixed" HALOTYPE="so" CLF_METHOD=1 DENS_CALC="False" CPU_FRAC=0.75 TEST_TRAIN_OPT="box_sample_frac" REMOVE_FILES="False" DV=1.0 HOD_N=0 DV_MODELS_N='0.9_0.925_0.95_0.975_1.0_1.025_1.05_1.10' SAMPLE_METHOD="subsample" BIN_VAL="nbins" ml_analysis
make ML_ANALYSIS="hod_fixed" HALOTYPE="so" CLF_METHOD=1 DENS_CALC="False" CPU_FRAC=0.75 TEST_TRAIN_OPT="box_sample_frac" REMOVE_FILES="False" DV=1.0 HOD_N=0 DV_MODELS_N='0.9_0.925_0.95_0.975_1.0_1.025_1.05_1.10' SAMPLE_METHOD="binning" BIN_VAL="nbins" ml_analysis
##
## --- Fixed HOD and DV ---
# make ML_ANALYSIS='hod_dv_fixed' HALOTYPE='so' CLF_METHOD=1 DENS_CALC="False" CPU_FRAC=0.75 TEST_TRAIN_OPT="sample_frac" SAMPLE_FRAC=0.1 REMOVE_FILES="False" DV=1.0 HOD_N=0 SAMPLE_METHOD="subsample" RESAMPLE_OPT="under" BIN_VAL="nbins" ml_analysis
make ML_ANALYSIS='hod_dv_fixed' HALOTYPE='so' CLF_METHOD=1 DENS_CALC="False" CPU_FRAC=0.75 TEST_TRAIN_OPT="sample_frac" SAMPLE_FRAC=0.1 REMOVE_FILES="False" DV=1.0 HOD_N=0 SAMPLE_METHOD="binning" BIN_VAL="nbins" ml_analysis
