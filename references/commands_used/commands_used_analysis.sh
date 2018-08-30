## Fixed HOD and DV
make HALOTYPE="so" CLF_METHOD=1 DENS_CALC="False" CPU_FRAC=0.75 TEST_TRAIN_OPT="sample_frac" SAMPLE_FRAC=0.2 REMOVE_FILES="False" DV=1.0 HOD_N=0 SAMPLE_METHOD="binning" BIN_VAL="nbins" ML_ANALYSIS="hod_dv_fixed" PLOT_OPT="mhalo" ml_analysis
make HALOTYPE="so" CLF_METHOD=1 DENS_CALC="False" CPU_FRAC=0.75 TEST_TRAIN_OPT="sample_frac" SAMPLE_FRAC=0.2 REMOVE_FILES="False" DV=1.0 HOD_N=0 SAMPLE_METHOD="subsample" RESAMPLE_OPT="under" BIN_VAL="nbins" ML_ANALYSIS="hod_dv_fixed" PLOT_OPT="mhalo" ml_analysis
make HALOTYPE="so" CLF_METHOD=1 DENS_CALC="False" CPU_FRAC=0.75 TEST_TRAIN_OPT="sample_frac" SAMPLE_FRAC=0.2 REMOVE_FILES="False" DV=1.0 HOD_N=0 SAMPLE_METHOD="subsample" RESAMPLE_OPT="under" BIN_VAL="nbins" ML_ANALYSIS="hod_dv_fixed" PLOT_OPT="mgroup" ml_analysis
make HALOTYPE="so" CLF_METHOD=1 DENS_CALC="False" CPU_FRAC=0.75 TEST_TRAIN_OPT="sample_frac" SAMPLE_FRAC=0.2 REMOVE_FILES="False" DV=1.0 HOD_N=0 SAMPLE_METHOD="subsample" RESAMPLE_OPT="over" BIN_VAL="nbins" ML_ANALYSIS="hod_dv_fixed" PLOT_OPT="mhalo" ml_analysis

make HALOTYPE="so" CLF_METHOD=1 DENS_CALC="False" CPU_FRAC=0.75 TEST_TRAIN_OPT="sample_frac" SAMPLE_FRAC=0.1 REMOVE_FILES="False" DV=1.0 HOD_N=0  PLOT_OPT="mhalo" ml_analysis
make HALOTYPE="so" CLF_METHOD=1 DENS_CALC="False" CPU_FRAC=0.75 TEST_TRAIN_OPT="sample_frac" SAMPLE_FRAC=0.1 REMOVE_FILES="False" DV=1.0 HOD_N=0  PLOT_OPT="mgroup" ml_analysis





##
## HOD Comparison - Same DV
# make ML_ANALYSIS="dv_fixed" HALOTYPE="so" CLF_METHOD=1 DENS_CALC="False" CPU_FRAC=0.75 TEST_TRAIN_OPT="boxes_n" REMOVE_FILES="False" DV=1.0 HOD_N=0 HOD_MODELS_N='0_1_2_3_4_5_6_7_8' SAMPLE_METHOD="subsample" RESAMPLE_OPT="under" BIN_VAL="nbins" ml_analysis
make ML_ANALYSIS="dv_fixed" HALOTYPE="so" CLF_METHOD=1 DENS_CALC="False" CPU_FRAC=0.75 TrueEST_TRAIN_OPT="boxes_n" REMOVE_FILES="False" DV=1.0 HOD_N=0 HOD_MODELS_N='0_1_2_3_4_5_6_7_8' SAMPLE_METHOD="binning" BIN_VAL="nbins" ml_analysis
##
## DV Comparison - Same HOD
# make ML_ANALYSIS="hod_fixed" HALOTYPE="so" CLF_METHOD=1 DENS_CALC="False" CPU_FRAC=0.75 TEST_TRAIN_OPT="box_sample_frac" REMOVE_FILES="False" DV=1.0 HOD_N=0 DV_MODELS_N='0.9_0.925_0.95_0.975_1.0_1.025_1.05_1.10' SAMPLE_METHOD="subsample" BIN_VAL="nbins" ml_analysis
make ML_ANALYSIS="hod_fixed" HALOTYPE="so" CLF_METHOD=1 DENS_CALC="False" CPU_FRAC=0.75 TEST_TRAIN_OPT="box_sample_frac" REMOVE_FILES="False" DV=1.0 HOD_N=0 DV_MODELS_N='0.9_0.925_0.95_0.975_1.0_1.025_1.05_1.10' SAMPLE_METHOD="binning" BIN_VAL="nbins" ml_analysis
##
## Fixed HOD and DV
# make ML_ANALYSIS='hod_dv_fixed' HALOTYPE='so' CLF_METHOD=1 DENS_CALC="False" CPU_FRAC=0.75 TEST_TRAIN_OPT="sample_frac" SAMPLE_FRAC=0.1 REMOVE_FILES="False" DV=1.0 HOD_N=0 SAMPLE_METHOD="subsample" RESAMPLE_OPT="under" BIN_VAL="nbins" ml_analysis
make ML_ANALYSIS='hod_dv_fixed' HALOTYPE='so' CLF_METHOD=1 DENS_CALC="False" CPU_FRAC=0.75 TEST_TRAIN_OPT="sample_frac" SAMPLE_FRAC=0.1 REMOVE_FILES="False" DV=1.0 HOD_N=0 SAMPLE_METHOD="binning" BIN_VAL="nbins" ml_analysis