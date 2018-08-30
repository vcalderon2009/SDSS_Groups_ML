##
## Date executed: 2018-05-31
##
#
# 1 Box
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
# Total Sample for DV=1.0 and HOD_N = 0
make HALOTYPE="so" CLF_METHOD=1 DENS_CALC="False" CPU_FRAC=0.75 TEST_TRAIN_OPT="sample_frac" SAMPLE_FRAC=0.1 REMOVE_FILES="False" DV=1.0 HOD_N=0 data_preprocess
#
# Original work - Data Science Symposium
make HALOTYPE="so" CLF_METHOD=1 DENS_CALC="False" CPU_FRAC=0.75 TEST_TRAIN_OPT="sample_frac" SAMPLE_FRAC=0.05 REMOVE_FILES="False" DV=1.0 HOD_N=0 data_preprocess
#
# General Figures
make HALOTYPE="so" CLF_METHOD=1 DENS_CALC="False" CPU_FRAC=0.75 TEST_TRAIN_OPT="sample_frac" SAMPLE_FRAC=0.1 REMOVE_FILES="False" DV=1.0 HOD_N=0 catl_main_props


