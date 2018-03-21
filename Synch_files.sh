#!/bin/sh
#
# Description: Synchronizes files and figures from Bender
#
# Parameters
# ----------
# file_opt: string
#     Options:
#         - h (help)
#         - halobias
#         - catalogues
#         - interim
#         - data

# Defining Directory
DIR_LOCAL="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
BENDER_SSH="caldervf@bender.accre.vanderbilt.edu:"
DIR_BENDER="${BENDER_SSH}/fs1/caldervf/Repositories/Large_Scale_Structure/SDSS/SDSS_Groups_ML"
# option
file_opt=$1
echo "\n==> Option: ${file_opt}"
## Help option
usage="Synch_files.sh [-h] [file_opt] -- Program that synchronizes files between 'local' and Bender
where:
    -h           show this help text

    Options for 'file_opt':
        - 'catalogues'  Synchronizes catalogues files in 'data' folder
        - 'interim'     Synchronizes all files in 'interim' folder
        - 'data'        Synchronizes all files in 'processed' folder"

if [[ ${file_opt} == '-h' ]]; then
  echo "==> Usage: $usage\n"
  # exit 0
fi
##
## Synchronizing
# Catalogues
if [[ ${file_opt} == 'catalogues' ]]; then
    echo "==> rsync -chavzP --stats "${DIR_BENDER}/data/processed/SDSS" "${DIR_LOCAL}/data/processed/"\n"
    rsync -chavzP --stats "${DIR_BENDER}/data/processed/SDSS" "${DIR_LOCAL}/data/processed/"
fi
# Interim Files
if [[ ${file_opt} == 'interim' ]]; then
    echo "==> rsync -chavzP --stats "${DIR_BENDER}/data/interim" "${DIR_LOCAL}/data/"\n"
    rsync -chavzP --stats "${DIR_BENDER}/data/interim" "${DIR_LOCAL}/data/"
fi
# Data Files
if [[ ${file_opt} == 'data' ]]; then
    echo "==> rsync -chavzP --exclude '*.ff' --stats "${DIR_BENDER}/data/raw" "${DIR_LOCAL}/data/"\n"
    rsync -chavzP --exclude '*.ff' --exclude '*Mr20*.hdf5' --exclude '*Mr21*.hdf5' --stats "${DIR_BENDER}/data/raw" "${DIR_LOCAL}/data/"
    echo "==> rsync -chavzP --exclude '*.ff' --stats "${DIR_BENDER}/data/processed" "${DIR_LOCAL}/data/"\n"
    rsync -chavzP --exclude '*.ff' --exclude '*Mr20*.hdf5' --exclude '*Mr21*.hdf5' --stats "${DIR_BENDER}/data/processed" "${DIR_LOCAL}/data/"
    echo "==> rsync -chavzP --exclude '*.ff' --stats "${DIR_BENDER}/data/interim" "${DIR_LOCAL}/data/"\n"
    rsync -chavzP --exclude '*.ff' --exclude '*Mr20*.hdf5' --exclude '*Mr21*.hdf5' --stats "${DIR_BENDER}/data/interim" "${DIR_LOCAL}/data/"
fi
