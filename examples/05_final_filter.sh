#!/bin/bash

# activate python environment

export PYTHONPATH=$PATH_TO_REPO:$PYTHONPATH
echo $PATH_TO_REPO

TARGET_PDB_COMP=$PATH_TO_REPO/data/herceptin_dataset/1n8z_Ag_trunc.pdb
PREFIX=$PATH_TO_REPO/examples/herceptin_cdrh3
echo $TARGET_PDB
echo $PREFIX


DIR=$PATH_TO_REPO/examples/herceptin_cdrh3

start_run=0
end=50
echo ${start_run}
echo ${end}

#Virtual screening
cd ${REPOHOME}
python3 $PATH_TO_REPO/filter.py $TARGET_PDB_COMP \
 --csv_forward_folded $DIR/forward_folding/results/consolidated_ff_lowest_N050.csv \
 --csv_complexes $DIR/virtual_binding/relaxed_mutants_data/results/improved_dG_sequences_0-50.csv \
 --rmsd_filter H3,2.0 \
 --outdir $DIR/results_filtered_output \
 --cdr_list h3
