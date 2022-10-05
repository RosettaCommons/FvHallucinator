#!/bin/bash

#activate python environment

export PYTHONPATH=$PATH_TO_REPO:$PYTHONPATH
echo $PATH_TO_REPO

TARGET_PDB=$PATH_TO_REPO/data/herceptin_dataset/1n8z_Ag_trunc.pdb
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
python3 $PATH_TO_REPO/generate_complexes_from_sequences.py $TARGET_PDB \
 $DIR/results/sequences_indices.fasta \
 --plot_consolidated_dG \
 --outdir $DIR \
 --cdr h3 \
 --start ${start_run} \
 --end ${end}

