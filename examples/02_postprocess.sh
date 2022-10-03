#!/bin/bash

# activate virtual environment
module unload python
module load anaconda
conda deactivate
conda activate /scratch16/jgray21/smahaja4_active/repositories/conda_deeph3_hal

# set pythonpath
export PYTHONPATH=$PATH_TO_REPO:$PYTHONPATH
echo $PATH_TO_REPO

TARGET_PDB=$PATH_TO_REPO/data/herceptin_dataset/1n8z.truncated.pdb
PREFIX=$PATH_TO_REPO/examples/herceptin_cdrh3
echo $TARGET_PDB
echo $PREFIX

python3 $PATH_TO_REPO/process_designs.py \
  --trajectory_path $PREFIX \
  --target $TARGET_PDB \
  --cdr h3 \
  --outdir $PREFIX/results\
  --cdr_cluster_database data/cdr_clusters_pssm_dict.pkl

