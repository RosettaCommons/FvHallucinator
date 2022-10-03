#!/bin/bash

# activate virtual environment

# set pythonpath
export PYTHONPATH=$PATH_TO_REPO:$PYTHONPATH
echo $PATH_TO_REPO

TARGET_PDB=$PATH_TO_REPO/data/herceptin_dataset/1n8z.truncated.pdb
PREFIX=$PATH_TO_REPO/examples/herceptin_cdrh3
echo $TARGET_PDB
echo $PREFIX

# Generating 50 designs; recommended number of designs for cdrh3 is > 500.
start=0
stop=1
for ((j = $start; j < $stop; j++)); do
python3 $PATH_TO_REPO/hallucinate.py \
--target $TARGET_PDB \
--iterations 50 \
--suffix $j \
--prefix $PREFIX \
--seed $j \
--cdr_list h3 \
--disallow_aas_at_all_positions C
done
