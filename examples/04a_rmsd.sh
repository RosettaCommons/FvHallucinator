#!/bin/bash

# activate environment

# set pythonpath
export PYTHONPATH=$PATH_TO_REPO:$PYTHONPATH
echo $PATH_TO_REPO

TARGET_PDB=$PATH_TO_REPO/data/herceptin_dataset/1n8z.truncated.pdb
DIR=$PATH_TO_REPO/examples/herceptin_cdrh3
echo $TARGET_PDB
echo $PREFIX
start_run=0
end=50
# consolidate designs from all folding runs 
python3 $PATH_TO_REPO/generate_fvs_from_sequences.py $TARGET_PDB \
 $DIR/results/sequences.fasta \
 --plot_consolidated_funnels \
 --path_forward_folded $DIR/forward_folding \
 --outdir $DIR \
 --cdr_list h3 \
 --start ${start_run} \
 --end ${end}
