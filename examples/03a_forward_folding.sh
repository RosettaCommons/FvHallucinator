#!/bin/bash
### Script for parallel forward folding of designed sequences with DeepAb ###
#SBATCH --nodes=1 # request one node
#SBATCH --partition=<partition_name>
#SBATCH --time=48:00:00
#SBATCH --ntasks-per-node=2
#SBATCH --mem=0

#SBATCH --output=ff.%J.out
#SBATCH --error=ff.%J.err 
#SBATCH --job-name="ff" 

#SBATCH --array=0-1 # 0-N to run on N+1 nodes in parallel

#activate virtual environment

export PYTHONPATH=$PATH_TO_REPO:$PYTHONPATH
echo $PATH_TO_REPO

TARGET_PDB=$PATH_TO_REPO/data/herceptin_dataset/1n8z.truncated.pdb
PREFIX=$PATH_TO_REPO/examples/herceptin_cdrh3
echo $TARGET_PDB
echo $PREFIX


DIR=$PATH_TO_REPO/examples/herceptin_cdrh3

cur_id=$SLURM_ARRAY_TASK_ID
next_id=$(( SLURM_ARRAY_TASK_ID + 1 ))
start_run=$(( cur_id*25 ))
end=$(( next_id*25 ))
echo ${start_run}
echo ${end}
DECOYS=5

#Fold
cd ${REPOHOME} ; python3 $PATH_TO_REPO/generate_fvs_from_sequences.py $TARGET_PDB \
 $DIR/results/sequences.fasta \
 --pdbs_from_model \
 --decoys 5 \
 --outdir $DIR \
 --slurm_cluster_config $PATH_TO_REPO/examples/slurm_config.json \
 --scratch_space $DIR/tmp_scratch_ff \
 --slurm_scale $DECOYS \
 --cdr_list h3 \
 --start ${start_run} \
 --end ${end}
