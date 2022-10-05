#!/bin/bash
### Script for parallel estimation of bidning energies with Rosetta ###
#SBATCH --nodes=1 # request one node
#SBATCH --partition=<>
#SBATCH --time=48:00:00
#SBATCH --ntasks-per-node=2
#SBATCH --mem=0

#SBATCH --output=dg.%J.out
#SBATCH --error=dg.%J.err 
#SBATCH --job-name="dg" 

#SBATCH --array=0-1 # 0-N to run on N+1 nodes in parallel

# activate python environment

export PYTHONPATH=$PATH_TO_REPO:$PYTHONPATH
echo $PATH_TO_REPO

TARGET_PDB=$PATH_TO_REPO/data/herceptin_dataset/1n8z_Ag_trunc.pdb
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

#Virtual screening
cd ${REPOHOME} ; python3 $PATH_TO_REPO/generate_complexes_from_sequences.py $TARGET_PDB \
 $DIR/results/sequences_indices.fasta \
 --get_relaxed_complex \
 --partner_chains HL_X \
 --decoys 5 \
 --outdir $DIR \
 --slurm_cluster_config $PATH_TO_REPO/examples/slurm_config.json \
 --scratch_space $DIR/tmp_scratch_dg \
 --slurm_scale $DECOYS \
 --cdr_list h3 \
 --start ${start_run} \
 --end ${end}
