# FvHallucinator
The code for [FvHallucinator](https://www.biorxiv.org/content/10.1101/2022.06.06.494991v3) are made available under the [Rosetta-DL](https://github.com/RosettaCommons/Rosetta-DL) license as part of the Rosetta-DL bundle.

FvHallucinator designs sequences that fold into a desired Fv structure by leveraging a pretrained sequence-to-structure prediction DL model. We adapted the trDesign (Norn 2021 Nature) approach where the problem of predicting sequence given structure has been reframed as the problem of maximizing the conditional probability of a sequence given structure. In the case of the Fv, we are primarily interested in designing a subset of the residues (CDRs, VH-VL interface), so we split the sequence S into fixed and designable positions, SF and SD. We then seek the design subsequence SD that maximizes the conditional probability of the sequence S given a target structure T and the fixed sequence SF.


# Requirements
FvHallucinator requires python3.6 or higher. For a full list of requirements, see requirements.txt.
For folding hallucinated sequences with DeepAb, you will additionally need a Pyrosetta license (for installing pyrosetta, use conda).

# Getting Started
Start by setting up a python virtual environment (or conda) with python3.6 or higher
```
python3 -m venv <path_to_env> 
source <path_to_env>/bin/activate
# Use the requirements.txt file to install dependencies
python3 -m pip install -f requirements.txt
```
# Designing CDR loops with FvHallucinator
We recommend running hallucination on gpus. Designs can be generated in parallel.
## Unrestricted hallucination
To design CDR loops for a target CDR conformation, run unrestricted hallucination.
In this mode of hallucination, sequences are only constrained by the target structure/conformation.
Below is an example bash script. (For all options, run python3 hallucinate.py -h)
```
#!/bin/bash
#activate virtual environment
export PYTHONPATH=<path_to_FvHallucinator>:$PYTHONPATH
# Generating 50 designs; recommended number of designs for cdrh3 is > 500.
TARGET_PDB=<chothia_numbered_pdb>
PREFIX=hallucination_cdrh3
start=0
stop=50
for ((j = $start; j <= $stop; j++)); do
python3 -W ignore hallucinate.py \
  --target $TARGET_PDB \ # **chothia numbered target structure of the Fv region**
  --iterations 50 \
  --suffix $j \ #suffix to use for design
  --prefix $PREFIX \ # name of the output folder
  --seed $j \ # seeding each design with a different seed
  --cdr_list h3 \
  --disallow_aas_at_all_positions C #disallow the method from designing cysteines at all positions
done
```
This script will generate hallucination trajectories and final sequences in $PREFIX/trajectories/

## Designing any subsequence on the Fv
It is also possible to design other subsequences on the Fv regions with the following options:
```
# if not option is specified, all cdrs will be designed
--indices <string of indices to design with chains and chothia numbering> # e.g. h:20,31A/l:56,57
--hl_interface # design residues at the Vh-Vl interface (only non-cdr residues)
--framework # design all framework residues
--exclude <string of indices to exclude from design with chains and chothia numbering> # e.g. h:31A,52/l:56,57
# --exclude can be combined with --hl_interface, --framework, --cdr_list
```
If no design region is specified, the full Fv will be designed. This mode was not explored in the published work and we do not recommend it.

## Post-processing and generating sequence logos
```
python3 -W ignore process_designs.py \
  --trajectory_path $PREFIX \
  --target $TARGET_PDB \
  --cdr h3 \
  --outdir $PREFIX/results #where the post-processing results will be stored
```
Results will include sequences of all h3 designs in the file $PREFIX/results/sequences_indices.fasta, full Fv sequence of all designs in $PREFIX/results/sequences.fasta and sequence logos.

## Hallucination with wildtype seeding
Hallucinated designs can be seeded with residues from the starting antibody (target_pdb) instead of random initialization with ``` --seed_with_WT ```.

## Restricted hallucination
You can additionally guide hallucination towards relevant sequence spaces with sequence based losses as described below.

### Sequence-restricted hallucination
This mode adds a loss during optimization to keep the designed sequence close to the starting sequence. To enable this loss set a non-zero weight for sequence loss with ```--seq_loss_weight 25 ```, where the weight determines the relative weight of the sequence loss and geometric loss. We recommend weights between 10-30. A higher weight will lead to designs closer to starting sequence and vice-versa.
### Motif-restricted hallucination
This mode adds a loss during optimization to sample specified design positions from a restricted set of amino acids at a desired frequency/proportion. For example, to specify that position 100A (must be chothia numbered) on the cdr h3 loop, samples tyrosine and trytophan in equal proportions use options, 
```
--restricted_positions_kl_loss_weight 100 \ #recommended loss weight
--restrict_positions_to_freq h:100A-W=0.50-Y=0.50 \
```
## Other options
For a full list of options, run ```python3 hallucinate.py -h ```.

## Folding hallucinated sequences with DeepAb
For folding hallucinated sequences with DeepAb and obtaining RMSDs, run:
```
start_run=0
end=10
python3 generate_fvs_from_sequences.py $TARGET_PDB \
 $DIR/results/sequences.fasta \ # this file was generated by process_designs.py
 --pdbs_from_model \ #This option is for folding
 --decoys 5 \ #2-5 decoys is more than sufficient
 --outdir $DIR \
 --scratch_space $DIR/tmp_scratch \ #if using a cluster
 --slurm_cluster_config config.json \ #if using a cluster, provide json file with cluster config for dask
 --cdr h3 \
 --model $MODEL_FILE \
 --start ${start_run} \
 --end ${end}

# consolidate designs from all folding runs 
python3 generate_fvs_from_sequences.py $TARGET_PDB \
 $DIR/results/sequences.fasta \
 --plot_consolidated_funnels \
 --path_forward_folded $DIR/forward_folding \
 --outdir $DIR \
 --cdr_list h3 \
 --start ${start_run} \
 --end ${end}
```
We recommend running folding on a cluster (cpus). When the cluster option is enabled with ```--slurm_cluster_config config.json ```, dask will generate decoys in paralle. Using options ```--start and --end ```, many such scripts can be run in parallel to fold chunks (e.g. 0-10, 10-20, 100-200 etc.) of designed sequences.
This step requires pyrosetta.

The folded pdbs will be in  $DIR/forward_folding/ and the consolidated root-mean-square-deviations with respect to the target pdb will be in $DIR/forward_folding/results

## Virtual Screening with Rosetta
To virtually screen hallucinated designs, provide a pdb with the structure of the **antibody (Fv only)** and the antigen and run:
```
python3 generate_complexes_from_sequences.py $TARGET_PDB_COMP \
 $DIR/results/sequences_indices.fasta \
 --get_relaxed_complex \ #this option is for virtual screening
 --partner_chains HL_X \ #provide chain names for antibody and antigen chains separated by an underscore
 --outdir $DIR \
 --cdr_list h3 \
 --start ${start_run} \
 --end ${end} \
 --slurm_cluster_config config.json \
 --scratch_space $DIR/tmp_scratch_dg

# consolidate from parallel runs and screen
python3 generate_complexes_from_sequences.py $TARGET_PDB \
 $DIR/results/sequences_indices.fasta \
 --plot_consolidated_dG \
 --outdir $DIR \
 --indices h:95,96,97,98,99,100,100A,100B,100C,101 \
 --start ${start_run} \
 --end ${end}
```

## Filtering final set of designs for folding and binding

```
python3 filter.py $TARGET_PDB_COMP \
 --csv_forward_folded $DIR/forward_folding/results/consolidated_ff_lowest_N010.csv \
 --csv_complexes $DIR/virtual_binding/relaxed_mutants_data/results/improved_dG_sequences_0-10.csv \
 --rmsd_filter H3,2.0 \
 --outdir $DIR/results_filtered_output \
 --cdr_list h3
```

