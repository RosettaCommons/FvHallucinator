# FvHallucinator
The code for [FvHallucinator](https://www.biorxiv.org/content/10.1101/2022.06.06.494991v3) are made available under the [Rosetta-DL](https://github.com/RosettaCommons/Rosetta-DL) license as part of the Rosetta-DL bundle.

# Requirements
FvHallucinator requires python3.6 or higher. For a full list of requirements, see requirements.txt.
For folding hallucinated sequences, you will additionally need a Pyrosetta license (for installing pyrosetta, use conda)

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
--restricted_positions_kl_loss_weight 100 \
--restrict_positions_to_freq h:100A-W=0.50-Y=0.50 \
```
