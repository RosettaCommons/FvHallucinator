# FvHallucinator
The code for [FvHallucinator](https://www.biorxiv.org/content/10.1101/2022.06.06.494991v3) is made available under the [Rosetta-DL license](https://github.com/RosettaCommons/Rosetta-DL/blob/main/LICENSE.md) as part of the [Rosetta-DL bundle](https://github.com/RosettaCommons/Rosetta-DL).

FvHallucinator designs sequences that fold into a desired Fv structure by leveraging a pretrained sequence-to-structure prediction DL model, [DeepAb](https://www.sciencedirect.com/science/article/pii/S2666389921002804) (Ruffolo et al. 2021 Patterns). We adapted the [trDesign](https://www.pnas.org/doi/10.1073/pnas.2017228118) (Norn et al. 2021 PNAS) approach where the problem of predicting sequence given structure has been reframed as the problem of maximizing the conditional probability of a sequence given structure. In the case of the Fv, we are primarily interested in designing a subset of the residues (CDRs, VH-VL interface), so we split the sequence S into fixed and designable positions, SF and SD. We then seek the design subsequence SD that maximizes the conditional probability of the sequence S given a target structure T and the fixed sequence SF. For more details, please refer to [Mahajan et al. 2022](https://www.biorxiv.org/content/10.1101/2022.06.06.494991v3).

All hallucinated sequences from publication are available on [Zenodo](10.5281/zenodo.7076478).

The FvHallucinator reository builds upon the [DeepAb](https://github.com/RosettaCommons/DeepAb) repository.

# Related repositories

trDesign: https://github.com/gjoni/trDesign <br />
RFDesign: https://github.com/RosettaCommons/RFDesign <br />
DeepAb: https://github.com/RosettaCommons/DeepAb <br />

# Requirements
FvHallucinator requires python3.6 or higher. For a full list of requirements, see requirements.txt.
For folding hallucinated sequences with DeepAb, you will additionally need a [PyRosetta](https://www.pyrosetta.org) license (for installing pyrosetta, use conda or with setup.py).

# Getting Started
Start by setting up a python virtual environment (or conda) with python3.6 or higher
```bash
python3 -m venv <path_to_env> 
source <path_to_env>/bin/activate
# Use the requirements.txt file to install dependencies
python3 -m pip install -r requirements.txt
```

# Download pretrained DeepAb model
Pretrained DeepAb models need 330MB of free space.
To download, run the following commands in your terminal.
```bash
wget https://data.graylab.jhu.edu/ensemble_abresnet_v1.tar.gz
tar -xvzf ensemble_abresnet_v1.tar.gz
```
Move the unzipped folder to <path_to_repo>/trained_models. The hallucinate.py script expects the model weights to be present in <path_to_repo>/trained_models/ensemble_abresnet/*.pt
# Designing CDR loops with FvHallucinator
We recommend running hallucination on GPUs. Designs can be generated in parallel.
## Unrestricted hallucination
To design CDR loops for a target CDR conformation, run unrestricted hallucination. In this mode of hallucination, sequences are only constrained by the target structure/conformation.

The pipeline requires all pdbs to be chothia-numbered. To obtain chothia-numbered pdbs, we recommend [Abnum](http://www.bioinf.org.uk/abs/abnum/) or [ANARCI](https://github.com/oxpig/ANARCI).
Below is an example bash script (also see examples/run_hallucination.sh). (For all options, run python3 hallucinate.py -h)
```bash
#!/bin/bash

# activate virtual environment

# set pythonpath
export PATH_TO_REPO=<path_to_repo>
export PYTHONPATH=$PATH_TO_REPO:$PYTHONPATH

# chothia-numbered target structure for the Fv region
# example pdb for Trastuzumab Fv from data
TARGET_PDB=$PATH_TO_REPO/data/1n8z.truncated.pdb

# name of the output folder
PREFIX=$PATH_TO_REPO/examples/hallucination_cdrh3

# Generating 50 designs; recommended number of designs for cdrh3 is > 500.
start=0
stop=50

# --seed <> manual seeding each design with a different seed
# --suffix <> suffix to use for each design
# disallow the method from designing cysteines at all positions

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

```
This script will generate hallucination trajectories and final sequences in $PREFIX/trajectories/

## Designing any subsequence on the Fv
It is also possible to design other subsequences on the Fv regions with the following options:
```bash
--indices <string of indices to design with chains and chothia numbering> # e.g. h:20,31A/l:56,57
--hl_interface # design residues at the Vh-Vl interface (only non-cdr residues)
--framework # design all framework residues
--exclude <string of indices to exclude from design with chains and chothia numbering> # e.g. h:31A,52/l:56,57
# --exclude can be combined with --hl_interface, --framework, --cdr_list
```
If no design region is specified, the full Fv will be designed. This mode was not explored in the published work and we do not recommend it.

## Post-processing and generating sequence logos

```bash
python3 $PATH_TO_REPO/process_designs.sh \
  --trajectory_path $PREFIX \
  --target $TARGET_PDB \
  --cdr h3 \
  --outdir $PREFIX/results #where the post-processing results will be stored
```

Results will include sequences of all CDR H3 designs in the file $PREFIX/results/sequences_indices.fasta, full Fv sequence of all designs in $PREFIX/results/sequences.fasta and sequence logos.

To compare hallucinated designs with PyIgClassify, you can additionally specify the path for PyIgClassify database. The latest version of this database can be downloaded from [here](http://dunbrack2.fccc.edu/PyIgClassify/Download/Download.aspx). Alternatively, you can use the database used to generate data for the publication from data/cdr_clusters_pssm_dict.pkl with the option:
```bash
--cdr_cluster_database data/cdr_clusters_pssm_dict.pkl
```
This option only works with the option ```--cdr <cdr_name>```.

For post-processing designs at the Vh-Vl interface, we additionally use ANARCI. This must be installed as described [here](https://github.com/oxpig/ANARCI). If ANARCI is not installed, FR scores will not be calculated.

## Hallucination with wildtype seeding
Design positions can be initialized with residues from the starting antibody (target_pdb) instead of random initialization with ``` --seed_with_WT ```.

## Restricted hallucination
You can additionally guide hallucination towards relevant sequence spaces with sequence based losses as described below.

### Sequence-restricted hallucination
This mode adds a loss during optimization to keep the designed sequence close to the starting sequence. To enable this loss set a non-zero weight for sequence loss with ```--seq_loss_weight 25 ```, where the weight determines the relative weight of the sequence loss and geometric loss. We recommend weights between 10-30. A higher weight will lead to designs closer to starting sequence and vice-versa.
### Motif-restricted hallucination
This mode adds a loss during optimization to sample specified design positions from a restricted set of amino acids at a desired frequency/proportion. For example, to specify that position 100A (must be chothia numbered) on the CDR H3 loop, samples tyrosine and trytophan in equal proportions use options, 

```bash
--restricted_positions_kl_loss_weight 100 \ #recommended loss weight
--restrict_positions_to_freq h:100A-W=0.50-Y=0.50 \
```
## Other options
For a full list of options, run ```python3 hallucinate.py -h ```.

## Folding hallucinated sequences with DeepAb

For folding hallucinated sequences with DeepAb and obtaining RMSDs, run:
```bash

# sequences.fasta file is generated by process_designs.py
# See examples/03_forward_folding.py for a parallel version using dask distributed

start_run=0
end=10
python3 $PATH_TO_REPO/generate_fvs_from_sequences.py $TARGET_PDB \
 $DIR/results/sequences.fasta \
 --pdbs_from_model \
 --decoys 5 \
 --outdir $DIR \
 --scratch_space $DIR/tmp_scratch
 --slurm_cluster_config config.json
 --cdr_list h3 \
 --start ${start_run} \
 --end ${end}

# consolidate designs from all folding runs 
python3 $PATH_TO_REPO/generate_fvs_from_sequences.py $TARGET_PDB \
 $DIR/results/sequences.fasta \
 --plot_consolidated_funnels \
 --path_forward_folded $DIR/forward_folding \
 --outdir $DIR \
 --cdr_list h3 \
 --start ${start_run} \
 --end ${end}
```

We recommend running folding on a cluster (cpus). When the cluster option is enabled with ```--slurm_cluster_config config.json ```, dask will generate decoys in parallel. Using options ```--start and --end ```, many such scripts can be run in parallel to fold chunks (e.g. 0-10, 10-20, 100-200 etc.) of designed sequences.

Example config file for slurm cluster
```json
{
	"cores": 1,
	"memory": "9GB",
	"processes": 1,
	"queue": "defq",
	"job_cpu": 2,
	"walltime": "10:00:00"
}
```

The folded pdbs will be in  $DIR/forward_folding/ and the consolidated root-mean-square-deviations with respect to the target pdb will be in $DIR/forward_folding/results

## Virtual Screening with Rosetta
To virtually screen hallucinated designs, provide a pdb with the structure of the **antibody (Fv only) and the antigen** and run:

```bash
python3 $PATH_TO_REPO/generate_complexes_from_sequences.py $TARGET_PDB_COMP \
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
python3 $PATH_TO_REPO/generate_complexes_from_sequences.py $TARGET_PDB \
 $DIR/results/sequences_indices.fasta \
 --plot_consolidated_dG \
 --outdir $DIR \
 --cdr h3 \
 --start ${start_run} \
 --end ${end}
```

## Filtering final set of designs for folding and binding

To obtain designs that satisfy both folding and binding criteria, run:

```bash
python3 $PATH_TO_REPO/filter.py $TARGET_PDB_COMP \
 --csv_forward_folded $DIR/forward_folding/results/consolidated_ff_lowest_N010.csv \
 --csv_complexes $DIR/virtual_binding/relaxed_mutants_data/results/improved_dG_sequences_0-10.csv \
 --rmsd_filter H3,2.0 \
 --outdir $DIR/results_filtered_output \
 --cdr_list h3
```

Here, we selected designs wth CDR H3 RMSD less than or equal to 2.0 angstrom.
You can select based on multiple thresholds as a json file with the option 

```bash
--rmsd_filter_json <json file>  # specify CDR RMSD cutoffs with respect to the starting antibody
```

# Citations
```
Mahajan, S. P., Ruffolo, J. A., Frick, R., & Gray, J. J. A deep learning framework to hallucinate structure-conditioned and antigen-specific antibodies. bioRxiv 2022.06.06.494991 (2022)

```

