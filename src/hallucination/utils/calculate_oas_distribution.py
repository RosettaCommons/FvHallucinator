import os
import argparse
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt

from src.util.util import _aa_dict


def count_residues(csv_file):
    all_dict = {"heavy": {}, "light": {}}
    human_dict = {"heavy": {}, "light": {}}
    mouse_dict = {"heavy": {}, "light": {}}
    rat_dict = {"heavy": {}, "light": {}}

    skipped_species = []

    with open(csv_file, "r") as csv_f:
        # Skip headers
        csv_f.readline()
        for line in tqdm(csv_f):
            data = line.strip().split(",")
            seq = data[0]
            species = data[4].lower()
            chain = data[5].lower()

            if "human" in species:
                _dict = human_dict
            elif "mouse" in species:
                _dict = mouse_dict
            elif "rat" in species:
                _dict = rat_dict
            else:
                if species not in skipped_species:
                    skipped_species.append(species)
                    print("Skipping species: {}".format(species))

            for aa in list(seq):
                if aa in all_dict[chain]:
                    all_dict[chain][aa] += 1
                else:
                    all_dict[chain][aa] = 1

                if aa in _dict[chain]:
                    _dict[chain][aa] += 1
                else:
                    _dict[chain][aa] = 1

    return all_dict, human_dict, mouse_dict, rat_dict


def write_distribution(res_dict, out_file):
    total_res = sum(res_dict.values())
    aas = list(_aa_dict.keys())

    res_dist = [(res_dict[aa] / total_res) if aa in res_dict else 0
                for aa in aas]
    res_dist = [round(d, 5) for d in res_dist]

    np.savetxt(out_file, np.array(res_dist), fmt="%f")

    return res_dist


#Visualization
def write_pie(res_dist, out_file):
    plt.figure(figsize=[6, 6], dpi=400)
    plt.pie(x=res_dist,
            autopct="%.1f%%",
            labels=list(_aa_dict.keys()),
            explode=[0.02] * 20,
            pctdistance=0.85)

    title = os.path.split(out_file)[1][:-4]
    plt.title(title, fontsize=14)
    # plt.show()
    plt.savefig(out_file)
    plt.close()


def cli():
    desc = 'Converts OAS csv to fasta for clustering'
    parser = argparse.ArgumentParser(description=desc)
    parser.add_argument('oas_csv',
                        type=str,
                        help='Processed antibody sequence CSV files')

    args = parser.parse_args()
    oas_csv = args.oas_csv

    out_dir = "data/hallucination/ab_seq_distributions"

    os.system("mkdir -p {}".format(out_dir))

    res_dicts = count_residues(oas_csv)
    species = ["all", "human", "mouse", "rat"]
    out_file = os.path.join(out_dir, "{}_{}_dist.{}")

    for d, s in zip(res_dicts, species):
        h_dist_file = out_file.format(s, "heavy", "csv")
        h_dist = write_distribution(d["heavy"], h_dist_file)
        h_logo_file = out_file.format(s, "heavy", "png")
        write_pie(h_dist, h_logo_file)

        l_dist_file = out_file.format(s, "light", "csv")
        l_dist = write_distribution(d["light"], l_dist_file)
        l_logo_file = out_file.format(s, "light", "png")
        write_pie(l_dist, l_logo_file)


if __name__ == '__main__':
    cli()
