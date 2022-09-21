import matplotlib.pyplot as plt
import numpy as np
from src.hallucination.params import latest_models
from src.hallucination.utils.util import comma_separated_chain_indices_to_dict, get_indices_from_different_methods
from src.util.pdb import get_pdb_chain_seq, get_pdb_numbering_from_residue_indices
from src.util.util import _aa_dict
import pandas as pd
import seaborn as sns
import os, sys
from src.hallucination.utils.sequence_utils import get_annotations, aalist_ordered


def plot_heavy_light_heatmap(df_list, labels_list, outfile, ref_seq=''):

    fig, axes = plt.subplots(2, 1)
    fig.set_size_inches(28, 10)

    center = 0.0
    annotations_list = None
    if ref_seq != '':
        #print(ref_seq)
        annotations = get_annotations(ref_seq,
                                      [t for t in range(len(ref_seq))],
                                      reorder=True)
        annotations_list = [
            annotations[:, :len(labels_list[0])],
            annotations[:, len(labels_list[0]):]
        ]
        #np.savetxt('test_0.txt', annotations_list[0], delimiter=',', fmt='%s')
        
    xtick_every = 2
    for i, df_in in enumerate(df_list):
        df_in.index = pd.CategoricalIndex(df_in.index,
                                          categories=aalist_ordered)
        df_in.sort_index(level=0, inplace=True)
        #print(annotations_list[i].shape, df_in.shape)
        #print(df_in)
        ax = sns.heatmap(df_in,
                         ax=axes[i],
                         cmap=sns.color_palette("vlag_r", as_cmap=True),
                         linewidths=0.3,
                         linecolor='black',
                         vmin=-0.10,
                         vmax=0.05,
                         center=center,
                         annot=annotations_list[i],
                         fmt="s",
                         xticklabels=xtick_every,
                         yticklabels=1,
                         cbar_kws={'fraction':0.08, 'shrink':0.7, 'pad': 0.02})
        #plt.draw()
        ax.set_yticklabels(aalist_ordered, fontsize=15, rotation=0)
        ax.set_xticklabels([labels_list[i][j] for j in range(0, len(labels_list[i]), xtick_every)], fontsize=16, rotation=45)
        #ax.figure.axes[-1].yaxis.set_ticklabels(ax.figure.axes[-1].yaxis.get_ticklabels(), fontsize=9)
        #ax.set_xlabel('Position', fontsize=22)
        ax.set_xlabel('')
        ax.set_ylabel('AA', fontsize=28)
    fig.tight_layout()
    plt.savefig(outfile, dpi=600)
    plt.close()


def plot_indices_heatmap(df, labels, indices, outfile, ref_seq='',
                        vmin=-0.10, vmax=0.05, center=0.0):

    fig = plt.figure()
    annot_sel = None
    if ref_seq != '':
        annotations = get_annotations(ref_seq,
                                      [t for t in range(len(ref_seq))],
                                      reorder=True)
        annot_sel = annotations[:, indices]
        print(annot_sel)


    df.index = pd.CategoricalIndex(df.index,
                                   categories=aalist_ordered)
    df.sort_index(level=0, inplace=True)
    ax = sns.heatmap(df,
                    cmap=sns.color_palette("vlag_r", as_cmap=True),
                    linewidths=0.3,
                    linecolor='black',
                    vmin=vmin,
                    vmax=vmax,
                    center=center,
                    annot=annot_sel,
                    fmt="s",
                    xticklabels=1,
                    yticklabels=1,
                    cbar_kws={'fraction':0.08, 'shrink':0.7, 'pad': 0.02})
    ax.set_yticklabels(aalist_ordered, fontsize=15, rotation=0)
    ax.set_xlabel('')
    ax.set_xticklabels(labels, fontsize=16, rotation=45)
    ax.set_ylabel('AA', fontsize=18)
    fig.tight_layout()
    plt.savefig(outfile, dpi=600, transparent=True)
    plt.close()


def plot_cce_from_csv(combined_csv, pdb, outfile, plot=True):
    df_cce_all = pd.read_csv(combined_csv)
    
    heavy_seq, light_seq = get_pdb_chain_seq(pdb,
                                             'H'),\
                           get_pdb_chain_seq(pdb,
                                             'L')
    len_h, _ = len(heavy_seq), len(light_seq)
    df_heavy = df_cce_all[df_cce_all['pos'] < len_h]
    df_light = df_cce_all[df_cce_all['pos'] >= len_h]

    df_heavy_cce = df_heavy.pivot(index='mut_aa', columns='pos', values='dcce')
    df_light_cce = df_light.pivot(index='mut_aa', columns='pos', values='dcce')

    if plot:

        indices_in_data = list(set(list(df_cce_all['pos'])))
        print(indices_in_data)
        total_h_indices = len([t for t in indices_in_data if t < len_h])
        labels = get_pdb_numbering_from_residue_indices(
            pdb, indices_in_data)
        labels_heavy = labels[:total_h_indices]
        labels_light = labels[total_h_indices:]
        heavy_seq_data = ''.join([s for i,s in enumerate(heavy_seq) if i in indices_in_data])
        light_seq_data = ''.join([s for i,s in enumerate(light_seq) if i+len_h in indices_in_data])
        print(heavy_seq_data, len(heavy_seq_data))
        print(light_seq_data, len(light_seq_data))

        plot_heavy_light_heatmap([df_heavy_cce, df_light_cce],
                                [labels_heavy, labels_light],
                                outfile,
                                ref_seq=heavy_seq_data + light_seq_data
                                )

    return df_heavy_cce, df_light_cce


def plot_cce_from_csvfiles(pdb, csvfile_path, plot=True):
    import glob
    csvfiles = glob.glob(csvfile_path + '*.csv')
    #print(csvfiles)

    df_list = []
    for csvfile in csvfiles:
        print(csvfile, os.path.basename(csvfile))
        if os.path.basename(csvfile) == 'combined_dcce.csv':
            continue
        if os.path.getsize(csvfile) == 0:
            pos = int(
                os.path.basename(csvfile).split('.csv')[0].split('_')[-1])
            df = pd.DataFrame(index=np.arange(20),
                              columns=['mut_aa', 'pos', 'seq', 'dcce', 'ref_cce', 'wt_aa'])
            df['pos'] = pos
            df['mut_aa'] = list(_aa_dict.keys())
        else:
            df = pd.read_csv(csvfile, sep=',')
        df_list.append(df)

    df_cce_all_unsorted = pd.concat(df_list).reset_index()
    if 'aa' and 'cce' in df_cce_all_unsorted.columns:
        df_cce_all_unsorted.rename(columns={'aa': 'mut_aa', 'cce': 'dcce'}, inplace=True)
    df_cce_all = df_cce_all_unsorted.sort_values(by=['pos', 'mut_aa'])
    pos_list = list(df_cce_all['pos'])
    ch_pos_list = get_pdb_numbering_from_residue_indices(
            pdb, pos_list)
    df_cce_all['chothia_pos'] = [t.rstrip().upper() for t in ch_pos_list]
    combined_csvfile = '{}/combined_dcce.csv'.format(csvfile_path)
    df_cce_all.to_csv(combined_csvfile, sep=',')
    
    outfile = os.path.join(csvfile_path, 'dcce_matrix.png')
    return plot_cce_from_csv(combined_csvfile, pdb, outfile, plot=plot)
    


def plot_hallucinated_matrices(geom_matrices,
                               prob_matrices,
                               hallucination_iterations,
                               outfile='entropy_geometries_{}.png',
                               sub_region=None):
    def highlight_indices(cur_ax):
        for i_sr in range(sub_region):
            cur_ax.get_xticklabels()[i_sr].set_color('red')
            cur_ax.get_xticklabels()[i_sr].set_color('red')

    # assumes latest model
    #plot distances
    fig, ax = plt.subplots(2, len(geom_matrices[:3]))
    fig.suptitle('Distances from Entropy Minimization')
    for j in range(len(geom_matrices[:3])):
        im = ax[0, j].imshow(geom_matrices[j])
        fig.colorbar(im, ax=ax[0, j], shrink=0.6)
        #if not sub_region is None:
        #    highlight_indices(ax[0, j])

    for j in range(len(prob_matrices[:3])):
        im = ax[1, j].imshow(prob_matrices[j], vmin=0, vmax=0.5)
        fig.colorbar(im, ax=ax[1, j], shrink=0.6)
        #if not sub_region is None:
        #    highlight_indices(ax[1, j])

    fig.tight_layout()
    plt.savefig(outfile.format('dist_%s' % (hallucination_iterations)),
                transparent=True,
                dpi=300)
    plt.close()

    #plot orientations
    fig, ax = plt.subplots(2, len(geom_matrices[3:]))
    fig.suptitle('Orientations from Entropy Minimization')
    for j in range(len(geom_matrices[3:])):
        im = ax[0, j].imshow(geom_matrices[j])
        fig.colorbar(im, ax=ax[0, j], shrink=0.6)
        #if not sub_region is None:
        #    highlight_indices(ax[0, j])

    for j in range(len(prob_matrices[3:])):
        im = ax[1, j].imshow(prob_matrices[j], vmin=0, vmax=0.5)
        fig.colorbar(im, ax=ax[1, j], shrink=0.6)
        #if not sub_region is None:
        #    highlight_indices(ax[1, j])

    fig.tight_layout()
    plt.savefig(outfile.format('orientations_%s' % (hallucination_iterations)),
                transparent=True,
                dpi=300)
    plt.close()


def plot_losses(geom_loss_list,
                hallucination_iterations,
                outfile,
                wt_geom_loss=None,
                plot_loglosses=False):
    cycle = plt.rcParams['axes.prop_cycle'].by_key()['color']
    legend_keys_dist = [r"$d_{CA}$", r"$d_{CB}$", r"$d_{NO}$", r"$WT$"]
    legend_keys_or = [r"$\omega$", r"$\theta$", r"$\phi$", r"$WT$"]
    n_d = 3

    axs = plt.figure(figsize=(12, 4)).subplots(1, 2)
    ax = axs[0]
    if wt_geom_loss is not None:
        ax.hlines(np.array(wt_geom_loss)[:n_d], 0, hallucination_iterations, linestyles='dashed',\
               label='WT', linewidths=0.5,
               colors=cycle[:len(wt_geom_loss[:n_d])])

    ax.plot(np.array(geom_loss_list)[:, :n_d])
    ax.set_xlim((0, hallucination_iterations))
    ax.set_title("Target Distances")
    ax.set_xlabel("Iteration")
    ax.set_ylabel("CCE Loss")
    ax.legend(legend_keys_dist)

    #Orientations
    ax = axs[1]
    if wt_geom_loss is not None:
        ax.hlines(np.array(wt_geom_loss)[n_d:], 0,
               hallucination_iterations, linestyles='dashed',\
               label='WT', linewidths=0.5,
               colors=cycle[:len(wt_geom_loss[n_d:])])

    ax.plot(np.array(geom_loss_list)[:, n_d:])
    ax.set_xlim((0, hallucination_iterations))
    ax.set_title("Target Orientations")
    ax.set_xlabel("Iteration")
    ax.set_ylabel("CCE Loss")
    ax.legend(legend_keys_or)

    plt.tight_layout()
    plt.savefig(outfile, dpi=600)
    plt.close()

    #log plot for distances to zoom into loss
    if plot_loglosses:
        plt.figure(figsize=(6, 4), dpi=500)
        plt.subplot(1, 1, 1)
        if wt_geom_loss is not None:
            plt.hlines(np.array(wt_geom_loss)[:n_d], 0, hallucination_iterations, linestyles='dashed',\
                label='WT', linewidths=0.5,
                colors=cycle[:len(wt_geom_loss[:n_d])])

        plt.plot(np.array(geom_loss_list)[:, :n_d])
        plt.xlim((0, hallucination_iterations))
        plt.yscale('log')
        plt.title("Target Distances")
        plt.xlabel("Iteration")
        plt.ylabel("CCE Loss")
        plt.legend(legend_keys_dist)
        plt.tight_layout()
        plt.savefig(outfile.split('.png')[0] + '_logloss.png',
                    transparent=True)
        plt.close()


def plot_losses_kl_res(kl_loss_list,
                       hallucination_iterations,
                       outfile,
                       ylabel="KL Divergence Loss",
                       title=""):
    plt.figure(figsize=(6, 4), dpi=500)
    plt.subplot(1, 1, 1)
    plt.plot(np.array(kl_loss_list))
    plt.xlim((0, hallucination_iterations))
    plt.title(title)
    plt.xlabel("Iteration")
    plt.ylabel(ylabel)
    plt.tight_layout()
    plt.savefig(outfile, transparent=True)
    plt.close()


def plot_losses_kl_bg(kl_loss_list,
                      hallucination_iterations,
                      outfile,
                      title=""):
    legend_keys_dist = [r"$d_{CA}$", r"$d_{CB}$", r"$d_{NO}$"]
    legend_keys_or = [r"$\omega$", r"$\theta$", r"$\phi$"]
    n_d = 3
    
    axs = plt.figure(figsize=(4, 20)).subplots(6, 1, sharex=True)
    legend_keys_all = legend_keys_dist + legend_keys_or
    n_all = n_d + 3
    for i in range(n_all):
        ax = axs[i]
        ax.plot(np.array(kl_loss_list)[:, i])
        ax.set_xlim((0, hallucination_iterations))
        ax.set_ylabel("KL Divergence Loss")
        print(legend_keys_all[i])
    axs[n_all - 1].set_xlabel("Iteration")

    axs[n_all-1].set_xlim((0, hallucination_iterations))
    plt.suptitle(title)
    plt.tight_layout()
    plt.savefig(outfile, transparent=True)
    plt.close()


def plot_losses_seq_regularize(reg_loss_list, hallucination_iterations,
                               outfile):
    plt.figure(figsize=(6, 4), dpi=500)
    plt.subplot(1, 1, 1)
    plt.plot(np.array([t[0] for t in reg_loss_list]), label='heavy')
    plt.plot(np.array([t[1] for t in reg_loss_list]), label='light')
    plt.xlim((0, hallucination_iterations))
    plt.title("Sequence Regularization Loss")
    plt.xlabel("Iteration")
    plt.ylabel("KL Divergence Loss")
    plt.legend()
    plt.tight_layout()
    plt.savefig(outfile, transparent=True)
    plt.close()


def plot_all_losses(traj_loss_dict,
                    outfile_base,
                    hallucination_iterations,
                    wt_geom_loss=None):

    if 'geom' in traj_loss_dict:
        outfile = outfile_base.format('geom')
        plot_losses(traj_loss_dict['geom'],
                    hallucination_iterations,
                    outfile,
                    wt_geom_loss=wt_geom_loss)

    if 'kl_res' in traj_loss_dict:
        outfile = outfile_base.format('kl_res')
        plot_losses_kl_res(traj_loss_dict['kl_res'], hallucination_iterations,
                           outfile, title="Sequence Restriction Loss")

    if 'kl_bg' in traj_loss_dict:
        outfile = outfile_base.format('kl_bg')
        plot_losses_kl_bg(traj_loss_dict['kl_bg'], hallucination_iterations,
                           outfile, title="KL Geometric Loss")

    if 'reg_seq' in traj_loss_dict:
        outfile = outfile_base.format('regularize')
        plot_losses_seq_regularize(traj_loss_dict['reg_seq'],
                                   hallucination_iterations, outfile)

    if 'entropy' in traj_loss_dict:
        outfile = outfile_base.format('entropy')
        plot_losses(traj_loss_dict['entropy'], hallucination_iterations,
                    outfile)

    if 'seq' in traj_loss_dict:
        outfile = outfile_base.format('seq')
        plot_losses_kl_res(traj_loss_dict['seq'], hallucination_iterations,
                           outfile, ylabel='Seq Loss', title="Sequence Loss")

    if 'netcharge' in traj_loss_dict:
        outfile = outfile_base.format('netcharge')
        plot_losses_kl_res(traj_loss_dict['netcharge'], hallucination_iterations,
                           outfile, ylabel='L1 Loss', title="Total Charge Loss")

    if 'max_aa_freq' in traj_loss_dict:
        outfile = outfile_base.format('aa_freq')
        plot_losses_kl_res(traj_loss_dict['max_aa_freq'], hallucination_iterations,
                           outfile, ylabel='L1 Loss', title="AA Frequency Loss")
    if 'binding' in traj_loss_dict:
        outfile = outfile_base.format('binding')
        plot_losses_kl_res(traj_loss_dict['binding'], hallucination_iterations,
                           outfile, ylabel='ddG_per_design_pos', title="Binding Loss")


def plot_worst_loss_offenders(loss_file, target_pdb, indices_str='', upper_q = 0.90):
    import glob, torch

    h_seq, l_seq = get_pdb_chain_seq(target_pdb, 'H'), get_pdb_chain_seq(target_pdb, 'L')
    ref_seq = h_seq + l_seq

    if indices_str != '':
        dict_indices = comma_separated_chain_indices_to_dict(indices_str)
        indices_hal = get_indices_from_different_methods(
                    target_pdb, include_indices=dict_indices)
        print(indices_hal)
        
    all_indices = get_indices_from_different_methods(
                    target_pdb)
    labels = get_pdb_numbering_from_residue_indices(target_pdb, all_indices)
    
    losses = np.load(loss_file, allow_pickle=True)
    losses_base = torch.tensor(losses).squeeze_(1).mean(dim=-1, keepdim=True)
    losses = torch.tensor(losses_base).expand(-1,-1,10).numpy()
    
    
    uq_array = np.zeros((losses_base.shape[0]))
    fig, axes = plt.subplots(2, 3)
    fig.suptitle('Mean per-residue loss quantile: {}'.format(upper_q))
    count=0
    for i in range(2):
        for j in range(3):
            high_losses = torch.quantile(losses_base[count].squeeze(-1), upper_q, dim=-1)
            #print(i, high_losses)
            uq_array[count]=torch.min(high_losses).item()
            axes[i,j].boxplot(losses_base[count].squeeze_(-1).numpy())
            axes[i,j].axhline(uq_array[count])
            count+=1
    fig.tight_layout()
    outpng = loss_file.split('.npy')[0] + '_per_res_mean_dist.png'
    plt.savefig(outpng, dpi=600)
    plt.close()
    for i in range(losses.shape[0]):
        fig, axes = plt.subplots(1, 2)
        im = axes[0].imshow(losses[i])
        plt.colorbar(im, ax=axes[0], shrink=0.5)
        axes[0].set_xticklabels('')
        losses[i][losses[i]>=uq_array[i]]=np.Infinity
        im = axes[1].imshow(losses[i])
        plt.colorbar(im, ax=axes[1], shrink=0.5)
        axes[1].set_xticklabels('')
        fig.tight_layout()
        outpng = loss_file.split('.npy')[0] + '_per_res_mean_{}.png'.format(i)
        plt.savefig(outpng, dpi=600)
        plt.close()


    losses = np.load(loss_file, allow_pickle=True)
    losses_base = torch.tensor(losses).squeeze_(1).sum(dim=-1, keepdim=True)
    losses = torch.tensor(losses_base).expand(-1,-1,10).numpy()
    
    fig, axes = plt.subplots(2, 3)
    fig.suptitle('Mean per-residue loss quantile: {}'.format(upper_q))
    uq_array = np.zeros((losses_base.shape[0]))
    count=0
    for i in range(2):
        for j in range(3):
            high_losses = torch.quantile(losses_base[count].squeeze(-1), upper_q, dim=-1)
            #print(i, high_losses)
            uq_array[count]=torch.min(high_losses).item()
            axes[i,j].boxplot(losses_base[count].squeeze_(-1).numpy())
            axes[i,j].axhline(uq_array[count])
            count+=1
    fig.tight_layout()
    outpng = loss_file.split('.npy')[0] + '_per_res_total_dist.png'
    plt.savefig(outpng, dpi=600)
    plt.close()

    for i in range(losses.shape[0]):
        fig, axes = plt.subplots(1, 2)
        im1 = axes[0].imshow(losses[i])
        plt.colorbar(im1, ax=axes[0], shrink=0.5)
        axes[0].set_xticklabels('')

        losses[i][losses[i]>=uq_array[i]]=np.Infinity
        im2 = axes[1].imshow(losses[i])
        plt.colorbar(im2, ax=axes[1], shrink=0.5)
        axes[1].set_xticklabels('')
        fig.tight_layout()
        outpng = loss_file.split('.npy')[0] + '_per_res_total_{}.png'.format(i)
        plt.savefig(outpng, dpi=600)
        plt.close()
