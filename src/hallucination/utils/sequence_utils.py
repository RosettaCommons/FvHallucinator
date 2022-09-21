
from util.util import letter_to_num, _aa_dict
from torch.nn import functional as F
from scipy.stats import entropy as scipy_entropy
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import torch
import logomaker
import textdistance
import imageio
import os

biopython_developability_keys = ['Charge at pH7', 'Gravy', 'Instability Index']

columns_logo_aa = list(_aa_dict.keys())
cmap = sns.cubehelix_palette(rot=-.2, as_cmap=True)

aalist_ordered = [
    'L', 'V', 'A', 'I', 'M', 'C', 'P', 'G', 'F', 'W', 'Y', 'S', 'T', 'N', 'Q',
    'H', 'K', 'R', 'D', 'E'
]

aa_dict_reordered = {}
for i, aa in enumerate(aalist_ordered):
    aa_dict_reordered[aa] = '{}'.format(i)

#Visualization
def makelogo(df_logo, dict_residues={ 'reslist':[], 'labellist':[]}, outfile='templogo.png', show=False, mode='prob',\
            add_xticks=True, add_yticks=True, text='', transform_to='', transform_from='probability'):
    #plt.style.use('classic')
    N = len(df_logo)
    fig = plt.figure(figsize=((N)*0.6,1.5*3))
    ax = plt.gca()
    if transform_to != '':
        df_logo = df_logo.replace(0.0, 1e-3)
        df_logo = logomaker.transform_matrix(df_logo, from_type=transform_from, to_type=transform_to)
        ss_logo = logomaker.Logo(df_logo,
                             font_name='Stencil Std',
                             color_scheme='NajafabadiEtAl2017',
                             vpad=.1,
                             width=.8,
                             shade_below=.5,
                             fade_below=.5,
                             ax=ax)
    else:
        ss_logo = logomaker.Logo(df_logo,
                             font_name='Stencil Std',
                             color_scheme='NajafabadiEtAl2017',
                             vpad=.1,
                             width=.8,
                             ax=ax)
    #ss_logo = logomaker.Logo(df_logo)
    if add_xticks:
        if len(dict_residues['reslist']) > 20:
            ss_logo.style_xticks(anchor=0, spacing=5, rotation=45)
        
        ss_logo.ax.set_xticks(range(1, len(dict_residues['reslist']) + 1))
        if 'labellist' in dict_residues:
            ss_logo.ax.set_xticklabels(dict_residues['labellist'],
                                        fontsize=30)
        else:
            ss_logo.ax.set_xticklabels(dict_residues['reslist'],
                                        fontsize=20)
        plt.xticks(rotation=45)
    if add_yticks:
        if mode == 'bits':
            ss_logo.ax.set_yticks([0.0, 1.0, 2.0, 3.0, 4.0])
            ss_logo.ax.set_yticklabels([0.0, 1.0, 2.0, 3.0, 4.0], fontsize=15)
        if mode == 'prob':
            ss_logo.ax.set_yticks([0.0, 0.25, 0.5, 0.75, 1.0])
            ss_logo.ax.set_yticklabels([0.0, 0.25, 0.5, 0.75, 1.0],
                                       fontsize=18)
    else:
        ss_logo.ax.set_yticks([])
        ss_logo.ax.set_yticklabels([])
    ss_logo.ax.yaxis.set_ticks_position('none')
    ss_logo.ax.yaxis.set_tick_params(pad=-1)
        
    plt.box(False)
    if text != '':
        plt.text(0.5, -0.5, text)
    plt.tight_layout()
    plt.savefig(outfile, dpi=600)

    if show:
        plt.show()
    plt.close()



#Visualization
def makelogo_with_reference(df_logos, df_labels, dict_residues={ 'reslist':[], 'labellist':[]}, outfile='templogo.png', show=False, mode='prob',\
            add_xticks=True, add_yticks=True):
    N = len(df_logos[0])
    fig, axes = plt.subplots(1, len(df_logos), figsize=((N)*0.65*len(df_logos),1.5*3))
    for i in range(len(df_logos)):
        ss_logo = logomaker.Logo(df_logos[i],
                                font_name='Stencil Std',
                                color_scheme='NajafabadiEtAl2017',
                                vpad=.1,
                                width=.8,
                                ax=axes[i])
        if add_xticks:
            ss_logo.style_xticks(rotation=45)
            ss_logo.ax.set_xticks(range(1, len(dict_residues['reslist']) + 1))
            if 'labellist' in dict_residues:
                ss_logo.ax.set_xticklabels(dict_residues['labellist'],
                                        fontsize=30)
            else:
                ss_logo.ax.set_xticklabels(dict_residues['reslist'],
                                        fontsize=20)
        if add_yticks:
            if mode == 'bits':
                ss_logo.ax.set_yticks([0.0, 1.0, 2.0, 3.0, 4.0])
                ss_logo.ax.set_yticklabels([0.0, 1.0, 2.0, 3.0, 4.0], fontsize=18)
            if mode == 'prob':
                ss_logo.ax.set_yticks([0.0, 0.25, 0.5, 0.75, 1.0])
                ss_logo.ax.set_yticklabels([0.0, 0.25, 0.5, 0.75, 1.0],
                                        fontsize=22)
        else:
            ss_logo.ax.set_yticks([])
            ss_logo.ax.set_yticklabels([])
        ss_logo.ax.yaxis.set_ticks_position('none')
        ss_logo.ax.yaxis.set_tick_params(pad=-1)
        ss_logo.ax.set_title(df_labels[i], fontsize=30)
        ss_logo.ax.spines['left'].set_visible(False)
        ss_logo.ax.spines['right'].set_visible(False)
        ss_logo.ax.spines['top'].set_visible(False)
        ss_logo.ax.spines['bottom'].set_visible(False)

    fig.tight_layout()
    fig.savefig(outfile, dpi=600, transparent=True)

    if show:
        plt.show()
    plt.close()


def makelogo_with_reference_and_perplexity(df_logos, df_labels, df_perplexity,
                                            wt_seq='', dict_residues={ 'reslist':[], 'labellist':[]},
                                            outfile='templogo_with_ref_perp.png', 
                                            show=False, mode='prob',\
                                            add_xticks=True, add_yticks=True,
                                            palette=sns.color_palette("colorblind")):
    N = len(df_logos[0])
    fig, axes = plt.subplots(1, len(df_logos) + 1, figsize=((N)*0.65*(len(df_logos)+1),1.5*3))
    for i in range(len(df_logos)):
        ss_logo = logomaker.Logo(df_logos[i],
                                font_name='Stencil Std',
                                color_scheme='NajafabadiEtAl2017',
                                vpad=.1,
                                width=.8,
                                ax=axes[i])
        if wt_seq != '':
            ss_logo.style_glyphs_in_sequence(sequence=wt_seq, color='darkgrey')
        if add_xticks:
            ss_logo.style_xticks(rotation=45)
            ss_logo.ax.set_xticks(range(1, len(dict_residues['reslist']) + 1))
            if 'labellist' in dict_residues:
                ss_logo.ax.set_xticklabels(dict_residues['labellist'],
                                        fontsize=30)
            else:
                ss_logo.ax.set_xticklabels(dict_residues['reslist'],
                                        fontsize=20)
        if add_yticks:
            if mode == 'bits':
                ss_logo.ax.set_yticks([0.0, 1.0, 2.0, 3.0, 4.0])
                ss_logo.ax.set_yticklabels([0.0, 1.0, 2.0, 3.0, 4.0], fontsize=18)
            if mode == 'prob':
                ss_logo.ax.set_yticks([0.0, 0.25, 0.5, 0.75, 1.0])
                ss_logo.ax.set_yticklabels([0.0, 0.25, 0.5, 0.75, 1.0],
                                        fontsize=22)
        else:
            ss_logo.ax.set_yticks([])
            ss_logo.ax.set_yticklabels([])
        ss_logo.ax.yaxis.set_ticks_position('none')
        ss_logo.ax.yaxis.set_tick_params(pad=-1)
        ss_logo.ax.set_title(df_labels[i], fontsize=30)
        ss_logo.ax.spines['left'].set_visible(False)
        ss_logo.ax.spines['right'].set_visible(False)
        ss_logo.ax.spines['top'].set_visible(False)
        ss_logo.ax.spines['bottom'].set_visible(False)

    ax = sns.barplot(data=df_perplexity, x='Positions', y='Perplexity', hue='Distribution',
                palette=palette, edgecolor=".2", linewidth=1.5, ax=axes[-1])
    h, l = ax.get_legend_handles_labels()
    ax.legend(h, l, title="", frameon=False, fontsize=20)
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.set_title("Perplexity", fontsize=30)
    ax.set_ylim([0, 20])
    ax.set_yticks([t for t in range(0, 24, 4)])
    ax.set_yticklabels([t for t in range(0, 24, 4)])
    ax.set_xticklabels(dict_residues['labellist'], fontsize=30, rotation=45)
    fig.tight_layout()
    fig.savefig(outfile, dpi=600, transparent=True)

    if show:
        plt.show()
    plt.close()


#Visualization
def makelogo_stack(df_logo_list, dict_residues_list=[{}], outfile='templogo.png', show=False, mode='prob',\
            add_xticks=True, add_yticks=True, text=''):
    #plt.style.use('classic')
    fig, axes = plt.subplots(len(df_logo_list), 1)
    for i, df_logo in enumerate(df_logo_list):
        ss_logo = logomaker.Logo(df_logo,
                                 font_name='Stencil Std',
                                 color_scheme='NajafabadiEtAl2017',
                                 width=.8,
                                 ax=axes[i])
        #ss_logo = logomaker.Logo(df_logo)
        dict_residues = dict_residues_list[i]
        if add_xticks:
            ss_logo.ax.set_xticks(
                range(1,
                      len(dict_residues['reslist']) + 1, 1))
            if 'labellist' in dict_residues:
                labels = [
                    dict_residues['labellist'][t]
                    for t in range(0, len(dict_residues['labellist']), 1)
                ]
                ss_logo.ax.set_xticklabels(labels, fontsize=5, Rotation=45)
            else:
                labels = [
                    dict_residues['reslist'][t]
                    for t in range(0, len(dict_residues['reslist']), 1)
                ]
                ss_logo.ax.set_xticklabels(labels, fontsize=5, Rotation=45)
        if add_yticks:
            if mode == 'bits':
                ss_logo.ax.set_yticks([0.0, 1.0, 2.0, 3.0, 4.0])
                ss_logo.ax.set_yticklabels([0.0, 1.0, 2.0, 3.0, 4.0],
                                           fontsize=5)
            if mode == 'prob':
                ss_logo.ax.set_yticks([0.0, 0.25, 0.5, 0.75, 1.0])
                ss_logo.ax.set_yticklabels([0.0, 0.25, 0.5, 0.75, 1.0],
                                           fontsize=5)
        #axes[1].box(False)
        if text != '' and i == 0:
            # used for iteration
            ss_logo.ax.text(-1.0, 0.95, text)
    fig.tight_layout()
    plt.savefig(outfile, transparent=True, dpi=800)

    if show:
        plt.show()
    plt.close()


def setup_and_make_stack(seqs_prob,
                         dict_residues,
                         outfile_logo,
                         add_yticks=True,
                         text=''):
    df_logos = []
    residues_subsets = []
    for i in range(0, len(seqs_prob), 50):
        subset = seqs_prob[i:min(i + 50, len(seqs_prob)), :]
        dict_residues_subset = {}
        for key in dict_residues:
            #print(key, dict_residues[key][i:min(i + 50, len(seqs_prob))])
            dict_residues_subset[key] = dict_residues[key][
                i:min(i + 50, len(seqs_prob))]

        df_logo = pd.DataFrame(data=subset,
                               index=range(1, subset.shape[0] + 1),
                               columns=columns_logo_aa)
        df_logos.append(df_logo)
        residues_subsets.append(dict_residues_subset)
    makelogo_stack(df_logos,
                   residues_subsets,
                   outfile=outfile_logo,
                   mode='prob',
                   add_yticks=add_yticks,
                   text=text)


def setup_and_make_stack_df(df,
                            dict_residues,
                            outfile_logo,
                            add_yticks=True,
                            text=''):
    df_logos = []
    residues_subsets = []
    for i in range(0, len(df), 50):
        dict_residues_subset = {}
        for key in dict_residues:
            #print(key, dict_residues[key][i:min(i + 50, len(df))])
            dict_residues_subset[key] = dict_residues[key][
                i:min(i + 50, len(df))]

        df_logo = df[i:min(i + 50, len(df))]
        # renumber from 1 to len
        df_logo = df_logo.reset_index()
        df_logo = df_logo.drop(columns=['index'])
        df_logos.append(df_logo)
        residues_subsets.append(dict_residues_subset)
    makelogo_stack(df_logos,
                   residues_subsets,
                   outfile=outfile_logo,
                   mode='prob',
                   add_yticks=add_yticks,
                   text=text)


def makelogomovie(df_logo_list, dict_residues, outfile='templogo.gif', show=False,\
                  add_xticks=True, loop=1, fps=2):
    #TODO: Feel free to try other movie formats; gif seemed to be the easiest
    # See makelogomovie2 for ffmpeg movie

    images = []
    for i, df_logo in enumerate(df_logo_list):
        if len(df_logo) > 50:
            setup_and_make_stack_df(df_logo,
                                    dict_residues,
                                    '.tmp_{}.png'.format(i),
                                    text='iteration = {}'.format(i))
        else:
            makelogo(df_logo,
                     dict_residues=dict_residues,
                     outfile='.tmp_{}.png'.format(i),
                     add_xticks=add_xticks,
                     text='iteration = {}'.format(i))
        images.append(imageio.imread('.tmp_{}.png'.format(i)))

    imageio.mimsave(outfile, images, loop=loop, fps=fps)
    os.system('rm .tmp_*.png')


def calculate_positional_entropy(sequences: list):
    positional_prob = sequences_to_probabilities(sequences, mode='prob')
    positional_entropy = np.zeros((positional_prob.shape[0]))
    for i in range(positional_prob.shape[0]):
        positional_entropy[i] = scipy_entropy(positional_prob[i, :], base=20)
    return positional_entropy


def sequences_to_probabilities(sequences: list, mode: str = 'prob'):
    '''
    sequences: N strings of length shape L
    '''
    seqs_num = np.array([letter_to_num(seq, _aa_dict) for seq in sequences])
    seqs_oh = F.one_hot(torch.tensor(seqs_num).long(),
                        num_classes=20).permute(1, 2, 0)

    if mode == 'prob':
        seq_oh_avg = seqs_oh.sum(dim=-1) / float(seqs_oh.shape[-1])

        return np.array(seq_oh_avg)
    else:
        raise KeyError('mode {} not available'.format(mode))


def arrays_to_logo(seqs_prob,
                   dict_residues,
                   outfile_logo='tmp_logo.png',
                   mode='prob',
                   add_yticks=True,
                   ref_seq=''):

    df_logo = pd.DataFrame(data=seqs_prob,
                           index=range(1, seqs_prob.shape[0] + 1),
                           columns=columns_logo_aa)

    makelogo(df_logo,
             dict_residues,
             outfile=outfile_logo,
             mode='prob',
             add_yticks=add_yticks)


def get_annotations(ref_seq, indices, reorder=False):
    if not reorder:
        ref_seq_oh = F.one_hot(torch.tensor(letter_to_num(ref_seq, _aa_dict)),
                               num_classes=20)
    else:
        ref_seq_oh = F.one_hot(torch.tensor(
            letter_to_num(ref_seq, aa_dict_reordered)),
                               num_classes=20)
    ref_seq_sel = ref_seq_oh[indices, :].numpy()
    ref_seq_str_array = np.array([t for t in ref_seq])[indices]
    ref_seq_str_array = np.expand_dims(ref_seq_str_array, axis=1)
    ref_seq_labels = np.broadcast_to(ref_seq_str_array,
                                     ref_seq_sel.shape).astype(str)
    ref_seq_labels[ref_seq_sel == 0] = ""
    return np.transpose(ref_seq_labels)


def array_to_heatmap(mat,
                      dict_residues,
                      outfile='tmp_heatmap.png',
                      mode='prob',
                      add_yticks=True,
                      ref_seq='',
                      vmin=None,
                      vmax=None,
                      center=0.0):

    mat = np.transpose(mat)
    df_logo = pd.DataFrame(data=mat,
                           index=columns_logo_aa,
                           columns=dict_residues['labellist'])

    annotations = None
    if (ref_seq != ''):
        annotations = get_annotations(ref_seq, dict_residues['reslist'])

    print(df_logo)

    if vmin is not None:
        sns.heatmap(df_logo,
                    cmap='vlag_r',
                    vmin=vmin,
                    vmax=vmax,
                    center=center,
                    annot=annotations,
                    fmt="s")
    else:
        sns.heatmap(df_logo,
                    cmap='vlag_r',
                    center=center,
                    annot=annotations,
                    fmt="s")
    plt.tight_layout()
    plt.savefig(outfile, dpi=600)
    plt.close()


def arrays_to_heatmap(mat,
                      dict_residues,
                      outfile='tmp_heatmap.png',
                      mode='prob',
                      add_yticks=True,
                      ref_seq='',
                      vmin=None,
                      vmax=None,
                      center=0.0):

    if isinstance(mat, str):
        # load from file
        mat_full = np.load(mat, allow_pickle=True)
    else:
        mat_full = np.array(mat)

    mat_select = np.transpose(mat_full[dict_residues['reslist'], :])
    df_logo = pd.DataFrame(data=mat_select,
                           index=columns_logo_aa,
                           columns=dict_residues['labellist'])

    annotations = None
    if (ref_seq != ''):
        annotations = get_annotations(ref_seq, dict_residues['reslist'])

    if vmin is not None:
        sns.heatmap(df_logo,
                    cmap='vlag_r',
                    vmin=vmin,
                    vmax=vmax,
                    linewidths=0.3,
                    linecolor='black',
                    center=center,
                    annot=annotations,
                    fmt="s")
    else:
        sns.heatmap(df_logo,
                    cmap='vlag_r',
                    center=center,
                    linewidths=0.3,
                    linecolor='black',
                    annot=annotations,
                    fmt="s")
    plt.tight_layout()
    plt.savefig(outfile, dpi=600)
    plt.close()


def arrays_to_aggregate_mat(mats,
                            dict_residues,
                            outfile='tmp_heatmap.png',
                            mode='prob',
                            add_yticks=True,
                            ref_seq='',
                            threshold=0.06,
                            vmin=None,
                            vmax=None,
                            center=0.0):
    array_list = []
    for mat in mats:
        if isinstance(mat, str):
            # load from file
            mat_full = np.load(mat, allow_pickle=True)
        else:
            # np.array type array
            mat_full = mat
        array_list.append(mat_full)

    comb_array = np.stack(array_list)
    agg_array = np.mean(comb_array, axis=0)
    #agg_array[agg_array<=0.051] = 0

    mat_select = np.transpose(agg_array[dict_residues['reslist'], :])
    annotations = None
    if (ref_seq != ''):
        annotations = get_annotations(ref_seq, dict_residues['reslist'], reorder=True)

    df_logo = pd.DataFrame(data=mat_select,
                           index=columns_logo_aa,
                           columns=dict_residues['labellist'])
    df_logo.index = pd.CategoricalIndex(df_logo.index,
                                          categories=aalist_ordered)
    df_logo.sort_index(level=0, inplace=True)
    if vmin is not None:
        ax = sns.heatmap(df_logo,
                    cmap='vlag_r',
                    vmin=vmin,
                    vmax=vmax,
                    center=center,
                    annot=annotations,
                    fmt="s")
    else:
        ax = sns.heatmap(df_logo,
                    cmap='vlag_r',
                    center=center,
                    annot=annotations,
                    fmt="s")
    ax.set_yticks([t for t in range(len(aalist_ordered))])
    ax.set_yticklabels(aalist_ordered, fontsize=10, Rotation=45)
    ax.set_xticks([t for t in range(len(dict_residues['labellist']))])
    ax.set_xticklabels(dict_residues['labellist'], fontsize=12)
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
         rotation_mode="anchor")
    plt.tight_layout()
    plt.savefig(outfile, dpi=600)
    plt.close()


def arrays_to_heatmap_movie(mat_files,
                            dict_residues,
                            outfile='tmp_heatmap_mov.gif',
                            mode='prob',
                            add_yticks=True,
                            ref_seq='',
                            vmin=None,
                            vmax=None):
    images = []
    for i, mat_file in enumerate(mat_files):
        mat_full = np.load(mat_file, allow_pickle=True)

        mat_select = np.transpose(mat_full[dict_residues['reslist'], :])
        df_logo = pd.DataFrame(data=mat_select,
                           index=columns_logo_aa,
                           columns=dict_residues['labellist'])
        if vmin is not None:
            sns.heatmap(df_logo, cmap='vlag_r', vmin=vmin, vmax=vmax)
        else:
            sns.heatmap(df_logo, cmap='vlag_r')
        plt.savefig('.tmp_frame_{}.png'.format(i), transparent=True, dpi=600)
        plt.close()
        images.append(imageio.imread('.tmp_frame_{}.png'.format(i)))

    imageio.mimsave(outfile, images, loop=1, fps=2)
    os.system('rm .tmp_frame_*.png')


def sequences_to_logo_without_weblogo(sequences,
                                      dict_residues={
                                          'reslist': [],
                                          'labellist': []
                                      },
                                      outfile_logo='tmp_logo.png',
                                      mode='prob',
                                      add_yticks=False,
                                      ref_seq='',
                                      text='',
                                      outfile_logo_ref='tmp_ref_logo.png',
                                      transform_to='',
                                      transform_from=''
                                      ):
    seqs_prob = sequences_to_probabilities(sequences, mode=mode)

    if seqs_prob.shape[0] > 50:
        setup_and_make_stack(seqs_prob, dict_residues, outfile_logo=outfile_logo, add_yticks=add_yticks, text=text)

    else:
        df_logo = pd.DataFrame(data=seqs_prob,
                           index=range(1, seqs_prob.shape[0] + 1),
                           columns=columns_logo_aa)
        makelogo(df_logo,
             dict_residues,
             outfile=outfile_logo,
             mode='prob',
             add_yticks=add_yticks,
             text=text,
             transform_from=transform_from,
             transform_to=transform_to)
        if ref_seq != '':
            single_sequence_to_logo(ref_seq, dict_residues, outfile_logo_ref, mode,
                                    add_yticks=False,
                                    text=text)


def calc_probdistr_perplexity(df_prob=None, arr_prob=None):
    import math
    if arr_prob is None:
        arr_prob = df_prob.to_numpy()
    dist_entropy = np.zeros((arr_prob.shape[0]))
    for i in range(arr_prob.shape[0]):
        dist_entropy_per_aa = 0.0
        for j in range(arr_prob.shape[1]):
            if arr_prob[i, j] > 0.0:
                dist_entropy_per_aa += arr_prob[i, j] * np.log2(arr_prob[i, j])
                #print(dist_entropy_per_aa)
        dist_entropy[i] = -1.0 * dist_entropy_per_aa
    perplexity = np.power(2, dist_entropy)
    #print(perplexity)
    return perplexity


def plot_perplexity(df_combined, positions, outfile="perplexity.png"):
    
    import matplotlib
    theme = {'axes.grid': True,
            'grid.linestyle': '',
            'xtick.labelsize': 20,
            'ytick.labelsize': 20,
            "font.weight": 'regular',
            'xtick.color': 'black',
            'ytick.color': 'black',
            "axes.titlesize": 20,
            "axes.labelsize": 18
        }
    
    matplotlib.rcParams.update(theme)
    
    palette = {'Hallucination': 'cornflowerblue', 'PyIgClassify': 'gainsboro'}
    ax = sns.barplot(data=df_combined, x='Positions', y='Perplexity', hue='Distribution',
                palette=palette, edgecolor=".2", linewidth=2.5)
    ax.set_ylim([0, 20])
    ax.set_yticks([t for t in range(0, 24, 4)])
    ax.set_yticklabels([t for t in range(0, 24, 4)])
    ax.set_xticklabels(positions, rotation=45)
    plt.tight_layout()
    plt.savefig(outfile, dpi=600, transparent=True)
    plt.close()


def sequences_to_logo_with_ref(sequences,
                                ref_seq,
                                ref_pssm,
                                dict_residues={
                                    'reslist': [],
                                    'labellist': []
                                },
                                outfile_logo='tmp_logo.png',
                                mode='prob'):

    seqs_prob = sequences_to_probabilities(sequences, mode=mode)

    df_logo = pd.DataFrame(data=seqs_prob,
                        index=range(1, seqs_prob.shape[0] + 1),
                        columns=columns_logo_aa)
    
    df_ref_pssm = pd.DataFrame(data=ref_pssm, index=range(1, seqs_prob.shape[0] + 1),
                        columns=columns_logo_aa)

    seq_len = len(ref_seq)
    array_ref = np.zeros((seq_len, 20))
    for i in range(array_ref.shape[0]):
        tmp = torch.tensor(np.array([letter_to_num(ref_seq[i],
                                                   _aa_dict)])).long()
        array_ref[i, :] = (F.one_hot(tmp, num_classes=20))
    df_logo_ref = pd.DataFrame(data=array_ref,
                           index=range(1, seq_len + 1),
                           columns=columns_logo_aa)

    df_logos = [df_logo, df_ref_pssm, df_logo_ref]
    df_labels = ['Hallucination', 'PyIgClassify', 'Wildtype']
    makelogo_with_reference(df_logos, df_labels, dict_residues=dict_residues,
                            outfile=outfile_logo, mode=mode, add_yticks=False)
    
    ref_perplexity = calc_probdistr_perplexity(df_ref_pssm)
    hal_perplexity = calc_probdistr_perplexity(df_logo)
    rand_perplexity = calc_probdistr_perplexity(arr_prob = np.full((seq_len, 20), 1.0/20.0))
    df_combined = pd.DataFrame()
    df_combined['Perplexity'] = list(hal_perplexity) + list(ref_perplexity)
    #+ list(rand_perplexity)
    df_combined['Distribution'] = ['Hallucination' for _ in range(hal_perplexity.shape[0])] + \
                                   ['PyIgClassify' for _ in range(ref_perplexity.shape[0])] 
    #['Uniform' for _ in range(rand_perplexity.shape[0]) ]
    positions = dict_residues['labellist']
    if dict_residues['labellist'] == []:
        positions = [t for t in range(hal_perplexity.shape[0])]
    df_combined['Positions'] = positions + positions
    outfile_perp = os.path.join(os.path.dirname(outfile_logo), 'perplexity.png')
    plot_perplexity(df_combined, positions, outfile=outfile_perp)
    outfile_logo_perp = os.path.join(os.path.dirname(outfile_logo), 'logo_with_ref_perplexity.png')
    palette = {'Hallucination':'cornflowerblue', "PyIgClassify": "peachpuff"}
    makelogo_with_reference_and_perplexity([df_logo, df_ref_pssm], 
                                           ['Hallucination', 'PyIgClassify'],
                                           df_combined, wt_seq=ref_seq,
                                           dict_residues=dict_residues,
                                           outfile=outfile_logo_perp, mode=mode,
                                           add_yticks=False,
                                           palette=palette)


def sequences_to_logo_with_ref_dict(sequences,
                                   ref_pssm_dict,
                                   dict_residues={
                                    'reslist': [],
                                    'labellist': []
                                   },
                                   outfile_logo='tmp_logo.png',
                                   mode='prob'):

    seqs_prob = sequences_to_probabilities(sequences, mode=mode)

    df_logo = pd.DataFrame(data=seqs_prob,
                        index=range(1, seqs_prob.shape[0] + 1),
                        columns=columns_logo_aa)
    df_logos = [df_logo]
    df_labels = ['Hallucination']

    for key in ref_pssm_dict:
        df_ref_pssm = pd.DataFrame(data=ref_pssm_dict[key], index=range(1, seqs_prob.shape[0] + 1),
                        columns=columns_logo_aa)
        df_logos.append(df_ref_pssm)
        df_labels.append(key)
    
    makelogo_with_reference(df_logos, df_labels, dict_residues=dict_residues,
                            outfile=outfile_logo, mode=mode, add_yticks=False)



def run_weblogo(seq_file, outfile='temp_weblogo', format_='png'):
    command = "weblogo -A 'protein' -F %s < %s > %s " % (format_, seq_file,
                                                         outfile)
    os.system(command)
    return outfile


def sequence_list_to_logo_movie(seqs_list,
                                dict_residues,
                                outfile_logo='logo_traj_{}.png',
                                mode='prob',
                                max_frames=50):

    for seqlist in seqs_list:
        df_logos = []
        textlist = []
        if len(seqlist) > 50:
            jump = int(len(seqlist) / 50)
            seqlist = [seqlist[t] for t in range(0, len(seqlist), jump)]
        for (traj, seq) in seqlist:
            seq_len = len(seq)
            # one hot encode seq:
            tmp = torch.tensor(np.array([letter_to_num(seq, _aa_dict)])).long()
            seq_array = np.array((F.one_hot(tmp, num_classes=20)).squeeze(0))

            df_logo = pd.DataFrame(data=seq_array,
                                   index=range(1, seq_len + 1),
                                   columns=columns_logo_aa)
            # print(df_logo)
            textlist.append(seq)
            df_logos.append(df_logo)
        traj = seqlist[0][0]
        makelogomovie(df_logos,
                      dict_residues,
                      outfile=outfile_logo.format(traj))


def single_sequence_to_logo(ref_seq,
                      dict_residues={},
                      outfile_logo='logo_ref.png',
                      mode='prob',
                      add_yticks=False,
                      text=''):
    seq_len = len(ref_seq)
    array_ref = np.zeros((seq_len, 20))
    for i in range(array_ref.shape[0]):
        tmp = torch.tensor(np.array([letter_to_num(ref_seq[i],
                                                   _aa_dict)])).long()
        array_ref[i, :] = (F.one_hot(tmp, num_classes=20))
    df_logo = pd.DataFrame(data=array_ref,
                           index=range(1, seq_len + 1),
                           columns=columns_logo_aa)
    makelogo(df_logo, dict_residues, outfile=outfile_logo, mode=mode,
            text=text)


def sequence_to_logo(seqfile,loop_length,dict_residues,outfile_logo='logo_prob.png',\
                    mode='prob',ref_seq='',outfile_logo_ref='logo_prob_ref.png'):
    logodatafile = run_weblogo(seqfile, format_='logodata')
    array_pos_by_resinfo = np.loadtxt(logodatafile,
                                      comments="#",
                                      delimiter="\t",
                                      unpack=False)
    array_freq = array_pos_by_resinfo[:, 1:21]
    total_freq = float(sum(array_freq[0, :]))

    array_prob = array_freq / total_freq

    df_logo = pd.DataFrame(data=array_prob,
                           index=range(1, loop_length + 1),
                           columns=columns_logo_aa)
    makelogo(df_logo, dict_residues, outfile=outfile_logo, mode='prob')

    if ref_seq != '':
        single_sequence_to_logo(ref_seq, dict_residues, outfile_logo_ref, mode)


def calculate_seq_distance(sequences, base, calc_lds=True):
    list_ld = []
    list_lds = []
    for seq in sequences:
        ld = textdistance.levenshtein(seq, base)
        list_ld.append(ld)
        if calc_lds:
            lds = textdistance.levenshtein.normalized_similarity(seq, base)
            list_lds.append(lds)
        
    return list_ld, list_lds

def calculate_sequence_matches(sequences, base):
    list_matches = []
    for seq in sequences:
        assert len(seq) == len(base)
        ld = textdistance.levenshtein(seq, base)
        list_matches.append(len(base) - ld)
    return list_matches


def fast_ld_calculation(seqs, ref_seqs):
    int_dseqs = np.array([letter_to_num(t, _aa_dict) for t in seqs])
    int_pseqs = np.array([letter_to_num(t, _aa_dict) for t in ref_seqs])
    overlap_pseqs_not = [np.logical_not(np.equal(int_pseqs, int_dseqs[i, :])).astype(int) \
            for i in range(int_dseqs.shape[0])]
    ld =[np.sum(t, axis=1) for t in overlap_pseqs_not]
    return ld

def get_sequence_matches_at_indices(seqs, ref_seqs, indices):
    int_dseqs = np.array([letter_to_num(t, _aa_dict) for t in seqs])[:, indices]
    int_pseqs = np.array([letter_to_num(t, _aa_dict) for t in ref_seqs])[:, indices]
    assert int_dseqs.shape[0] == len(seqs)
    assert int_dseqs.shape[1] == len(indices)
    overlap_pseqs = [(np.equal(int_pseqs, int_dseqs[i, :])).astype(int) \
            for i in range(int_dseqs.shape[0])]
    matches =[np.sum(t, axis=1) for t in overlap_pseqs]
    return matches


def calculate_sequence_recovery(sequences, base):
    matches_per_seq = calculate_sequence_matches(sequences, base)
    total_correct = sum(matches_per_seq)
    total_tested = len(matches_per_seq)*len(base)
    return total_correct, total_tested


def get_proteinparams_from_biopython(chain_sequences):
    from Bio.SeqUtils.ProtParam import ProteinAnalysis
    
    params_dict = {}
    params_dict['Charge at pH7'] = [ProteinAnalysis(seq).charge_at_pH(7) for seq in chain_sequences]
    params_dict['Instability Index'] = [ProteinAnalysis(seq).instability_index() for seq in chain_sequences]
    params_dict['Gravy'] = [ProteinAnalysis(seq).gravy() for seq in chain_sequences]
        
    return params_dict


def biopython_developability_dataframes(all_sequences_full,
                                        len_heavy,
                                        indices_hal,
                                        full_chain=True,
                                        pad_length=10,
                                        full_fv=False):
    all_chains = set(['H' if ind < len_heavy else 'L' for ind in indices_hal])
    print('Chains in design :', all_chains)
    assert len(all_chains) <=2
    developability_dict_chains = {}
    for chain in all_chains:
        if full_chain:
            all_sequences_full_chain = [t[:len_heavy] if chain=='H' else t[len_heavy:] for t in all_sequences_full]
            developability_dict = get_proteinparams_from_biopython(all_sequences_full_chain)
        elif full_fv:
            developability_dict = get_proteinparams_from_biopython(all_sequences_full)
        else:
            #this assumes indices are ordered in continuous eg. 2-10 etc.
            start, end = indices_hal[0]-pad_length, indices_hal[-1]+pad_length
            start = max(0, start)
            if chain=='H':
                end = min(len_heavy, end)
            else:
                end = min(end, len(all_sequences_full[0]))
            all_sequences_design = [t[start:end] for t in all_sequences_full]
            developability_dict = get_proteinparams_from_biopython(all_sequences_design)
        for key in developability_dict:
            if key in developability_dict_chains:
                developability_dict_chains[key] += developability_dict[key]
            else:
                developability_dict_chains[key] = developability_dict[key]
    
    developability_dict_chains['Seq'] = []
    developability_dict_chains['Chain'] = []
    for chain in all_chains:
        developability_dict_chains['Seq'] += [''.join([t[ind] for ind in indices_hal]) for t in all_sequences_full]
        developability_dict_chains['Chain'] += [chain for _ in range(len(all_sequences_full))]
    developability_dict_chains['Full_Chain'] = [full_chain for _ in range(len(developability_dict_chains['Seq']))] 
    developability_dict_chains['Full_Fv'] = [full_fv for _ in range(len(developability_dict_chains['Seq']))] 
    developability_dict_chains['Padding'] = [pad_length for _ in range(len(developability_dict_chains['Seq']))] 
    for key in developability_dict_chains:
        print(key, len(developability_dict_chains[key]))
    
    df_developability = pd.DataFrame.from_dict(developability_dict_chains)
    return df_developability


def write_and_plot_biopython_developability(all_sequences_full,
                                        len_heavy,
                                        indices_hal,
                                        wt_seq='',
                                        out_path='.'):
    
    df_developability = biopython_developability_dataframes(all_sequences_full, len_heavy, indices_hal)
    outfile = os.path.join(out_path, 'df_developability_biopython.csv')
    df_developability.to_csv(outfile, index=False)
    df_developability_Wt = biopython_developability_dataframes([wt_seq], len_heavy, indices_hal)
    outfile = os.path.join(out_path, 'df_developability_biopython_Wt.csv')
    df_developability_Wt.to_csv(outfile, index=False)

    import matplotlib
    theme = {'axes.grid': True,
            'grid.linestyle': '',
            'xtick.labelsize': 18,
            'ytick.labelsize': 18,
            "font.weight": 'regular',
            'xtick.color': 'black',
            'ytick.color': 'black',
            "axes.titlesize": 20,
            "axes.labelsize": 20
        }
    matplotlib.rcParams.update(theme)
    import seaborn as sns

    unique_chains = set(list(df_developability['Chain']))
    for chain in unique_chains:
        df_chain = df_developability[df_developability['Chain']==chain]
        df_chain_wt = df_developability_Wt[df_developability_Wt['Chain']==chain]
        for param in biopython_developability_keys:
            fig = plt.figure(figsize=(6,5))
            ax = plt.gca()
            ax = sns.histplot(data=df_chain, x=param, ax=ax, stat='probability', color='grey')
            print('Wt ', param, list(df_chain_wt[param])[0])
            ax.axvline(list(df_chain_wt[param])[0], ls='--', lw=2.0, c='black', zorder=1)
            plt.xticks(rotation=45)
            ax.set_xlabel(param)
            ax.set_ylabel('P({})'.format(param))
            plt.tight_layout()
            clean_param_name = param.replace(' ','')
            plt.savefig('{}/biodevelop_{}_dist_chain{}.png'.format(out_path, clean_param_name, chain), dpi=600, transparent=True)
            plt.close()
