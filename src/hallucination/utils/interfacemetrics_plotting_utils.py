import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os, os.path, argparse, sys, glob
import pandas as pd
import numpy as np
import seaborn as sns

scatter_plot_cols = ["hbonds_int",
                    "dSASA_int",
                    "dSASA_hphobic",
                    "dSASA_polar",
                    "sc_value",
                    'total_score']
label_dict = {"hbonds_int": "# H-bonds",
               "dSASA_int": r"\{Delta}SASA Total ($\AA^{2}$)",
               "dSASA_hphobic": r"\{Delta}SASA HPhobic ($\AA^{2}$)",
               "dSASA_polar": r"\{Delta}SASA Polar ($\AA^{2}$)",
               "sc_value": "Shape Complementarity",
               "total_score": "Total Score (REU)",
               "dG_separated": r"\{Delta}G Binding (REU)"
               }

def parse_list_file(listfile):
    path = os.path.split(os.path.relpath(listfile))[0]
    f = open(listfile, 'r')
    pdbs = f.readlines()
    pdbs = [pdb.rstrip() for pdb in pdbs]
    pdbs = [os.path.join(path, pdb) for pdb in pdbs if (pdb.endswith('.pdb') and not pdb.startswith('_.'))]
    return pdbs


def iam_score_df_from_pdbs(list_of_pdbs, outfile=None):

    print(len(list_of_pdbs))
    score_dict ={}
    for pdb in list_of_pdbs:
        f = open(pdb, 'r')
        dat = f.readlines()
        pdb_name = os.path.split(pdb)[1].rstrip('.pdb')
        
        my_dict = {}
        for i in range(len(dat)):
            if dat[i].startswith('design'):
                dict_dat = dat[i+1:i+22]
                for item in dict_dat:
                    s = item.split(' ')
                    my_dict[s[0]]=s[1].rstrip()
            elif dat[i].startswith('pose'):
                pose_scores = dat[i].split()
                pose_score_terms = dat [i-2].split()
                for i in range(1, len(pose_score_terms)):
                    my_dict[pose_score_terms[i]] = pose_scores[i]
        score_dict[pdb_name] = my_dict
        
    df = pd.DataFrame.from_dict(score_dict)
    df = df.transpose()
    df = df.rename(columns={'post_relax_total_score': 'total_score'})
    df['filename'] = list_of_pdbs
    if not outfile is None:
        df.to_csv(outfile)
    return df


def scatter_hist(df, ref=None, out="scatterhist.png", highlight=[], by =["dG_separated"], main_term="dG_separated"):
    ''': \n
    OPTIONAL: x_percent to hightlight subgroups and lower and upper limits for y or x axis: e.g.ylim_lower = float \n
    '''
    score_terms_of_interest = scatter_plot_cols
    number_score_terms = len(score_terms_of_interest)
    cols = 3
    rows = 2
    
    figsize = (18, 16/cols*rows)

    fig, axes = plt.subplots(rows, cols, figsize=figsize)
    ax_flatten = axes.flatten()
    fig.subplots_adjust(hspace=0.4, wspace=0.4)
    plt.rcParams.update({'font.size': 15})

    df[main_term] = df[main_term].astype('float')
    if not ref is None:
        ref[main_term] = ref[main_term].astype('float')
        ref_val_y = list(ref[main_term])
    for i in range(0, number_score_terms):
        df[score_terms_of_interest[i]] = df[score_terms_of_interest[i]].astype('float')
        ax = ax_flatten[i]
        if not ref is None:
            if (score_terms_of_interest[i] in ref.keys()):
                ref[score_terms_of_interest[i]] = ref[score_terms_of_interest[i]].astype('float')
                ref_val = list(ref[score_terms_of_interest[i]])
                ax.axhline(min(ref_val), ls='--', lw=2.0, c='black', zorder=1)
            
                ax.axvline(min(ref_val_y), ls='--', lw=2.0, c='black', zorder=1)
            
        ax = sns.scatterplot(x=df[main_term],
                             y=df[score_terms_of_interest[i]],
                             alpha = 0.3, s=15, color='#4b4c4d',
                             ax = ax)
        if highlight != []:
            df_highlight = df.loc[highlight]
            ax.scatter(df_highlight[main_term],
                       df_highlight[score_terms_of_interest[i]],
                       s=15, color='tomato')
        ax.set_xlabel(label_dict[main_term], fontsize=18)
        ax.set_ylabel(label_dict[str(score_terms_of_interest[i])], fontsize=18)

    figure_name = os.path.join(os.path.dirname(out), 'all_scatter.png')
    if highlight != [] and out !='':
        figure_name = out
        plt.savefig(figure_name)
    plt.close()


def plot_distributions(df, ref="", out="dist_{}.png", highlight=[], by =["dG_separated"], main_term="dG_separated"):
    theme = {'axes.grid': True,
            'grid.linestyle': '',
            'xtick.labelsize': 18,
            'ytick.labelsize': 18,
            "font.weight": 'regular',
            'xtick.color': 'black',
            'ytick.color': 'black',
            "axes.titlesize": 20,
            "axes.labelsize": 18
        }
    import matplotlib
    matplotlib.rcParams.update(theme)
    score_terms_of_interest = scatter_plot_cols
    number_score_terms = len(score_terms_of_interest)
    cols = 3
    rows = 2
    
    figsize = (18, 16/cols*rows)

    for stat in ['probability', 'count']:
        fig, axes = plt.subplots(rows, cols, figsize=figsize)
        ax_flatten = axes.flatten()
        fig.subplots_adjust(hspace=0.4, wspace=0.4)
        plt.rcParams.update({'font.size': 15})

        for i in range(0, number_score_terms):
            df[score_terms_of_interest[i]] = df[score_terms_of_interest[i]].astype('float')
            ax = ax_flatten[i]
            if score_terms_of_interest[i] in ref.keys():
                ref[score_terms_of_interest[i]] = ref[score_terms_of_interest[i]].astype('float')
                ref_val = list(ref[score_terms_of_interest[i]])
                ax.axvline(min(ref_val), ls='--', lw=2.0, c='tomato', zorder=1)

            min_x = min(list(df[score_terms_of_interest[i]]) + ref_val)
            max_x = max(list(df[score_terms_of_interest[i]]) + ref_val)
            binrange = [min_x, max_x]
            sns.histplot(data=df, x=score_terms_of_interest[i], element='step',
                        fill=False, stat=stat,
                        lw=3,
                        color='darkblue',
                        ax=ax,
                        binrange=binrange)
            
            ax.set_xlabel(label_dict[score_terms_of_interest[i]])
            if stat == 'probability':
                ax.set_ylabel('Fraction of Designs')
            else:
                ax.set_ylabel('# of Designs')

        figure_name = os.path.join(os.path.dirname(out), 'all_dist_{}.png'.format(stat))
        if highlight != [] and out !='':
            figure_name = out.format(stat)
        print("Figure name: ", figure_name)
        plt.savefig(figure_name)
        plt.close()


def get_dict_mean_sd(df):
    dict_mean_sd = {}
    for i in df.columns:
        try:
            df[i] = df[i].astype('float16')
            dict_mean_sd[i] = (np.mean(df[i]), np.std(df[i]))
        except TypeError:
            dict_mean_sd[i] = ('NaN', 'NaN')
        except ValueError:
            dict_mean_sd[i] = ('NaN', 'NaN')
    return(dict_mean_sd)


def get_zscore_df(df, high_score_term):
    dict_mean_sd = get_dict_mean_sd(df)
    print(dict_mean_sd)
    zscore_df = pd.DataFrame()
    for col in df.columns:
        print(col)
        (mean, sd) = dict_mean_sd[col]
        print(mean, sd)
        try:
            if col in high_score_term:
                zscore_df[col] = ((df[col] - mean)/sd) * -1
            else:
                zscore_df[col] = (df[col] - mean)/sd
        except TypeError:
            zscore_df[col] ='NaN'
    return(zscore_df)


def select_best_designs_by_sum(df, by, 
                                high_score_term=["hbonds_int",
                                                  "dSASA_int",
                                                  "dSASA_hphobic",
                                                  "dSASA_polar",
                                                  "sc_value",
                                                  "packstat"], 
                                n=1, 
                                extra_weight=[],
                                pdb_dir="",
                                out_path=''):
    '''
    n is the number of designs I want to select
    '''
    print("Selecting the best designs")
    df['dG_separated'] = df['dG_separated'].astype('float16')
    df_filtered = df[df['dG_separated'] < 0]
    if len(df_filtered['dG_separated']) < 10:
        sys.exit('Less than models with negative dG_separated')
    zscore_df = get_zscore_df(df_filtered, high_score_term)
    outfile = os.path.join(out_path, 'zscores.csv')
    zscore_df.to_csv(outfile)

    print("Sorting by: ", (', ').join(by))
    zscore_df['sum'] = zscore_df[by[0]]
    for i in range(1, len(by)):
        zscore_df['sum'] += zscore_df[by[i]]
    # print('vorher: ', zscore_df['sum'])
    for term in extra_weight:
        try:
            zscore_df['sum'] += zscore_df[term]
        except KeyError:
            print('WARNING: the extra weight for the following score term could not be added because it does not exist: ', term)

    sorted_df = zscore_df.sort_values(by=['sum'], ascending=True)
    top_decoy_description = sorted_df.head(n).index.tolist()
    sub_df = df.loc[top_decoy_description]
    selection_name = ('_').join(by)
    out_dir = os.path.join(out_path, "top_%s_by_%s" %(n, selection_name))
    os.makedirs(out_dir, exist_ok=True)
    sub_df.to_csv(os.path.join(out_dir, "top_" + str(n) + '_by_' + selection_name + '_clean.csv'))

    if pdb_dir != "":
        os.makedirs(pdb_dir, exist_ok=True)
        top_decoy_filenames = list(sub_df['filename'])
        for filename in top_decoy_filenames:
            os.system('cp %s %s/' %(filename, pdb_dir))

    return(top_decoy_description)


def plot_scores_and_select_designs(df_mutants, df_ref, out_path=".", pdb_dir=".", by=['dG_separated'], n=20, extra_weight=''):
    
    best_decoys = select_best_designs_by_sum(df_mutants, by=by, \
        n=n, extra_weight=extra_weight, pdb_dir=pdb_dir, out_path=out_path)

    print(df_mutants.shape)
    print('\n---------------------\n')
    print('Best decoys by: %s\n'%by)
    for item in best_decoys:
        print(item)
    print('\n---------------------\n')

    df_mutants_neg = df_mutants[df_mutants['dG_separated'] < -10.0]
    selection_name = ('_').join(by)
    out_pdf = os.path.join(out_path, "top_%s_by_%s/multi_score_histplot_{}.png" %(n, selection_name))
    plot_distributions(df_mutants, ref=df_ref, out=out_pdf, highlight=best_decoys, by=by)
    out_pdf = os.path.join(out_path, "top_%s_by_%s/multi_score_histplot_dgneg_{}.png" %(n, selection_name))
    plot_distributions(df_mutants_neg, ref=df_ref, out=out_pdf, highlight=best_decoys, by=by)
    out_pdf = os.path.join(out_path, "top_%s_by_%s/multi_score_scatterplot.png" %(n, selection_name))
    scatter_hist(df_mutants, ref=df_ref, out=out_pdf, highlight=best_decoys, by=by)
    out_pdf = os.path.join(out_path, "top_%s_by_%s/multi_score_scatterplot_dgneg.png" %(n, selection_name))
    scatter_hist(df_mutants_neg, ref=df_ref, out=out_pdf, highlight=best_decoys, by=by)


def parse_args():
    '''Parse command line args. 
    '''
    parser=argparse.ArgumentParser()
    parser.add_argument("--mutants_pdb_path", \
                help="Provide path containing relaxed mutants", \
                default='')
    parser.add_argument('--ref_pdb_path', \
                default='', \
                help='Provide path containing pdb files for the reference, i.e. relaxed WT')
    parser.add_argument("--mutants_pdb_list", \
                help="Provide Rosetta style list file containing relaxed mutants", \
                default='')
    parser.add_argument('--ref_pdb_list', \
                default='', \
                help='Provide Rosetta style list file containing pdb files for the reference, i.e. relaxed WT')
    parser.add_argument("-high_score_term", \
                nargs="*", \
                dest="high_score_term", \
                help="List any score_terms where a high number is better than a low number, e.g. \'hbonds_int, other_term\'", \
                default=["hbonds_int", "dSASA_int", "dSASA_hphobic", "dSASA_polar", "sc_value", "packstat"])
    parser.add_argument("-n",
                dest="n",
                help="Number of top decoys to sort out",
                type=int,
                default=25)
    parser.add_argument("-by", \
                dest="by", \
                nargs="*",\
                default=["dG_separated"],
                help='What score terms to combine into sum. These will be highlighted in the scatter plot as well.')
    parser.add_argument("-extra_weight", \
                nargs="*", \
                default=[], \
                dest="extra_weight", \
                help="Provide score terms that should carry extra weight in the calculation")
    parser.add_argument("-pdb_dir", \
                dest="pdb_dir", \
                help="Directory where the analyzed pdbs are currently stored.", \
                default='out_pdb')
    parser.add_argument("-out_dir", \
                dest="out_dir", \
                help="Directory where the output pdb files are stored.", \
                default='interface_metrics')
    return parser.parse_args()


if __name__=="__main__":
    args = parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    mutants_score_file = os.path.join(args.out_dir, 'mutants_interface_metrics.csv')
    if args.mutants_pdb_path != '':
        design_pdbs = list(sorted(glob.glob(args.mutants_pdb_path + '')))
        df_mutants = iam_score_df_from_pdbs(design_pdbs, mutants_score_file)
    elif args.mutants_pdb_list != '':
        design_pdbs = parse_list_file(args.mutants_pdb_list)
        df_mutants = iam_score_df_from_pdbs(design_pdbs, mutants_score_file)
    else:
        sys.exit('Provide either: --mutants_pdb_path or --mutants_pdb_list')

    ref_score_file = os.path.join(args.out_dir, 'ref_interface_metrics.csv')
    if args.ref_pdb_path != '':
        ref_pdbs = list(sorted(glob.glob(args.ref_pdb_path + '*.pdb')))
        df_ref = iam_score_df_from_pdbs(ref_pdbs, ref_score_file)
    elif args.ref_pdb_list != '':
        ref_pdbs = parse_list_file(args.ref_pdb_list)
        df_ref = iam_score_df_from_pdbs(ref_pdbs, ref_score_file)
    else:
        sys.exit('Provide either: --ref_pdb_path or --ref_pdb_list')

    plot_scores_and_select_designs(df_mutants, df_ref, out_path=args.out_dir,
                                    pdb_dir=args.pdb_dir, by=args.by, n=args.n,
                                    extra_weight=args.extra_weight)


