import matplotlib.pyplot as plt
import numpy as np


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
