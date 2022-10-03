
import argparse

def _get_args():
    """Gets command line arguments"""
    desc = ('''
        Desc pending
        ''')
    parser = argparse.ArgumentParser(description=desc)
    # Model architecture arguments
    parser.add_argument('--iterations',
                        type=int,
                        default=50,
                        help='Number of iterations')
    parser.add_argument('--n_every',
                        type=int,
                        default=100,
                        help='Write intermediate output every')
    parser.add_argument('--seed',
                        type=int,
                        default=111,
                        help='torch manual seed')
    parser.add_argument('--random_seed',
                        action='store_true',
                        default=False,
                        help='use random seed')
    parser.add_argument('--prefix',
                        type=str,
                        default='test',
                        help='basename for output')
    parser.add_argument('--suffix',
                        type=str,
                        default='suf',
                        help='suffix for output files')
    parser.add_argument('--target_pdb',
                        type=str,
                        default='',
                        help='path to target structure')
    parser.add_argument('--cdr_list',
                        type=str,
                        default='',
                        help='comma separated list of cdrs: l1,h2')
    parser.add_argument('--framework',
                        action='store_true',
                        default=False,
                        help='design framework residues. Default: false')
    parser.add_argument(
        '--indices',
        type=str,
        default='',
        help=
        'comma separated list of chothia numbered residues to design: h:12,20,31A/l:56,57'
    )
    parser.add_argument(
        '--exclude',
        type=str,
        default='',
        help=
        'comma separated list of chothia numbered residues to exclude from design: h:31A,52,53/l:97,99'
    )
    parser.add_argument('--hl_interface',
                        action='store_true',
                        default=False,
                        help='Not implemented! hallucinate hl interface')
    parser.add_argument(
        '--use_ensemble',
        action='store_true',
        default=False,
        help='use averaged output from multiple trained models')
    parser.add_argument('--geometric_loss_weight',
                        type=float,
                        default=1.0,
                        help='Weight for geometric component of loss')
    parser.add_argument(
        '--seq_loss_weight',
        type=float,
        default=0.0,
        help='Weight for sequence (non-design) component of loss')
    parser.add_argument('--restricted_positions_kl_loss_weight',
                        type=float,
                        default=10.0,
                        help='Weight for kl divergence(component of loss)\
                             from restricted positional frequencies as\
                             specified by --restrict_positions_to_freq.\
                             This loss component requires \
                             --restrict_positions_to_freq to be set.')
    parser.add_argument(
        '--restrict_total_charge',
        action='store_true',
        default=False,
        help='Constrain max charge of designed residues to -+2')
    parser.add_argument(
        '--restrict_max_aa_freq',
        action='store_true',
        default=False,
        help='Constrain max number of AA of same type to max of 4.')
    parser.add_argument('--geometric_loss_list',
                        type=str,
                        default='1,1,1,1,1',
                        help='Specific loss weights for distance, orientation etc.\
                            First 3 for distances; remaining for dihedral and planar geometries.\
                            See DeepAb methods sections for more details.')
    parser.add_argument('--restrict_positions_to_freq',
                        type=str,
                        default='',
                        help='restrict prob/freq at these positions: \
                            h:100A-W=0.33-Y=0.33-F=0.33,100B-D=0.50,E=0.50/\
                            l:99-D=0.10,E=0.90. Frequencies will be normalized.\
                            Unlisted AA frequencies will be set to zero.')
    parser.add_argument('--restrict_positions_to_aas',
                        type=str,
                        default='',
                        help='Only allow specified AAs at these positions: \
                            h:100A-WYF/\
                            l:99-DE.')
    parser.add_argument('--restrict_positions_to_aas_except',
                        type=str,
                        default='',
                        help='Disallow specified AAs at these positions: \
                            Eg. h:100A-CP/\
                            l:99-CP. \
                            Disallows Cs at l88 and l89 (chothia numbered).\
                            Complement of --restrict_positions_to_aas. '
                        )
    parser.add_argument('--disallow_aas_at_all_positions',
                        type=str,
                        default='',
                        help='Disallow specified AAs at all design positions: \
                            eg. CP. Recommended to disallow C at all positions \
                            when designing CDR loops.')
    parser.add_argument(
        '--avg_seq_reg_loss_weight',
        type=float,
        default=0.0,
        help='loss weight for applying sequence regularization.\
                Use average AA frequencies derived from SAbDAb Antibodies for H and L chain\
                to restrict average AA frequencies sampled.\
                Beware: if your antibody heavy/light is missing an AA; \
                this loss will lead to overrepresentation of the missing AA in the designs.')
    parser.add_argument('--overwrite',
                        action='store_true',
                        default=False,
                        help='overwrite existing dirs')
    parser.add_argument('--disable_autostop',
                        default=False,
                        action='store_true',
                        help=
                        'Default:Automatically stop trajectory if all losses are equilibrated. \
                        Use this option for disabling autostop. May be useful if \
                        hallucinating large regions (full Fv). \
                        Application has not been optimized for this mode.')
    parser.add_argument('--seed_with_WT',
                        default=False,
                        action='store_true',
                        help='Seed with WT (Wildtype seeding in published work) at initialization.')
    parser.add_argument('--disable_lr_scheduler',
                        action='store_true',
                        default=False,
                        help='Learning rate scheduler is applied by default. \
                              See default settings --lr settings. Disable with this option.\
                              May be useful for determining the right lr settings for your system.\
                              Default settings usually work for cases described in the work.')
    parser.add_argument('--lr_settings',
                        type=str,
                        default='0.05,30,10',
                        help='learning rate, patience for LR Plateau scheduler, cooldown for LR Plateau scheduler')
    parser.add_argument('--apply_distribution_from_pssm',
                        type=str,
                        default='',
                        help='read numpy array from a pssm. \
                            Specify sequence restrictions for all\
                            positions as a numpy array.')
    parser.add_argument('--use_global_loss',
                        action='store_true',
                        default=False,
                        help='Geometric losses are calculated\
                             within a distance cutoff of 10 angstrom by default. This option\
                             disables that. Expect more fluctuation in losses. Check loss convergence.\
                             Adjust number of --iterations, --lr_settings to get better convergence.')
    parser.add_argument('--disable_nondesign_mask',
                        action='store_true',
                        default=False,
                        help='disable nondesign 2d mask; ie calculate loss over full structure;'
                                'Not recommended.')
    

    return parser.parse_args()


