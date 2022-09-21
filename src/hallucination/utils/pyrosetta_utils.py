from src.deepab.build_fv.build_cen_fa import get_fa_min_mover
from pyrosetta import *
from dask.distributed import get_client

import pyrosetta
import pyrosetta.distributed.dask
import pyrosetta.rosetta.protocols.simple_moves as psm
from pyrosetta.rosetta.protocols.docking import *
from src.util.util import _aa_1_3_dict as aa_1to3_map


def heavy_bb_rmsd_from_atom_map(pose_1, pose_2, residue_list):
    assert pose_1.size() == pose_2.size()
    atom_id_map = pyrosetta.rosetta.std.map_core_id_AtomID_core_id_AtomID()
    pyrosetta.rosetta.core.scoring.setup_matching_protein_backbone_heavy_atoms(pose_1, pose_2, atom_id_map)
    residues_mask_vector = pyrosetta.rosetta.utility.vector1_bool()
    residues_mask_vector.extend([True if i in residue_list else False for i in range(pose_1.size())])
    rmsd_vector = pyrosetta.rosetta.core.scoring.per_res_rms_at_corresponding_atoms_no_super(pose_1, pose_2, atom_id_map, residues_mask_vector)
    return rmsd_vector

def show_selector(selector, pose, name='selection '):
    metric = pyrosetta.rosetta.core.simple_metrics.metrics.SelectedResiduesMetric(
    )
    metric.set_residue_selector(selector)
    output = metric.calculate(pose)
    print('{}:\n{}\n'.format(name, output))


def get_complex_pose_from_antibody_pose(ab_pose, complex_pose, ag_chain_id, debug=False):

    complex_ab_pose = complex_pose.clone()
    complex_ag_pose = complex_pose.clone()

    ag_chain_int = pyrosetta.rosetta.core.pose.get_chain_id_from_chain(ag_chain_id, complex_pose)
    ag_begin = complex_pose.chain_begin(ag_chain_int)
    ag_end = complex_pose.chain_end(ag_chain_int)
    # remove antigen
    complex_ab_pose.delete_residue_range_slow(ag_begin, ag_end)

    assert complex_ab_pose.num_chains() == complex_pose.num_chains() - 1
    assert complex_ab_pose.size() == ab_pose.size()
    #superpose antibody pose to antibody pose from complex
    pyrosetta.rosetta.core.scoring.calpha_superimpose_pose(ab_pose, complex_ab_pose)

    # get antigen from complex
    # assumes order H, L, Antigen
    complex_ag_pose.delete_residue_range_slow(1, ag_begin - 1)
    complex_ag_pose.pdb_info().obsolete(False)

    if debug:
        print('comp ', complex_ag_pose.num_chains())
        for i in range(1, complex_ag_pose.num_chains()+1):
            print(pyrosetta.rosetta.core.pose.get_chain_from_chain_id(i, complex_ab_pose))

    #append antigen to complex aligned antibody
    ab_pose.append_pose_by_jump(complex_ag_pose, ag_begin - 1)
    ab_pose.pdb_info().obsolete(False)
    if debug:
        print('ab ',ab_pose.num_chains())
        for i in range(1, ab_pose.num_chains()+1):
            print(pyrosetta.rosetta.core.pose.get_chain_from_chain_id(i, ab_pose))
    return ab_pose

def align_to_complex(ab_pose, complex_pose, partner_chain_str, aligned_pdb):
    ag_chain_id = partner_chain_str.split('_')[1]
    aligned_pose = get_complex_pose_from_antibody_pose(ab_pose, complex_pose, ag_chain_id)
    aligned_pose.dump_pdb(aligned_pdb)
    

def get_tf_ab(pose, rosetta_indices, nn_distance=6.0):
    import pyrosetta.rosetta.core.pack.task.operation as operation
    posi_selector = pyrosetta.rosetta.core.select.residue_selector.ResidueIndexSelector(
    )
    for ind in rosetta_indices:
        posi_selector.append_index(ind)

    # Select Neighbor Position on self
    pose.update_residue_neighbors()
    nbr_selector = pyrosetta.rosetta.core.select.residue_selector.NeighborhoodResidueSelector(
    )
    nbr_selector.set_distance(nn_distance)
    nbr_selector.set_focus_selector(posi_selector)
    nbr_selector.set_include_focus_in_subset(True)
    pyrosetta.rosetta.core.select.get_residues_from_subset(
        nbr_selector.apply(pose))

    # The task factory accepts all the task operations
    tf = pyrosetta.rosetta.core.pack.task.TaskFactory()

    # Repack Everything
    tf.push_back(operation.RestrictToRepacking())

    # These are pretty standard
    tf.push_back(
        pyrosetta.rosetta.core.pack.task.operation.InitializeFromCommandline())
    #tf.push_back(pyrosetta.rosetta.core.pack.task.operation.IncludeCurrent())
    tf.push_back(
        pyrosetta.rosetta.core.pack.task.operation.NoRepackDisulfides())

    # Disable Packing of residues Not in residue-selector
    do_not_pack = pyrosetta.rosetta.core.select.residue_selector.NotResidueSelector(
        posi_selector)
    prevent_repacking_rlt = pyrosetta.rosetta.core.pack.task.operation.PreventRepackingRLT(
    )
    prevent_subset_repacking = \
      pyrosetta.rosetta.core.pack.task.operation.OperateOnResidueSubset(prevent_repacking_rlt, do_not_pack, True )
    tf.push_back(prevent_subset_repacking)

    print(posi_selector)
    print(do_not_pack)

    return tf

def quick_relax_pose_for_scoring(pose, 
                                 relax_max_iter: int = 400):
    
    sf_fa = get_fa_scorefxn()
    
    mmap = pyrosetta.rosetta.core.kinematics.MoveMap()
    mmap.set_bb(True)
    mmap.set_chi(True)
    mmap.set_jump(True)

    relax = pyrosetta.rosetta.protocols.relax.FastRelax()
    relax.set_scorefxn(sf_fa)
    relax.max_iter(relax_max_iter)
    relax.dualspace(True)
    relax.set_movemap(mmap)
    relax.apply(pose)

    fa_min_mover = get_fa_min_mover()
    fa_min_mover.apply(pose)

    return pose
 

def get_multiple_chain_selector(partner, pose) \
  -> pyrosetta.rosetta.core.select.residue_selector.OrResidueSelector:
    partner_grp = pyrosetta.rosetta.core.select.residue_selector.OrResidueSelector(
    )
    #for ch in partner:
    chain_selector = pyrosetta.rosetta.core.select.residue_selector.ChainSelector(
    )
    vec_ch = pyrosetta.rosetta.utility.vector1_std_string()
    for ch in partner:
        vec_ch.append(ch)
    chain_selector.set_chain_strings(vec_ch)
    #show_selector(chain_selector, pose)

    partner_grp.add_residue_selector(chain_selector)
    return partner_grp


def get_tf_complex(pose, chains, nn_distance=8.0, nearby_atom_cut=6.5):
    import pyrosetta.rosetta.core.pack.task.operation as operation
    partner_1 = [t for t in chains.split('_')[0]]
    partner_2 = [t for t in chains.split('_')[1]]

    print('Chains for interface calc: ', partner_1, partner_2)

    partner_grp_1 = get_multiple_chain_selector(partner_1, pose)
    partner_grp_2 = get_multiple_chain_selector(partner_2, pose)

    interface_selector = \
      pyrosetta.rosetta.core.select.residue_selector.InterGroupInterfaceByVectorSelector()

    pose.update_residue_neighbors()
    interface_selector.cb_dist_cut(nn_distance)
    interface_selector.nearby_atom_cut(nearby_atom_cut)
    interface_selector.group1_selector(partner_grp_1)
    interface_selector.group2_selector(partner_grp_2)
    interface_selector.apply(pose)
    #pyrosetta.rosetta.core.select.get_residues_from_subset()
    show_selector(interface_selector, pose, name='InterfaceSelector')
    # The task factory accepts all the task operations
    tf = pyrosetta.rosetta.core.pack.task.TaskFactory()

    # Repack Only
    tf.push_back(operation.RestrictToRepacking())

    tf.push_back(
        pyrosetta.rosetta.core.pack.task.operation.InitializeFromCommandline())
    tf.push_back(
        pyrosetta.rosetta.core.pack.task.operation.NoRepackDisulfides())

    # Disable Packing of residues Not in residue-selector
    do_not_pack = pyrosetta.rosetta.core.select.residue_selector.NotResidueSelector(
        interface_selector)
    prevent_repacking_rlt = pyrosetta.rosetta.core.pack.task.operation.PreventRepackingRLT(
    )
    prevent_subset_repacking = \
      pyrosetta.rosetta.core.pack.task.operation.OperateOnResidueSubset(prevent_repacking_rlt, do_not_pack, True )
    tf.push_back(prevent_subset_repacking)

    return tf


def get_packer_for_ab_pose(pose, rosetta_indices_list, scorefxn, nn_distance=6.0)->\
pyrosetta.rosetta.protocols.minimization_packing.PackRotamersMover:

    init_score = scorefxn(pose)
    pyrosetta.rosetta.core.pose.setPoseExtraScore(pose, 'pre_pack_total_score',
                                                  init_score)

    tf = get_tf_ab(pose, rosetta_indices_list, nn_distance=nn_distance)
    # Create Packer
    packer = pyrosetta.rosetta.protocols.minimization_packing.PackRotamersMover(
    )
    packer.task_factory(tf)

    return packer


def fast_relax_pose_ab(pose, ros_indices, outfile='relax_ab.pdb', dump=True):
    pyrosetta.init()
    tf = get_tf_ab(pose, ros_indices)
    relax = pyrosetta.rosetta.protocols.relax.FastRelax()
    scorefxn = get_fa_scorefxn()
    relax.set_scorefxn(scorefxn)
    relax.constrain_relax_to_start_coords(True)
    relax.set_task_factory(tf)
    relax.apply(pose)
    total_score = scorefxn(pose)
    pyrosetta.rosetta.core.pose.setPoseExtraScore(pose,
                                                  'post_relax_total_score',
                                                  total_score)
    if dump:
        pose.dump_pdb(outfile)

    return total_score, pose


def get_interface_analyzer_basic(partner_chain_str, scorefxn,
                           pack_separated=False) \
  -> pyrosetta.rosetta.protocols.analysis.InterfaceAnalyzerMover:
    interface_analyzer = pyrosetta.rosetta.protocols.analysis.InterfaceAnalyzerMover(
    )
    interface_analyzer.fresh_instance()
    interface_analyzer.set_interface(partner_chain_str)
    interface_analyzer.set_scorefunction(scorefxn)
    interface_analyzer.set_compute_interface_energy(True)
    interface_analyzer.set_compute_interface_sc(True)
    interface_analyzer.set_calc_dSASA(True)
    interface_analyzer.set_pack_separated(pack_separated)

    return interface_analyzer


def fast_relax_pose_complex(pdb,
                            chains,
                            idecoy,
                            seq='',
                            outfile='relax_complex.pdb',
                            dump=True,
                            max_iter=800,
                            dry_run=False,
                            dock=False,
                            induced_docking_res=[],
                            per_res_data=False,
                            pose=None):

    pyrosetta.init('--mute all')
    if pose is None:
        pose = pose_from_file(pdb)
    pyrosetta.rosetta.core.pose.setPoseExtraScore(pose, 'design', seq)
    
    if not dock:
        print('Not docking ')
        tf = get_tf_complex(pose, chains)
        mmap = pyrosetta.rosetta.core.kinematics.MoveMap()
        mmap.set_bb(False)
        mmap.set_chi(True)
        mmap.set_jump(False)
    else:
        setup_foldtree(pose, chains, Vector1([1]))
        print(pose.fold_tree())
        tf = get_tf_complex(pose, chains)
        mmap = pyrosetta.rosetta.core.kinematics.MoveMap()
        print(induced_docking_res)
        mmap.set_bb_true_range(induced_docking_res[0], induced_docking_res[1])
        mmap.set_chi(True)
        mmap.set_jump(False)

    scorefxn = get_fa_scorefxn()
    
    if dock:
        #Backrub residues in the presence of antigen
        bb_mover = pyrosetta.rosetta.protocols.backrub.BackrubProtocol()
        bb_mover.set_scorefunction(scorefxn)
        bb_mover.set_taskfactory(tf)
        bb_mover.set_movemap(mmap)
        vec_piv_res = pyrosetta.rosetta.utility.vector1_unsigned_long()
        vec_piv_res.extend([t for t in range(induced_docking_res[0], induced_docking_res[1]+1)])
        bb_mover.set_pivot_residues(vec_piv_res)
        if not dry_run:
            bb_mover.apply(pose)
            total_score = scorefxn(pose)
            pyrosetta.rosetta.core.pose.setPoseExtraScore(pose,
                                                  'post_backrub_total_score',
                                                  total_score)
            if dump:
                pose.dump_pdb(outfile.split('.pdb')[0]+'_bb.pdb')

    relax = pyrosetta.rosetta.protocols.relax.FastRelax()
    relax.set_scorefxn(scorefxn)
    relax.max_iter(max_iter)
    relax.set_movemap(mmap)
    if dock:
        relax.dualspace(True)
    relax.set_task_factory(tf)
    if not dry_run:
        relax.apply(pose)

    total_score = scorefxn(pose)
    pyrosetta.rosetta.core.pose.setPoseExtraScore(pose,
                                                  'post_relax_total_score',
                                                  total_score)
    if not dock:
        interface_analyzer = get_interface_analyzer_basic(chains, scorefxn)
    else:
        interface_analyzer = get_interface_analyzer_basic(chains, scorefxn)
    interface_analyzer.apply(pose)
    interface_analyzer.add_score_info_to_pose(pose)
    if dump:
        pose.dump_pdb(outfile)
    
    dg = interface_analyzer.get_interface_dG()
    if per_res_data:
        dg_per_res = interface_analyzer.get_all_per_residue_data()
        return dg_per_res
    return (idecoy, total_score, dg)


def pack_pose(pose,
              indices_list,
              dump=True,
              outfile='pose.pdb',
              nn_distance=6.0):
    '''Adapted from computational DMS
        to pack and minimize mutants without backbone moves
      '''
    #Perform The Move
    scorefxn = pyrosetta.create_score_function('ref2015')
    packer_mover = get_packer_for_ab_pose(pose,
                                          indices_list,
                                          scorefxn,
                                          nn_distance=nn_distance)
    packer_mover.apply(pose)
    total_score = scorefxn(pose)
    pyrosetta.rosetta.core.pose.setPoseExtraScore(pose,
                                                  'post_pack_total_score',
                                                  total_score)
    if dump:
        print('Dumping file: ', outfile)
        pose.dump_pdb(outfile)

    return total_score, pose


def MutateResidue(pose_mod, target_indices, new_residues):
    assert (len(target_indices) == len(new_residues))
    for tind, nr in zip(target_indices, new_residues):
        mutate_mover = psm.MutateResidue(tind, nr)
        mutate_mover.apply(pose_mod)

    return pose_mod


def mutate_pose(base_pose,
                ros_positions,
                seq,
                outfile='mutated.pdb',
                dump=True):
    pose = base_pose.clone()
    seqlist = [aa_1to3_map[t] for t in seq]
    new_pose = MutateResidue(pose, ros_positions, seqlist)

    if dump:
        new_pose.dump_pdb(outfile)

    return new_pose


def relax_pose(pdb,
               outfile_relax,
               iseq,
               chains,
               seq='',
               use_cluster=False,
               decoys=20,
               dry_run=False,
               dock=False,
               induced_docking_res=[]):
    relaxed_poses = []
    if use_cluster:
        for index_decoy in range(decoys):
            client = get_client()
            score_tuple = client.submit(fast_relax_pose_complex,
                                        pdb,
                                        chains,
                                        index_decoy,
                                        seq=seq,
                                        outfile=outfile_relax.format(
                                            '%03d_%03d' % (iseq, index_decoy)),
                                        dry_run=dry_run,
                                        dock=dock,
                                        induced_docking_res=induced_docking_res)
            relaxed_poses.append(score_tuple)
        relaxed_poses = client.gather(relaxed_poses)
    else:
        for index_decoy in range(decoys):
            score_tuple = fast_relax_pose_complex(
                pdb,
                chains,
                index_decoy,
                seq=seq,
                outfile=outfile_relax.format('%03d_%03d' %
                                             (iseq, index_decoy)),
                dry_run=dry_run,
                dock=dock,
                induced_docking_res=induced_docking_res)

        relaxed_poses.append(score_tuple)

    return relaxed_poses


def score_pdb(pdb, relax=False):
    pose = pyrosetta.pose_from_pdb(pdb)
    
    sf_fa = get_fa_scorefxn()
    if not relax:
        return sf_fa(pose)
    else:
        relaxed_pose = quick_relax_pose_for_scoring(pose)
        return sf_fa(relaxed_pose)


def get_sapscores(pdb, sel_rosetta_indices=None, sasa_dist=5):
    pose = pose_from_file(pdb)
    base_sel = pyrosetta.rosetta.core.select.residue_selector.LayerSelector()
    #select surface
    base_sel.set_layers(False, False, True)
    score_sel = base_sel
    sap_sel = base_sel
    sasa_sel = base_sel

    if not sel_rosetta_indices is None:
        posi_selector = pyrosetta.rosetta.core.select.residue_selector.ResidueIndexSelector()
        for ind in sel_rosetta_indices:
            posi_selector.append_index(ind)
        
        # SASA of design residues + 5A neighborhood
        # 5A gives best preds according to Lauer .. Trout 2012 SAP score paper (doi: 10.1002/jps.22758)
        
        pose.update_residue_neighbors()
        nbr_selector = pyrosetta.rosetta.core.select.residue_selector.NeighborhoodResidueSelector(
        )
        nbr_selector.set_distance(sasa_dist)
        nbr_selector.set_focus_selector(posi_selector)
        nbr_selector.set_include_focus_in_subset(True)
        pyrosetta.rosetta.core.select.get_residues_from_subset(
            nbr_selector.apply(pose))

        score_sel = posi_selector
        sap_sel = posi_selector
        sasa_sel = nbr_selector

    score = pyrosetta.rosetta.core.pack.guidance_scoreterms.sap.calculate_sap(pose, score_sel, sap_sel, sasa_sel)
    return score