import os.path
from src.util.pdb import get_pdb_chain_seq, get_pdb_numbering_from_residue_indices

def get_frequency_of_specific_residue(one_letter_code, list_of_sequences):
    frequencies = []
    for seq in list_of_sequences:
        frequencies.append(seq.count(one_letter_code))
    return frequencies


def slice_seq(seq, indices):
    return ''.join([seq[i] for i in indices])

def slice_sequences(sequences, indices):
    # Slicing out the sequence stretch of interest and writing to file
    return [ slice_seq(seq, indices) for seq in sequences]

class SequenceTrajectory(object):

    def __init__(self, final_fasta_file):

        self.final_fasta_file = final_fasta_file
        self.generate_sequences_dict_by_iter()

    def generate_sequences_dict_by_iter(self):
       
        dat = open(self.final_fasta_file, 'r').readlines()

        self.sequence_by_iter_dict = {}
        self.number_of_iterations = int(len(dat)/5 -1)

        for iter in range(0,self.number_of_iterations + 1):
            assert dat[iter*5].startswith(">H")
            heavy = dat[iter*5 + 1].rstrip()
            assert dat[iter*5 + 2].startswith(">L")
            light = dat[iter*5 + 3].rstrip()
            seq = heavy +light
            self.sequence_by_iter_dict[iter] = seq

        self.final_sequence = self.sequence_by_iter_dict[self.number_of_iterations]
        self.design_id = int(os.path.basename(self.final_fasta_file).split('_')[1])

class HallucinationDataReader(object):

    def __init__(self, indices_hal, target_pdb, hallucination_dir=''):
        
        self.hallucination_dir = hallucination_dir
        self.target_pdb = target_pdb 
        self.len_heavy = len(get_pdb_chain_seq(target_pdb, 'H'))
        self.name = os.path.split(self.target_pdb)[1].rstrip('.pdb')
        self.indices_hal = indices_hal
        self.pdb_res_nums = get_pdb_numbering_from_residue_indices(self.target_pdb, self.indices_hal)
        if hallucination_dir != '':
            self.list_of_final_fastas = [os.path.join(os.path.abspath(hallucination_dir), f) 
                                            for f in os.listdir(hallucination_dir) 
                                            if f.endswith("_final.fasta") and f.startswith('sequence')]
            self.sequence_trajectories = [SequenceTrajectory(f) for f in self.list_of_final_fastas]
            self.dict_of_all_sequence_dicts = \
                {st.design_id:st.sequence_by_iter_dict for st in self.sequence_trajectories}
            self.list_of_all_sequence_dicts = \
                [st.sequence_by_iter_dict for st in self.sequence_trajectories]
            self.list_of_final_sequences = \
                [list(dictionary.values())[-1] for dictionary in self.list_of_all_sequence_dicts]
            self.list_of_final_des_subsequences = slice_sequences(self.list_of_final_sequences, indices_hal)
            self.dict_of_final_sequences = {st.design_id:st.final_sequence for st in self.sequence_trajectories}
            self.dict_of_final_des_subsequences = {desid:slice_seq(self.dict_of_final_sequences[desid], indices_hal)\
                                                     for desid in self.dict_of_final_sequences}
            
            self.dict_of_all_des_subsequence_dicts = {design_id:slice_sequences(list(self.dict_of_all_sequence_dicts[design_id].values()),
                                                                                indices_hal) 
                                                        for design_id in self.dict_of_all_sequence_dicts}
        else:
            self.list_of_final_sequences = []

    def write_final_sequences_to_fasta(self, outfile):
        fasta_format = '>sequences_{}_final:H\n{}\n>sequences_{}_final:L\n{}\n'
        sorted_keys = sorted(list(self.dict_of_final_sequences.keys()))
        self.dict_of_final_sequences = {key:self.dict_of_final_sequences[key]
                                                    for key in sorted_keys}
        with open(outfile, 'w') as f:
            for desid in self.dict_of_final_sequences:
                heavy_seq = self.dict_of_final_sequences[desid][:self.len_heavy]
                light_seq = self.dict_of_final_sequences[desid][self.len_heavy:]
                f.write(fasta_format.format(desid, heavy_seq, desid, light_seq))

    def write_final_subsequences_to_fasta(self, outfile):
        fasta_format = '>sequences_{}_final\n{}\n'
        sorted_keys = sorted(list(self.dict_of_final_des_subsequences.keys()))
        self.dict_of_final_des_subsequences = {key:self.dict_of_final_des_subsequences[key]
                                                    for key in sorted_keys}
        with open(outfile, 'w') as f:
            for desid in self.dict_of_final_des_subsequences:
                heavy_seq = self.dict_of_final_des_subsequences[desid][:self.len_heavy]
                light_seq = self.dict_of_final_des_subsequences[desid][self.len_heavy:]
                f.write(fasta_format.format(desid, heavy_seq, desid, light_seq))