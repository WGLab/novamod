import numpy as np
import pod5 as p5
import blosc2
import pysam
import multiprocessing as mp
import os, re, argparse, math
import numpy as np
from pathlib import Path
from numba import jit
from itertools import repeat


base_to_num_map={'A':0, 'C':1, 'G':2, 'T':3, 'U':3,'N':4}

num_to_base_map={0:'A', 1:'C', 2:'G', 3:'T', 4:'N'}

comp_base_map={'A':'T','T':'A','C':'G','G':'C','[':']', ']':'['}

cigar_map={'M':0, '=':0, 'X':0, 'D':1, 'I':2, 'S':3,'H':3, 'P':4, 'B':4, 'N':5}
cigar_pattern = r'\d+[A-Za-z=]'

def revcomp(s):
    return ''.join(comp_base_map[x] for x in s[::-1])

def motif_check(motif):
    nt_dict={'R': 'GA',
             'Y': 'CT',
             'K': 'GT',
             'M': 'AC',
             'S': 'GC',
             'W': 'AT',
             'B': 'GTC',
             'D': 'GAT',
             'H': 'ACT',
             'V': 'GCA',
             'N': 'AGCT'}
    
    valid_alphabet=set(nt_dict.keys()).union({'A', 'C', 'G', 'T'})
    
    exp_motif_seq, final_motif_ind, valid = None, None, False
    
    motif=motif.split()
    if len(motif)<2:
        print('--motif not specified correctly. You need to specify a motif and at least one index',flush=True)
        return None, exp_motif_seq, final_motif_ind, valid
    
    elif len(set(motif[0])-valid_alphabet)>0:
        print('--motif not specified correctly. Motif should only consist of the following extended nucleotide letters: {}'.format(','.join(valid_alphabet)),flush=True)
        return None, exp_motif_seq, final_motif_ind, valid
    
    elif all([a.isnumeric() for a in motif[1:]])==False:
        print('--motif not specified correctly. Motif indices should be integers separated by whitespace and shoud come after the motif sequence.',flush=True)
        return None, exp_motif_seq, final_motif_ind, valid
    
    else:
        motif_seq=motif[0]
        motif_ind=[int(x) for x in motif[1:]]
        
        if len(set(motif_seq[x] for x in motif_ind))!=1 or len(set(motif_seq[x] for x in motif_ind)-set('ACGT'))>0:
            print('Base of interest should be same for all indices and must be one of A, C, G, or T.', flush=True)
            return motif_seq, exp_motif_seq, final_motif_ind, valid
        
        else:
            exp_motif_seq=motif_seq
            for nt in nt_dict:
                if nt in exp_motif_seq:
                    exp_motif_seq=exp_motif_seq.replace(nt, '[{}]'.format(nt_dict[nt]))
            return motif_seq, exp_motif_seq, motif_ind, True
        
def get_candidates(read_seq, align_data, aligned_pairs, ref_pos_dict, ):    
    is_mapped, is_forward, ref_name, reference_start, reference_end, read_length=align_data

    ref_motif_pos=ref_pos_dict[ref_name][0] if is_forward else ref_pos_dict[ref_name][1]

    common_pos=ref_motif_pos[(ref_motif_pos>=reference_start)&(ref_motif_pos<reference_end)]
    aligned_pairs_ref_wise=aligned_pairs[aligned_pairs[:,1]!=-1][common_pos-reference_start]

    aligned_pairs_ref_wise=aligned_pairs_ref_wise[aligned_pairs_ref_wise[:,0]!=-1]
    aligned_pairs_read_wise_original=aligned_pairs[aligned_pairs[:,0]!=-1]
    aligned_pairs_read_wise=np.copy(aligned_pairs_read_wise_original)

    if not is_forward:
        aligned_pairs_ref_wise=aligned_pairs_ref_wise[::-1]
        aligned_pairs_read_wise=aligned_pairs_read_wise[::-1]
        aligned_pairs_ref_wise[:,0]=read_length-aligned_pairs_ref_wise[:,0]-1
        aligned_pairs_read_wise[:,0]=read_length-aligned_pairs_read_wise[:,0]-1

    return aligned_pairs_ref_wise, aligned_pairs_read_wise_original


@jit(nopython=True)
def get_aligned_pairs(cigar_tuples, ref_start):
    alen=np.sum(cigar_tuples[:,0])
    pairs=np.zeros((alen,2)).astype(np.int32)

    i=0
    ref_cord=ref_start-1
    read_cord=-1
    pair_cord=0
    for i in range(len(cigar_tuples)):
        len_op, op= cigar_tuples[i,0], cigar_tuples[i,1]
        if op==0:
            for k in range(len_op):            
                ref_cord+=1
                read_cord+=1

                pairs[pair_cord,0]=read_cord
                pairs[pair_cord,1]=ref_cord
                pair_cord+=1

        elif op==2 or op==3:
            for k in range(len_op):            
                read_cord+=1            
                pairs[pair_cord,0]=read_cord
                pairs[pair_cord,1]=-1
                pair_cord+=1

        elif op==1 or op==5:
            for k in range(len_op):            
                ref_cord+=1            
                pairs[pair_cord,0]=-1
                pairs[pair_cord,1]=ref_cord
                pair_cord+=1
    return pairs

@jit(nopython=True)
def get_ref_to_num(x):
    b=np.full((len(x)+1,2),fill_value=0,dtype=np.int8)
    
    for i,l in enumerate(x):
        if l=='A':
            b[i,0]=0
            b[i,1]=3
            
        elif l=='T':
            b[i,0]=3
            b[i,1]=0
            
        elif l=='C':
            b[i,0]=1
            b[i,1]=2
            
        elif l=='G':
            b[i,0]=2
            b[i,1]=1
            
        else:
            b[i,0]=4
            b[i,1]=4
    
    b[-1,0]=4
    b[-1,1]=4
            
    return b

def get_ref_info(args):
    params, chrom=args
    motif_seq, motif_ind=params['motif_seq'], params['motif_ind']
    exp_motif_seq=params['exp_motif_seq']
    ref_fasta=pysam.FastaFile(params['ref'])
    seq=ref_fasta.fetch(chrom).upper()
    seq_array=get_ref_to_num(seq)
    fwd_pos_array, rev_pos_array=None, None
    if motif_seq:
        fwd_motif_anchor=np.array([m.start(0) for m in re.finditer(r'(?={})'.format(exp_motif_seq), seq)])
        rev_motif_anchor=np.array([m.start(0) for m in re.finditer(r'(?={})'.format(revcomp(exp_motif_seq)), seq)])

        fwd_pos_array=np.array(sorted(list(set.union(*[set(fwd_motif_anchor+i) for i in motif_ind])))).astype(int)
        rev_pos_array=np.array(sorted(list(set.union(*[set(rev_motif_anchor+len(motif_seq)-1-i) for i in motif_ind])))).astype(int)
    
    return chrom, seq_array, fwd_pos_array, rev_pos_array

def get_pos(path):
    labelled_pos_list={}
    strand_map={'+':0, '-':1}
    
    with open(path) as file:
            for line in file:
                line=line.rstrip('\n').split('\t')
                if line[0] not in labelled_pos_list:
                    labelled_pos_list[line[0]]={0:{}, 1:{}}
                    
                labelled_pos_list[line[0]][strand_map[line[5]]][int(line[1])]=1.0
    return labelled_pos_list

class Read():
    def __init__(self, move_table, signal, base_qual, full_seq, aligned_pairs, read_name, ref_name, shift, scale, norm_type, seq_type, div, flag, reference_start, reference_end, cigarstring):
        """
        move_table: A vector of length M, where 1's indicate when signal chunk for a new base starts
        signal: A matrix of size M x stride, containing un-normalized raw signal chunks. Each chunk corresponds to one move table entry.
        indexes: A vector of length M, that records which read coordinate corresponds to each signal chunk.
        base_qual: A vector of length N equal to read sequence length, containing base call error probability
        segments: A matrix of length Nx2, which tells the start and end (non-inclusive) index of signal chunk corresponding to each base.
        full_seq: A matrix of size R x 2, where R is the alignment length with reference genome.
                  First and second columns are one-hot encoding of read and reference sequence. 
                  The encoding is given by {'A':0, 'C':1, 'G':2, 'T':3, 'U':3,'N':4}, where N is a gap.
        aligned_pairs: A matrix of size R x 2, where each row shows which read coordinate (column 1) corresponds to 
                       which reference coordinate (column 2). A value of -1 indicates a gap.

        read_name: Read name
        ref_name: Reference contig that the read is aligned to (primary alignment only)
        shift: shift
        scale: scale
        norm_type: what type of normalization provides shift and scale
        div: alignment diverge stats (mismatches, indels, alignment length (ingorning splicing))
        ref_start:
        ref_end:
        """
        
        self.move_table=move_table
        self.signal=signal.reshape(len(move_table), -1)
        self.base_qual=base_qual
        self.full_seq=full_seq
        self.aligned_pairs=aligned_pairs
        self.length=len(self.base_qual)
        
        self.indexes=np.cumsum(move_table)-1
        segments=np.concatenate([np.where(move_table)[0], [len(move_table)]])
        self.segments=np.vstack([segments[:-1], segments[1:]]).T
        
        
        self.read_name=read_name
        self.ref_name=ref_name
        self.shift=shift
        self.scale=scale
        self.norm_type=norm_type
        self.seq_type=seq_type
        self.div=div
        
        self.flag=flag
        self.is_forward=flag&16==0
        self.is_reverse=flag&16>0
        
        self.ref_start = reference_start
        self.ref_end = reference_end
        
        self.cigarstring=cigarstring
    
    def normalize(self):
        """Normalize the signal using shift and scale"""
        
        self.signal=(self.signal-self.shift)/self.scale
    
    
    def get_base_seq(self, string=False, expanded=False):
        """Get basecall sequence in 5' to 3' direction, without N for indels/clipping.
        
        string [True, False]: Return one-hot encoding if string is False otherwise return nucleotide string sequence.
        
        expanded [True, False]: Expand the sequence to match the size of signal matrix, repeating values if a base has several strides (rows) in signal
        """
                    
        seq=self.full_seq[self.aligned_pairs[:,0]!=-1][:,0]
        if string:
            seq=np.vectorize(num_to_base_map.get)(seq)
        
        if expanded:
            return seq[self.indexes]
        else:
            return seq
        
            
    def get_ref_seq(self, string=False, expanded=False):
        """Get reference sequence in 5' to 3' direction.
        Return one-hot encoding if string is False otherwise return nucleotide string sequence
        
        string [True, False]: Return one-hot encoding if string is False otherwise return nucleotide string sequence.


        expanded [True, False]: If Ture, expand the sequence to match the size of signal matrix and only give bases that have read alignment with N's for indels, repeating values if a base has several strides (rows) in signal. 
        Otherwise return sequence for the without N for indels/clipping
        """
        
        if expanded:
            seq=self.full_seq[self.aligned_pairs[:,0]!=-1][:,1]
            return seq[self.indexes]
        
        else:
            seq=self.full_seq[self.aligned_pairs[:,1]!=-1][:,1]
        
        if string:
            seq=np.vectorize(num_to_base_map.get)(seq)
        
        return seq
    
    def get_alignment(self, string=False):
        """Get alignment of read vs ref sequence in 5' to 3' direction, including N for indels/clipping.
        Returns Mx2 matrix, where M is alignment length, first column basecall sequence
        and second column is reference sequence. Returns one-hot encoding 
        if string is False, otherwise return nucleotide string sequence"""
        
        if string:
            return np.vectorize(num_to_base_map.get)(full_seq) 
        else:
            return self.full_seq
    
    def get_divergence(self, include_mismatches=False):
        """Get percent divergence to assess read quality.
        include_mismatches [True, False]: By default do not include mismatches in diverge
                                          since they can be caused by modification.
        """
        mismatches, indels, alen=self.div
        return (indels+mismatches)/alen if include_mismatches else indels/alen
    
    def get_labels(self, ref_pos_dict, labelled_pos_list):
        ref_motif_pos=ref_pos_dict[self.ref_name][self.is_reverse]
        common_pos=ref_motif_pos[(ref_motif_pos>=self.ref_start)&(ref_motif_pos<self.ref_end)]
        
        tmp_aligned_pairs=np.hstack([self.aligned_pairs, np.arange(len(self.aligned_pairs))[:,np.newaxis]])
        if self.is_forward:
            aligned_pairs_ref_wise=tmp_aligned_pairs[tmp_aligned_pairs[:,1]!=-1][common_pos-self.ref_start]
        else:
            aligned_pairs=tmp_aligned_pairs[::-1]
            aligned_pairs_ref_wise=aligned_pairs[aligned_pairs[:,1]!=-1][common_pos-self.ref_start]
            aligned_pairs_ref_wise=aligned_pairs_ref_wise[::-1]
    
        aligned_pairs_ref_wise=aligned_pairs_ref_wise[aligned_pairs_ref_wise[:,0]!=-1]

        try:
            labels=np.vectorize(labelled_pos_list[self.ref_name][self.is_reverse].get)(aligned_pairs_ref_wise[:,1])
        except ValueError:
            return None
        
        rec_arr = np.core.records.fromarrays(np.hstack([aligned_pairs_ref_wise, labels[:,np.newaxis]]).T, dtype=[('read_pos', 'i4'), ('ref_pos', 'i4'),('align_idx', 'i4'), ('label', 'f4')])
        
        return rec_arr

def get_read_info(bam_read, ref_seq_dict, seq_type):
    is_mapped, is_forward, ref_name, reference_start, reference_end, read_length=bam_read.is_mapped, bam_read.is_forward, bam_read.reference_name, bam_read.reference_start, bam_read.reference_end, bam_read.query_length

    fq=bam_read.seq
    qual=bam_read.qual
    sequence_length=len(fq)
    reverse= not is_forward
    fq=revcomp(fq) if reverse else fq
    qual=qual[::-1] if reverse else qual

    base_qual=10**((33-np.array([ord(x) for x in qual]))/10)
    mean_qscore=-10*np.log10(np.mean(base_qual))
    base_qual=base_qual
            
    cigar_tuples = np.array([(int(x[:-1]), cigar_map[x[-1]]) for x in re.findall(cigar_pattern, bam_read.cigarstring)])
    ref_start=bam_read.reference_start
    aligned_pairs=get_aligned_pairs(cigar_tuples, ref_start)

    #pos_list_candidates, read_to_ref_pairs=get_candidates(fq, align_data, aligned_pairs, ref_pos_dict)

    if reverse:
        aligned_pairs=aligned_pairs[::-1]
        aligned_pairs[aligned_pairs[:,0]!=-1,0]=read_length-aligned_pairs[aligned_pairs[:,0]!=-1,0]-1

    base_seq=np.array([base_to_num_map[x] for x in fq]).astype(np.int8)
    ref_seq=ref_seq_dict[ref_name][:,1][aligned_pairs[:, 1]] if reverse else ref_seq_dict[ref_name][:,0][aligned_pairs[:, 1]]

    exp_base_seq=np.full(len(aligned_pairs),4).astype(np.int8)
    exp_base_seq[aligned_pairs[:,0]!=-1]=base_seq
    full_seq=np.stack([exp_base_seq, ref_seq]).T

    tags={x[0]:x[1] for x in bam_read.get_tags(with_value_type=True) if ~(x[2]=='A' and x[0]=='ts')}
    
    start=tags['ts']
    mv=tags['mv']

    stride=mv[0]
    move_table=np.array(mv[1:])

    shift, scale = tags['sm'], tags['sd']
    norm_type=tags['sv']
    shift=abs(shift)
    scale=max(scale, 1/scale)

    signal=tags['SG']
    signal = blosc2.decompress2(signal)
    signal = np.frombuffer(signal, dtype=np.int16)

    max_idx=start+len(move_table)*stride
    mat=signal[start:max_idx].reshape(-1, stride)
    
    if seq_type=='rna':
        mat=np.flip(mat)
        new_move_table=np.full(len(move_table),0)
        new_move_table[np.where(move_table)[0][1:]-1]=1
        new_move_table[-1]=1
        new_move_table=new_move_table[::-1]
        move_table=new_move_table

    indels=np.sum(cigar_tuples[:,0][(cigar_tuples[:,1]==1)|(cigar_tuples[:,1]==2)])
    mismatches=np.sum((full_seq[:,0]!=full_seq[:,1])[np.all(aligned_pairs!=-1,axis=1)])
    alen=np.sum(cigar_tuples[:,0][cigar_tuples[:,1]<=2])
    div=(mismatches, indels, alen)
    
    return move_table, mat, base_qual, full_seq, aligned_pairs, bam_read.qname, ref_name, shift, scale, norm_type, seq_type, div, bam_read.flag, bam_read.reference_start, bam_read.reference_end, bam_read.cigarstring

def pre_process(ref: str, seq_type: str, pos_list:str, motif: str, motif_label:int=0, chrom_list: list=[],bam: str=None):
    if motif and len(motif)>0:
        motif_seq, exp_motif_seq, motif_ind, valid_motif=motif_check(motif)
    else:
        motif_seq, exp_motif_seq, motif_ind=None, None, None
    
    if not chrom_list:
        if bam:
            chrom_list=pysam.Samfile(bam).references 
        else:
            chrom_list=pysam.FastaFile(ref).references
    
    params=dict(bam=bam, 
                seq_type=seq_type,
                pos_list=pos_list, 
                ref=ref,
                motif_seq=motif_seq,
                motif_ind=motif_ind,
                exp_motif_seq=exp_motif_seq,
                chrom=chrom_list)

    ref_seq_dict={}
    ref_pos_dict={}

    labelled_pos_list={}

    if params['pos_list']:
        labelled_pos_list=get_pos(params['pos_list'])
        params['chrom']=[x for x in params['chrom'] if x in labelled_pos_list.keys()]

    _=get_ref_to_num('ACGT')
    ref_seq_dict={}

    with mp.Pool(processes=4) as pool:
        res=pool.map(get_ref_info, zip(repeat(params), params['chrom']))
        for r in res:
            chrom, seq_array, fwd_pos_array, rev_pos_array=r
            ref_seq_dict[chrom]=seq_array

            if params['pos_list']:
                ref_pos_dict[chrom]=(np.array(sorted(list(labelled_pos_list[chrom][0].keys()))).astype(int), np.array(sorted(list(labelled_pos_list[chrom][1].keys()))).astype(int))

            elif params['motif_seq']:
                ref_pos_dict[chrom]=(fwd_pos_array, rev_pos_array)
                labelled_pos_list[chrom]={0:{}, 1:{}}
                for strand in [0,1]:
                    for pos in ref_pos_dict[chrom][strand]:
                        labelled_pos_list[chrom][strand][pos]=float(motif_label)

            else:
                ref_pos_dict[chrom]=(fwd_pos_array, rev_pos_array)
                labelled_pos_list[chrom]={0:{}, 1:{}}
                
    return ref_seq_dict, ref_pos_dict, labelled_pos_list