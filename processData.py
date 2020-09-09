from Bio import Align
from Bio.Align import substitution_matrices

from Bio import SeqIO

from random import choice, random

import pandas as pd

import glob

# Parses through the records and returns the sequence
def getFasta(fasta = "data/sequence.fasta"):
	records = []
	for record in SeqIO.parse(fasta, "fasta"):
		 records.append(record)
	return records


# Create a list of kmers given a sequence
def create_kmer_list(seq, k):
	return [seq[i:i + k] for i in range(len(seq) - k)]


# Select a random kmer from a list of kmers
def select_random_kmer(kmer_list):
	return kmer_list[int(random() * len(kmer_list))]


# Select random subsequence of length n from seq
def select_random_subseq(seq, n):
	i = int(random() * (len(seq) - n))
	return seq[i: i + n]


# Create table of alignment scores between subseq and every kmer in a list
def create_alignment_table(subseq, kmer_list):
	return [aligner.score(subseq, e) for e in kmer_list]


def insert_random_motif(seq, motif, mismatches=0):
	i = int(random() * (len(seq) - len(motif)))
	if mismatches:
		for j in range(mismatches):
			k = int(random() * len(motif))
			motif = motif[:k] + create_random_sequence(1) + motif[k:]
	return seq[:i] + motif + seq[i + len(motif):]


# Create random subsequence with a given length
# sequence = create_random_sequence(1000)
def create_random_sequence(n):
	dna = ["A","G","C","T"]
	seq = ''
	for i in range(n):
	    seq += choice(dna)
	return seq


# Given a list of motifs, the function returns the counts of each nucleotide (rows) across the length of the k-mers (columns).
def get_nuc_counts(motif_list):
	count_list = [[0] * len(motif_list[0]) for i in range(4)]
	for i in range(len(motif_list[0])):
		for k in range(len(nuc_list)):
			count_list[k][i] = [motif_list[j][i] for j in range(len(motif_list))].count(nuc_list[k])
	return count_list


# Given a matrix of nucleotide counts across each position in a k-mer, the function returns a frequency matrix using Lapace's theorem: P(x = 1 | x1, x2, ..., xn = 1) = (s + 1) / (n + 2).
def get_nuc_frequency_laplace(nuc_counts):
	nuc_counts = (np.array(nuc_counts) + 1) / 8
	return nuc_counts.tolist()


#parent_sequence = getFasta()
k = 15

motif = 'ATTGATTCGGATAGC'

max_number_of_mutations = 5

motifs = [motif[:i] + create_random_sequence(1) + motif[i:] for i in [int(random() * len(motif))] for _ in range(int(random() * max_number_of_mutations + 1)) for _ in range(10)]

nuc_frequency = get_nuc_frequency_laplace(get_nuc_counts(motifs))

# When we insert motifs, we want to select their nucleotides based on this frequency table.

for j in range(0, 10000):

	# biopython
	#sequence = select_random_subseq(parent_sequence[0], 1000)

	# Generate random seqeunce and insert the same motif n times
	sequence = create_random_sequence(1000)
	for i in range(int(random() * 11) + 5):
		# Change insert random motif so that the length doesnt increase
		sequence = insert_random_motif(sequence, motif, 2)

	# biopython
	#kmer_list = create_kmer_list(str(sequence.seq), k)
	kmer_list = create_kmer_list(sequence, k)
	
	random_kmer = motif
	# biopython
	#random_kmer = select_random_kmer(kmer_list)

	aligner = Align.PairwiseAligner()
	aligner.match_score = 1.0
	aligner.mismatch_score = -2.0
	aligner.gap_score = -2.5
	aligner.mode = 'local'
	alignment_scores = create_alignment_table(random_kmer, kmer_list)
	alignment_scores = [e / max(alignment_scores) for e in alignment_scores]

	data = pd.DataFrame({'kmer': kmer_list, 'score': alignment_scores})
	fdir = 'data/ref_sequences2/'
	filename = fdir + random_kmer + '_' + str(j) + '_' + str(k) + '.txt'
	if not glob.glob(filename):
		data.to_csv(filename, mode = 'a', index = False)

