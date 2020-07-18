from Bio import Align
from Bio.Align import substitution_matrices

from Bio import SeqIO

from random import choice, random

import pandas as pd

# Parses through the records and returns the sequence
def getFasta(fasta = "data/sequence.fasta"):
	records = []
	for record in SeqIO.parse(fasta, "fasta"):
		 records.append(record)
	return records


# Create random subsequence with a given length
# sequence = create_random_sequence(1000)
def create_random_sequence(n):
	dna = ["A","G","C","T"]
	seq = ''
	for i in range(n):
	    seq += choice(dna)
	return seq


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


parent_sequence = getFasta()
k = 20

for i in range(1000):
	sequence = select_random_subseq(parent_sequence[0], 1000)
	kmer_list = create_kmer_list(str(sequence.seq), k)
	random_kmer = select_random_kmer(kmer_list)

	aligner = Align.PairwiseAligner(match_score=1.0, mismatch_score = -2.0, gap_score = -2.5)
	alignment_scores = create_alignment_table(random_kmer, kmer_list)
	alignment_scores = [e / max(alignment_scores) for e in alignment_scores]

	data = pd.DataFrame({'kmer': kmer_list, 'score': alignment_scores})
	fdir = 'data/ref_sequences0/'
	data.to_csv(fdir + random_kmer + '_' + str(k) + '.txt', mode = 'a', index = False)

