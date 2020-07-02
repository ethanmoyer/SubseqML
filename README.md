# VERTICALLY INTEGRATED PROJECT (VIP) REPORT: NEUROMORPHIC SOLUTIONS TO HIGH DIMENSIONAL DNA SUBSEQUENCE MATCHING VIA K-MER WISE ALIGNMENT 

## Data Curation/Analysis

All the data preprocessed data used in the analysises are kept in the data directory. All data that is processed by using preprocessing.py is kepts in the processed_data directory.

### Matlab Pipeline

All of the Matlab code created to curate and cluster data is in the matlab-data-curation directory.

### generateDNA.m

This function generates a dna molecule of length n with the appropriate proporitons pa, pt, pc, and pg. The function returns the randomly generated set of DNA and writes it to dna.txt. An alternative to this function is randseq from the Bioinformatics Toolbox that generates a random sequence of given length n; however, randseq cannot generate sequences with specified propotions of nucleotides.

#### gatherData.m

This function gathers the data for the subsequence matching analysis through a series of iterative steps. To generate a representative data set it utilizes the output from the generateDNA function for processing. The function loops a specified range (default is 5 to 20 bp) and fragments the DNA sequence based on the current value in that range. In other words, it generates consecutive overlapping k-mers for each of the values in that range. Using these fragments, the function uses the generateNonopTable function to generate pairwise matches of each of the k-mer sequences. For instance, if the function is currently generating 5-mer sequences, it will compare each of the 5-mer sequences with all of the other 5-mer sequences in the DNA sequence. For each k-mer, the generateNonopTable function stores a table objective with the k-mers and their respective matching values of either a logical TRUE or FALSE.

### gatherNonopTable.m

Given either the reference set of fragments, the query sequence, name or location of the reference DNA, the length of the subsequence, and the maximum possible score for the given query sequence, this function generates nonoptimized (overlapping sequences) data based on the given conditions. The function returns the data and writes it to data_nonop.csv.

## Machine Learning/Neural Netowork Solutions

#### Required virtual environment

Run this command to load the virtual environment
```bash
$ source venv/bin/activate
```

When running any of the following scripts, please use Python 3.7.

Pyenv might also be a good tool for managing python versions: https://github.com/pyenv/pyenv

#### Long-Short Term Memory (LSTM) Classification

The 04272020_subseq_match_class.py script performs a multidimensional LSTM classification on the subsequence data generated from the MATLAB data curation files.

#### Long-Short Term Memory (LSTM) Prediction

The 20200618_subseq_match_pred.py script performs a multidimensional LSTM preiction on the subsequence data generated from the MATLAB data curation files.

## NEUROMORPHIC SOLUTIONS TO HIGH DIMENSIONAL DNA SUBSEQUENCE MATCHING VIA K-MER WISE ALIGNMENT REPORT

In the this repository is the write up of the project. This paper gives background to the problem and the avenue by which I went to solve it. Also it explores the alternatives to the machine learning algorithms that were explored as well.
