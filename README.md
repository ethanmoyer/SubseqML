# SubseqML - Subsequence matching using Machine Learning

## Origin

This project started in Spring of 2020 during the end of my Freshmen year at Drexel University. Working with Professor Anup Das, I explored possible neuromorphic approaches for subsequence matching algorithms. 

### Approaches Overview

Before working through possible neuromorphic approaches, there was a stretch on time where we worked through how to best represent the data. In previous works that attempted to either classify DNA or build a regression model using deep learning models, DNA data was commonly split up using spanning windows that slide across a sequence. Professor Das and I eventually decided to use non-overlapping subsequences of various sizes in the format of time steps. In order to relate subsequences of DNA with time series-like data, we segmented the subsequences into k-mers.

We began our investigation with Long Short-Term Memory (LSTM) classification. After we defined the research problem and scope, I started to research LSTM and Recurrent Neural Networks (RNNs). With the data formatted as k-mers, the output space were simply logical values based on a 95% threshold of local alignment scores using the NUC44 scoring matrix in MATLAB. After weeks of reworking the data and running mock trials, we noticed a discrepancy in the distributions of False cases compared to True cases--the global percent of true cases was only about 0.1%. Due to the time series nature of the data, simply stratifying the samples to get a more equal distribution was not possible. 

From this point, I decided to use LSTM for regression instead of classification. Instead of labeling subsequences as 95% similar based on local alignment scores, the alignment scores were simply used as the output space of the features. Regardless of this shift in machine learning, LSTM proved to be too computationally intensive for the scope of the problem. This alone is what finally pushed our research from LSTM to more traditional methods like Convolution Neural Networks (CNNs).

Using a similar structure of data, a 1D Convolution Neural Network (CNN) was built to predict local alignment scores based on a two subsequences, each 20 base pairs (bp) long. The best accuracy that the model appeared to be able to reach was 60%. This indicates that using local alignment for smaller fragments is (surprisingly) not reliable for this type of problem. My guess is that longer subsequences would fair better because there are more degrees of freedom. 

Despite the difficulty in exploring this type of problem, three different approaches were heavily studied and analyzed under the scope of subsequence classification and regression.

## Data Curation/Analysis

All the preprocessed data used in the analysis are kept in the data directory. All data that is processed by using preprocessing.py is kepts in the processed_data directory.

### Matlab Pipeline

All of the MATLAB code created to curate and cluster data is in the matlab-data-curation directory.

### generateDNA.m

This function generates a dna molecule of length n with the appropriate proportions pa, pt, pc, and pg. The function returns the randomly generated set of DNA and writes it to dna.txt. An alternative to this function is randseq from the Bioinformatics Toolbox that generates a random sequence of given length n; however, randseq cannot generate sequences with specified proportions of nucleotides.

#### gatherData.m

This function gathers the data for the subsequence matching analysis through a series of iterative steps. To generate a representative data set it utilizes the output from the generateDNA function for processing. The function loops a specified range (default is 5 to 20 bp) and fragments the DNA sequence based on the current value in that range. In other words, it generates consecutive overlapping k-mers for each of the values in that range. Using these fragments, the function uses the generateNonopTable function to generate pairwise matches of each of the k-mer sequences. For instance, if the function is currently generating 5-mer sequences, it will compare each of the 5-mer sequences with all of the other 5-mer sequences in the DNA sequence. For each k-mer, the generateNonopTable function stores a table objective with the k-mers and their respective matching values of either a logical TRUE or FALSE.

### gatherNonopTable.m

Given either the reference set of fragments, the query sequence, name or location of the reference DNA, the length of the subsequence, and the maximum possible score for the given query sequence, this function generates non-optimized (overlapping sequences) data based on the given conditions. The function returns the data and writes it to data_nonop.csv.

## Machine Learning/Neural Network Solutions

#### Required virtual environment

Run this command to load the virtual environment
```bash
$ source venv/bin/activate
```

When running any of the following scripts, please use Python 3.7.

Pyenv might also be a good tool for managing python versions: https://github.com/pyenv/pyenv

#### Long-Short Term Memory (LSTM) Classification

The SubseqLSTMClass.py script performs a multidimensional LSTM classification on the subsequence data generated from the MATLAB data curation files.

#### Long-Short Term Memory (LSTM) Regression

The SubseqLSTMRegress.py script performs a multidimensional LSTM regression on the subsequence data generated from the MATLAB data curation files.

### Convolution Neural Network (CNN) Regression

The SubseqCNNRegress.py script performs a multidimensional CNN regression on the subsequence data generated from the MATLAB data curation files.

### NEUROMORPHIC SOLUTIONS TO HIGH DIMENSIONAL DNA SUBSEQUENCE MATCHING VIA K-MER WISE ALIGNMENT REPORT

In the this repository is the write up of the project after the first approach--LSTM classification. This paper gives background to the problem and the avenue by which I went to solve it. Also it explores the alternatives to the machine learning algorithms that were explored as well.
