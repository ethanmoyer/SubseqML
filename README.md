# VERTICALLY INTEGRATED PROJECT (VIP) REPORT: NEUROMORPHIC SOLUTIONS TO DNA SUBSEQUENCE AND RESTRICTION SITE ANALYSIS

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

#### Support Vector Machines (SVMs)

The svm.py and svm_pca.py scripts are responsible for conducting the SVMs analysis. 

Both scripts prompt the user for the location of both the training data set and the test data set. The prompt requires that the provided data sets be provided from the processed_data directory and contain the words 'train' and 'test' in them; otherwise, the script will exit. In the console, the script will print the resulting confusion matrix and classification report on the analysis. Both will save the test data set under the analysis_data directory. The two scripts will have different prefixes: "SVM_" and "SVM_PCA_" respectively.

In the casse of SVMs 2-component PCA, it will also display a graph of the test data ajusted to 2-components with PCA plotted against the support vector.

#### Random Forest

The random_forest.py script is responsible for conducting the random forest analysis.

The script promptd the user for the location of both the training data set and the test data set. The prompt requires that the provided data sets be provided from the processed_data directory and contain the words 'train' and 'test' in them; otherwise, the script will exit.

#### Convolution Neural Networks (CNNs)

The cnn.py script is responsible for conducting the CNNs analysis.

Before executing cnn.py, it's important to first set up tensorflow following this tutorial: 
https://www.tensorflow.org/install/pip

Then when running cnn.py, use Python version 3.7 like the following.
```bash
$ python3.7 cnn.py
```

Pyenv might also be a good tool for managing python versions: https://github.com/pyenv/pyenv

The script promptd the user for the location of the training data set, the test data set, and the validation data set. The prompt requires that the provided data sets be provided from the processed_data directory and contain the words 'train' in the trainining data set and the validation data set and 'test' in the test data set; otherwise, the script will exit.

## NEUROMORPHIC SOLUTIONS TO DNA SUBSEQUENCE AND RESTRICTION SITE ANALYSIS REPORT

In the this repository is the write up of the project. This paper gives background to the problem and the avenue by which I went to solve it. Also it explores the alternatives to the machine learning algorithms that were explored as well.

