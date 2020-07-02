# VERTICALLY INTEGRATED PROJECT (VIP) REPORT: NEUROMORPHIC SOLUTIONS TO DNA SUBSEQUENCE AND RESTRICTION SITE ANALYSIS

## Data Curation/Analysis

All the data preprocessed data used in the analysises are kept in the data directory. All data that is processed by using preprocessing.py is kepts in the processed_data directory.

### Matlab Pipeline

All of the Matlab code created to curate and cluster data is in the matlab-data-curation directory.

#### currateData.m

This function accepts the location of a table object (.csv) and creates two arrays of equal size: one contains only true outputs and the other contains only false outputs. The result is a new table object (.csv) with the two concatenated together with the prefix 'split_' saved under the ../data/ directory.

#### select_test_data.m

This function accepts the location of a table (.csv) with equally split true and false outputs prefixed with "split_", the required number of samples from the table, and the proportion of true samples to include in new sample. The result is a curated table (.csv) prefixed with the proporiton of true samples and the number of samples.

#### getMismatches.m

This function accepts the location of a resulting table (.csv) from the machine learing algoirthms prefixed with "examine_" and the name of the algorithm (either "SVM_", "RF_", or "CNN_"). It creates four arrays contining the different mismatches (true/true, true/false, false/true, and false/false). The result is a new table object (.csv) with the four concatenated together with the prefix 'mismatched_' saved under the ../data/ directory.

#### generateFeaturePlots.m

This function accepts the location of a mismatched table (.csv) prefixed with 'mismatched_' and calls data1DPlot based on the feature set of the table, generating mismatched distributions for each feature.

#### data1DPlot.m

This function accepts the location of mismatched tables (.csv) prefixed with 'mismatched_' and a feature from the data set and plots the distrubition of the feature across both true mismatches and false mismatches. These plots are saved to the ../feature/ directory prefixed with the technique used, whether it's mapping true or false, and the name of the feature.

## Data preprocessing

The processing.py script prompts the user for the location of both the training data set and the test data set. The prompt requires that the provided data sets be provided from the data directory and contain the words 'train' and 'test' in them; otherwise, the script will exit. Given that the correct data sets are provided from the correct directory, the script normalizes all of the data for each entry across all features and produces two normalized training and test sets in the processed_data directory.

## Machine Learning/Neural Netowork Solutions

#### Required installations

Run this command to install sklearn.
```bash
$ pip install sklearn
```
Run this command to install pandas.
```bash
$ pip install pandas
```

Run this command to install numpy.
```bash
$ pip install numpy
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

