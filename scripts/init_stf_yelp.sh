#!/usr/bin/env bash

DATA_DIR=$1

# Set up data directory
mkdir -p $DATA_DIR
cd $DATA_DIR

# Download word vectors
#wget http://nlp.stanford.edu/data/glove.6B.zip  # GloVe vectors
#unzip glove.6B.zip -d word_vectors

# Download expanded set of word vectors
mkdir word_vectors
cd word_vectors
wget https://worksheets.codalab.org/rest/bundles/0xa57f59ab786a4df2b86344378c17613b/contents/blob/ -O glove.6B.300d_yelp.txt
cd ..

# Download datasets into data directory
wget https://worksheets.codalab.org/rest/bundles/0x99d0557925b34dae851372841f206b8a/contents/blob/ -O yelp_dataset_large_split.tar.gz
mkdir yelp_dataset_large_split
tar xvf yelp_dataset_large_split.tar.gz -C yelp_dataset_large_split