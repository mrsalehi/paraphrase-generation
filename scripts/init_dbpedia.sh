#!/usr/bin/env bash

DATA_DIR=$1

DATASET_URL="http://fs2.filegir.com/ub_maka/ned_more_sim.tar.bz2"
WORD_VECTOR_URL="http://fs2.filegir.com/ub_maka/glove.6B.300d_dbpedia_cut.txt.tar.bz2"

# Set up data directory
mkdir -p $DATA_DIR
cd $DATA_DIR

# Download word vectors
#wget http://nlp.stanford.edu/data/glove.6B.zip  # GloVe vectors
#unzip glove.6B.zip -d word_vectors

# Download expanded set of word vectors
mkdir word_vectors
cd word_vectors
wget $WORD_VECTOR_URL -O glove.6B.300d_dbpedia.tar.bz2
tar xvfj glove.6B.300d_dbpedia.tar.bz2
cd ..

# Download datasets into data directory
wget $DATASET_URL -O ned_more_sim.tar.bz2
mkdir dbpedia_split
tar xvfj ned_more_sim.tar.bz2 -C dbpedia_split