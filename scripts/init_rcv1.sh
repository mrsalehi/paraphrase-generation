#!/usr/bin/env bash

DATA_DIR=$1

NED_FILE=1tIfZFI3o4mdkKdRLeHm4m_fmJiYV64mn
WORD_VECTOR=1aXRoL421jbUqGcIdYabpxuMhdGcX324g

NED_FILENAME=rcv1_ned_unique.tar.bz2
WORD_VECTOR_FILENAME=glove.6B.300d_rcv1.txt

wget -O gdrive https://docs.google.com/uc?id=0B3X9GlR6EmbnQ0FtZmJJUXEyRTA&export=download
chmod a+x gdrive

./gdrive download $NED_FILE
./gdrive download $WORD_VECTOR

# Set up data directory
mkdir -p $DATA_DIR
mkdir -p $DATA_DIR/word_vectors

mv $NED_FILENAME $DATA_DIR
mv $WORD_VECTOR_FILENAME $DATA_DIR/word_vectors

cd $DATA_DIR

tar xvfj $NED_FILENAME
cp rcv1_split/test.tsv rcv1_split/valid.tsv