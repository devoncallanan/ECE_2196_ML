#! /bin/bash

# Extract data set
echo "Extracting data set"
tar -xf 20_newsgroups.tar.gz

# Preprocess data
echo "Preprocessing data"
python bayes.py p

echo "Setup complete"
