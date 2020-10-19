#! /bin/bash

# Extract data set
echo "Extracting data set"
tar -xf 20_newsgroups.tar.gz

# Preprocess data
echo "Preprocessing data (15 seconds runtime)"
python bayes.py p

echo "Setup complete"
