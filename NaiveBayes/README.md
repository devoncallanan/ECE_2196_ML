# Homework 1

Create a Naive Bayes document classifier

## To run:
1. Unzip 20_newsgroups.tar.gz
`tar -xf 20_newsgroups.tar.gz`
2. Run the pre-processor
`python bayes.py p`
3. Run training and testing
`python bayes.py`

OR

run `bash setup.sh`
then `python bayes.py`


## Results

The preprocessing step removes from the corpus any word found in the stopwords.short file. Using the longer stopwords.long file did not add a noticeable performance boost while negatively affecting the runtime. Words are then stripped of punctuation and special characters, made all lowercase, and written back to file. Any word over 20 characters is removed. Very few english words surpass this limit, leaving only the longer email addresses and other cruft from headers left after the text is split along whitespace.

To implement the Naive Bayes classification, the class with the largest likelyhood is chosen for each testing file. Probabilities are a sum of the log of the word frequency in the class, multiplied by the frequency in the file to be classified. Laplace smoothing is used to prevent log(0) issues.

See function classify in bayes.py for algorithm.

Using the first half of the files in each folder, accuracy of 85.477% is achieved. Using the second half of the files in each folder, accuracy of 85.016% is achieved.
