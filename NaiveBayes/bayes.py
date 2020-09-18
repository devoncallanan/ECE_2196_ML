"""Naive Bayes classification application
1. Pre-process text for training and testing
2. Train by creating feature vectors (bag of words)
3. Test by classifying remaining docs
"""

import os
import sys
import math

class ClassData:
    def __init__(self, word_count, bag, name):
        self.count = word_count
        self.bag = bag
        self.name = name

    def print_stats(self):
        print("----------")
        print("----------")
        print("name: " + str(self.name))
        print("COUNT: " + str(self.count))
        print("greater than 1000: ")
        for word in self.bag:
            if self.bag[word] > 1000:
                print(word + " \t\t " + str(self.bag[word]))

def preproc(text):
    articles = ["The", "the", "A", "a", "An", "an"]
    prepositions = ["aboard","about","above","across","after","against",
    "along","amid","among","anti","around","as","at","before","behind",
    "below","beneath","beside","besides","between","beyond","but","by",
    "concerning","considering","despite","down","during","except","excepting",
    "excluding","following","for","from","in","inside","into","like","minus",
    "near","of","off","on","onto","opposite","outside","over","past","per",
    "plus","regarding","round","save","since","than","through","to","toward",
    "towards","under","underneath","unlike","until","up","upon","versus","via",
    "with","within","without"]
    demonstratives = ["here", "there", "this", "that","these", "those"]
    punctuation = [".", ",","/","\'","\"",";",":","?","!","-",">","<", "(",")"]
    symbols = ["|", "|" ]
    # stopwords from https://www.ranks.nl/stopwords
    stopwords = open("./stopwords.short").read().split()

    output = []
    for word in text:
        word = word.lower()
        # cut out long words, often just emails or other cruft
        if len(word) > 20:
            continue
        # cut out articles, preps, puncs, demonsts
        if word  in articles \
            or word in prepositions \
            or word in punctuation \
            or word in demonstratives \
            or word in stopwords:
            continue
        # add the word to the ouput, stripping various symbols
        output.append(word.strip("".join(punctuation)))

    return output

def filetotext(filename):
    file = open(filename, "r", encoding="latin-1")
    try:
        contents = file.read()
    except:
        print(filename)
    text = contents.split()

    return text

def texttofile(filename, data):
    file = open(filename, "w", encoding="latin-1")
    try:
        file.write(" ".join(data))
    except:
        print(filename)

def securethebag(text, bag):
    for word in text:
        if word in bag:
            bag[word] += 1
        else:
            bag[word] = 1

    return bag

# Naive Bayes
def classify(trained_classes, file_bag):
    best = -math.inf
    guess = ""
    for group in trained_classes:
        likelyhood = 1/20
        for word in file_bag:
            if word in group.bag:
                likelyhood += math.log((group.bag[word] + 1)/(group.count))*file_bag[word]
            else:
                likelyhood += math.log(1/(group.count))*file_bag[word]
        if likelyhood > best:
            best = likelyhood
            guess = group.name
    return guess

####### main ########
mode = "r"
resource_dir = "./20_newsgroups"
train_dir = "./train_data"
test_dir = "./test_data"
# If an argument was passed in
if len(sys.argv) > 1:
    # assume it is p (preprocess)
    if sys.argv[1] == "p":
        mode = "p"

if mode == "p":
    try:
        os.mkdir(train_dir)
    except FileExistsError as err:
        print("directory train_data exists, consider removing old data dirs")
    try:
        os.mkdir(test_dir)
    except FileExistsError as err:
        print("directory test_data exists, consider removing old data dirs")

    dirs = os.listdir(resource_dir)
    for dir in dirs:
        try:
            os.mkdir(os.path.join(train_dir,dir))
        except FileExistsError as err:
            print("directory train_data/"+ dir +" exists, consider removing old data dirs")
        try:
            os.mkdir(os.path.join(test_dir,dir))
        except FileExistsError as err:
            print("directory test_data/"+ dir +" exists, consider removing old data dirs")
        #list each file
        files = os.listdir(os.path.join(resource_dir, dir))
        #we only want half the files
        for i in range(len(files)):
            text = filetotext(os.path.join(resource_dir,dir,files[i]))
            proc_text = preproc(text)
            # training set from first half of the files, testing for second half
            if i > len(files)/2:
                texttofile(os.path.join(train_dir,dir,files[i]), proc_text)
            else:
                texttofile(os.path.join(test_dir,dir,files[i]), proc_text)
else:

    available_dirs = os.listdir("./")
    if "train_data" not in available_dirs or "test_data" not in available_dirs:
        print("run `python bayes.py p` to run preprocess step and generate training and testing data")
        quit()
    print("++++++++++++++++++ TRAINING +++++++++++++++++++")
    # train the data
    train_classes = []

    # list each class directory
    dirs = os.listdir(train_dir)
    for dir in dirs:
        #list each file
        sum_words = 0
        class_bag = {}
        files = os.listdir(os.path.join(train_dir, dir))
        for i in range(int(len(files))):
            text = filetotext(os.path.join(train_dir,dir,files[i]))
            sum_words += len(text)
            class_bag = securethebag(text, class_bag)
            # update training bag
        train_classes.append(ClassData(sum_words, class_bag, dir))
        # train_classes[-1].print_stats()

    print("++++++++++++++++++ TRAINED +++++++++++++++++++")

    print("++++++++++++++++++ TESTING +++++++++++++++++++")

    total = 0
    correct = 0
    error = 0

    guesses = {}

    dirs = os.listdir(test_dir)
    # classify the remaining files
    for dir in dirs:
        #list each file
        files = os.listdir(os.path.join(test_dir, dir))
        for i in range(int(len(files))):
            total += 1
            file_bag = {}
            text = filetotext(os.path.join(test_dir,dir,files[i]))
            class_bag = securethebag(text, file_bag)
            guess = classify(train_classes, file_bag)
            if guess in guesses:
                guesses[guess] += 1
            else:
                guesses[guess] = 1
            # print(guess)
            if guess == dir:
                correct += 1
                # print(guess)
            else:
                error += 1

    print("+++++++++++++++++++++ CLASSIFIED +++++++++++++++++++++++")
    # print(correct)
    # print(error)
    print("Correct: " + str(((correct/total))*100) + "%")
    print("Incorrect: " + str(((error/total))*100) + "%")

    # print(guesses)
