"""Naive Bayes classification application
1. Pre-process text for training and testing
2. Train by creating feature vectors (bag of words)
3. Test by classifying remaining docs
"""

import os
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
        print("greater than 20: ")
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
    punctuation = [".", ",","/","\'","\"",";",":","?","!","-",">","<"]
    common = ["In", "in", "I", "you", "me", "they", "It", "it", "yes","no", "not", "can", "if", "and", "but", "or", "because", "is", "was", "be", "are", "were", "have"]
    symbols = ["|", "|",  ]

    output = []
    for word in text:
        # cut out long words, often just emails or other cruft
        if len(word) > 20:
            continue
        # cut out articles, preps, puncs, demonsts
        if word  in articles \
            or word in prepositions \
            or word in punctuation \
            or word in demonstratives \
            or word in common:
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

def securethebag(text, bag):
    for word in text:
        if word in bag:
            bag[word] += 1
        else:
            bag[word] = 1

    return bag

# multinomial classification strategy
def classify(trained_classes, file_bag):
    best = 0
    guess = ""
    for group in trained_classes:
        likelyhood = 1/20
        for word in file_bag:
            # if word in group.bag:
            #     likelyhood += file_bag[word]
            # else:
            #     likelyhood -= file_bag[word]
            if word in group.bag:
                likelyhood *= group.bag[word]/(len(group.bag)/1000)
            else:
                likelyhood *= 1/(len(group.bag)/10)
        # print(likelyhood)
            # if word in group.bag:
            #     likelyhood *= ((group.bag[word] + 1)/(group.count+len(group.bag)))**file_bag[word]
            # else:
            #     likelyhood *= (1/(group.count + len(group.bag)))**file_bag[word]
        if likelyhood > best:
            best = likelyhood
            guess = group.name
    return guess

# train the data
train_classes = []

# list each class directory
resource_dir = "./20_newsgroups"
dirs = os.listdir(resource_dir)
for dir in dirs:
    #list each file
    sum_words = 0
    class_bag = {}
    files = os.listdir(os.path.join(resource_dir, dir))
    #we only want half the files
    for i in range(int(len(files)/2)):
        text = filetotext(os.path.join(resource_dir,dir,files[i]))
        proc_text = preproc(text)
        sum_words += len(proc_text)
        class_bag = securethebag(proc_text, class_bag)
        # update training bag
    train_classes.append(ClassData(sum_words, class_bag, dir))
    # train_classes[-1].print_stats()

print("++++++++++++++++++ TRAINED +++++++++++++++++++")

# print(train_classes[0].name)
# print(train_classes[0].count)
# print(train_classes[0].bag)

total = 0
correct = 0
error = 0

guesses = {}

# file_bag = {}
# text = filetotext("./20_newsgroups/talk.religion.misc/84227")
# proc_text = preproc(text)
# class_bag = securethebag(proc_text, file_bag)
# guess = classify(train_classes, file_bag)
#
# print(guess)

# classify the remaining files
for dir in dirs:
    # print(dir)
    #list each file
    files = os.listdir(os.path.join(resource_dir, dir))
    #we only want the second half of the files
    for i in range(int(len(files)/2)):
        # print(os.path.join(resource_dir,dir,files[i + int((len(files)/2))]))
        total += 1
        file_bag = {}
        text = filetotext(os.path.join(resource_dir,dir,files[i + int((len(files)/2))]))
        proc_text = preproc(text)
        class_bag = securethebag(proc_text, file_bag)
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
print(correct)
print(error)
print(correct/total)
print(error/total)

print(guesses)
