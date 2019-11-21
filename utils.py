import numpy
import collections
#import spacy
import math
import tqdm
import os
import re
import logging
import argparse
import pickle
import scipy
import nltk

from scipy.sparse import csr_matrix, load_npz
from collections import defaultdict
from numpy import dot, sum
from numpy.linalg import norm
from math import sqrt
from tqdm import tqdm
from nltk.corpus import stopwords

'''
def normalise(vector):
    norm_vector = norm(vector)
    if norm_vector == 0:
        return vector
    vector = vector / norm_vector
    #print(sum([i*i for i in v]))
    return vector

#def norm(value):
    #v=float(value)
    #norm=v / sqrt((sum(v**2)))
    #return float(norm)

def cosine_similarity(vector_1, vector_2): 
    if len(vector_1) != len(vector_2):
        raise ValueError('Vectors must be of same length')
    vector_1 = numpy.squeeze(vector_1)
    vector_2 = numpy.squeeze(vector_2)
    denominator_a = numpy.dot(vector_1, vector_1)
    denominator_b = numpy.dot(vector_2, vector_2)
    denominator = math.sqrt(denominator_a) * math.sqrt(denominator_b)
    if float(denominator) == 0.0:
        cosine_similarity = 0.0
    else:
        cosine_similarity = dot(vector_1, vector_2) / denominator
    return cosine_similarity
'''

def top_similarities(query, vocabulary, word_vectors, number_similarities):
    
    test_index=vocabulary[query]
    similarities_dictionary = defaultdict(float)

    for other_item, other_index in vocabulary.items():
        character_sim=cosine_similarity(word_vectors[test_index], word_vectors[other_index])
        if character_sim > 0.01:
            similarities_dictionary[other_item] = character_sim

    similarities_dictionary = [sim for sim in sorted(similarities_dictionary.items(), key = lambda kv:(kv[1], kv[0]), reverse=True)][: number_similarities]
    print(similarities_dictionary)

class ReducedVocabulary:
    def __init__(self, args, corpus):

        self.minimum_count = args.min_count
        self.corpus = corpus
        if args.stopwords == True:
            self.stopwords = set(stopwords.words('english'))
        else:
            self.stopwords = []
        
        self.reduced_vocabulary, self.counters_with_index, self.total_count_words = self.trim(args)

    def count(self):
        logging.info('Counting the words in the corpus, so as to trim the vocabulary size')
        self.word_counters = defaultdict(int)
        for line in tqdm(self.corpus):
            for word in line:
                self.word_counters[word] += 1

    def trim(self, args):
        self.count()
        logging.info('Trimming the vocabulary')
        counters_with_index = defaultdict(int)
        reduced_vocabulary = defaultdict(int)
        total_count_words = 0
        for word, frequency in tqdm((self.word_counters).items()):
            total_count_words += 1
            if frequency >= self.minimum_count and word not in self.stopwords:
            #if frequency > self.minimum_count:
                #word_cleaned = re.sub('_test', '', word)
                reduced_vocabulary[word] = len(reduced_vocabulary.keys()) + 1
                counters_with_index[reduced_vocabulary[word]] = self.word_counters[word]
            elif frequency < self.minimum_count and '_test' in word:
                #word_cleaned = re.sub('_test', '', word)
                reduced_vocabulary[word] = len(reduced_vocabulary.keys()) + 1
                counters_with_index[reduced_vocabulary[word]] = self.word_counters[word]
            else:
                reduced_vocabulary[word] = 0

        return reduced_vocabulary, counters_with_index, total_count_words

class Corpus(object):
    def __init__(self, filedir):
        self.filedir = filedir
        self.files = [os.path.join(root, name) for root, dirs, files in os.walk(filedir) for name in files if '.txt' in name]
        print(self.files)
        
    def __iter__(self):

        for individual_file in self.files: 
            training_lines = open(individual_file).readlines()
            for line in tqdm(training_lines):
                line = line.strip().lower().split()
                yield line
