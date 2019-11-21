import os
import numpy
import collections
import sklearn
import pickle
import re
import scipy
import tqdm

from tqdm import tqdm
from scipy import stats
from sklearn.metrics.pairwise import cosine_similarity
from collections import defaultdict

def test_words():
    
    test_words = []

    MEN_file = open('tests/men.txt').readlines()
    test_words = []
    for evaluation in MEN_file:
        evaluation_no_pos = ((re.sub('_N|_V|_A|\n', '', evaluation)).split())[:2]
        for word in evaluation_no_pos:
            if word not in test_words:
                test_words.append(word)

    tests = ['similarity', 'relatedness']
    for test in tests:

        test_file = open('tests/wordsim353_{}_goldstandard.txt'.format(test)).readlines() 
        for evaluation in test_file:
            score = evaluation.strip('\n').split('\t')
            for word in score[:2]:
                if word not in test_words:
                    test_words.append(word)

    SimLex_file = open('tests/SimLex-999.txt').readlines()
    for evaluation in SimLex_file[1:]:
        score = (evaluation.split('\t'))[:2]
        for word in score:
            if word not in test_words:
                test_words.append(word)
                
    return test_words

def correlation_score(gold, prediction):
    
    if len(gold) != len(prediction):
        raise ValueError('Not the same amount of gold evaluations and predictions')
        
    gold = numpy.array(gold, dtype=numpy.double)
    prediction = numpy.array(prediction, dtype=numpy.double)

    return stats.pearsonr(gold, prediction)[0]

def sim_check_RI(test, space, vocabulary):

    print('original length of test: {}'.format(len(test)))

    gold_list = []
    predicted_list = []
    
    for evaluation in tqdm(test):

        #word1 = '{}'.format(evaluation[0])
        #word2 = '{}'.format(evaluation[1])
        word1 = '{}_test'.format(evaluation[0])
        word2 = '{}_test'.format(evaluation[1])
        golden_score = evaluation[2]

        if word1 in vocabulary.keys() and word2 in vocabulary.keys():

            if vocabulary[word1] != 0 and vocabulary[word2] != 0:

                vector_word1 = space[vocabulary[word1]].reshape(1,-1)
                vector_word2 = space[vocabulary[word2]].reshape(1,-1)

                predicted_similarity = cosine_similarity(vector_word1, vector_word2)
                print('{}\t{}\t{}'.format(word1, word2, predicted_similarity))

                if predicted_similarity != 0:
                        
                    gold_list.append(golden_score)
                    predicted_list.append(float(predicted_similarity))

    pearson_score = correlation_score(gold_list, predicted_list)

    return pearson_score, len(predicted_list)

def ws353_scores(space, vocabulary):

    tests = ['similarity', 'relatedness']
    results = []
    for test in tests:

        test_file = open('tests/wordsim353_{}_goldstandard.txt'.format(test)).readlines() 
        test_words = []
        for evaluation in test_file:
            score = evaluation.strip('\n').split('\t')
            test_words.append(score)
        current_test_results = sim_check_RI(test_words, space, vocabulary)
        results.append(current_test_results)

    return results

def SimLex_scores(space, vocabulary):

    SimLex_file = open('tests/SimLex-999.txt').readlines()
    test_words = []
    for evaluation in SimLex_file[1:]:
        score = (evaluation.split('\t'))[:4]
        del score[2]
        test_words.append(score)

    return sim_check_RI(test_words, space, vocabulary)

def MEN_scores(space, vocabulary):

    MEN_file = open('tests/men.txt').readlines()
    test_words = []
    for evaluation in MEN_file:
        evaluation_no_pos = (re.sub('_N|_V|_A|\n', '', evaluation)).split()
        test_words.append(evaluation_no_pos)

    return sim_check_RI(test_words, space, vocabulary)

#redux = pickle.load(open('dumped_redux_10000_min.pickle', 'rb'))
#redux = (pickle.load(open('dumped_bnc_redux_10_min_stopwords_removed_smoothed_1_squared.pickle', 'rb'))).reduced_vocabulary
#redux = (pickle.load(open('dumped_bnc_redux_10_min_stopwords_removed_smoothed_1_squared.pickle', 'rb')))
#vec = pickle.load(open('final_vectors_10000_min.pickle', 'rb'))
#vec = pickle.load(open('final_bnc_vectors_10_min_vector_size_1800_number_ones_12_window_10_stopwords_removed_PPMI_smoothed_1_squared.pickle', 'rb'))
#redux, vec = pickle.load(open('FIXED_final_bnc_NO_LEMMA_TEST_WORDS_vectors_10_min_vector_size_1800_number_ones_12_window_10_stopwords_removed_NO_PPMI.pickle', 'rb'))
#vec = pickle.load(open('FIXED_final_bnc_NO_LEMMA_TEST_WORDS_vectors_10_min_vector_size_1800_number_ones_12_window_10_stopwords_removed_PPMI_smoothed_1_squared.pickle', 'rb'))[1]
#vec = pickle.load(open('final_vectors_10000_min_vector_size_512.pickle', 'rb'))
#MEN_scores(vec, redux)
#SimLex_scores(vec, redux)
#ws353_scores(vec, redux)
