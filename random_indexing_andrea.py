import logging
import collections
import tqdm
import dill
import sklearn
import pickle
import copy
import argparse
import evaluation
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import scipy
import numpy

from sklearn.preprocessing import normalize
from utils import *
from collections import defaultdict
from tqdm import tqdm
from evaluation import *
from scipy import sparse

parser = argparse.ArgumentParser()

parser.add_argument('--ppmi', required = True, type = bool, help = 'Indicate whether PPMI correction should be applied or not')
parser.add_argument('--stopwords', required = True, type = bool, help = 'Indicate whether to remove stopwords using the NLTK stoplist or not')
parser.add_argument('--corpus_dir', required = True, type = str, help = 'Folder where the corpus file(s) is(are) contained')

### NON-COMPULSORY ARGUMENTS
parser.add_argument('--squared_ppmi', required = False, type = bool, help = 'Determine whether to square PPMI values or not', default = False)
parser.add_argument('--window_size', required = False, type = int, help = 'Indicates the window size to be used for collecting co-occurrences', default = 10)
parser.add_argument('--vector_size', required = False, type = int, help = 'Size of the word vectors', default = 1800)
parser.add_argument('--non_zero', required = False, type = int, help = 'Number of non-zero elements in the word vectors', default = 12)
parser.add_argument('--min_count', required = False, type = int, help = 'Sets the minimum word frequency threshold for considering words or not while collecting co-occurrences', default = 10)

args = parser.parse_args()

logging.basicConfig(format='%(asctime)s %(message)s', level = logging.INFO)

def train_sentence(args, word_cooccurrences, memory_vectors, final_vectors, reduced_vocabulary, sentence):

    sentence_length = len(sentence)

    for sentence_current_word_index, current_word in enumerate(sentence):

        vocabulary_current_word_index = reduced_vocabulary[current_word]
    
        if vocabulary_current_word_index != 0:

            lower_bound = max(0, sentence_current_word_index - args.window_size)
            upper_bound = min(sentence_length, sentence_current_word_index + args.window_size)
            window = [n for n in range(lower_bound, upper_bound) if n != sentence_current_word_index]
            #print([sentence[window] for n in window])

            for position in window:

                other_word = sentence[position] 
                other_word_index = reduced_vocabulary[other_word]

                if other_word_index != 0:

                    if args.ppmi == False:      
                        final_vectors[vocabulary_current_word_index] = final_vectors[vocabulary_current_word_index] + memory_vectors[other_word_index]
                        word_cooccurrences[vocabulary_current_word_index][other_word_index] += 1
                    else:
                        word_cooccurrences[vocabulary_current_word_index][other_word_index] += 1

    return word_cooccurrences, final_vectors

current_parameters = '{}_ppmi_{}_squared_ppmi_{}_stopwords_{}_window_{}_vector_{}_non_zero_{}_min_count_{}'.format(re.sub('^.+/', '', re.sub('/$', '', args.corpus_dir)), args.ppmi, args.squared_ppmi, args.stopwords, args.window_size, args.vector_size, args.non_zero, args.min_count)
#prova = Corpus('wiki_for_all/')
#prova = Corpus('bnc_for_all/')
prova = Corpus(args.corpus_dir)
#prova = Corpus('pnp/')

prova_corpus = ReducedVocabulary(args, prova)

redux = prova_corpus.reduced_vocabulary
vocab_with_counters = prova_corpus.counters_with_index

total_count = prova_corpus.total_count_words
#index2word = {v : k for k,v in redux.items() if v != 0}
#print('Now dumping the dictionary...')
#with open('FIXED_dumped_bnc_NO_LEMMA_TEST_WORDS_redux_100_min_stopwords_removed.pickle', 'wb') as r:
    #dill.dump(prova_corpus, r)

#redux = pickle.load(open('dumped_bnc_redux_10_min_stopwords_removed.pickle', 'rb'))
co_oc = defaultdict(lambda : defaultdict(int))

memory_vectors = {v : numpy.zeros(args.vector_size, dtype=int) for k,v in redux.items() if v != 0}
print('Now creating the vectors...')
for k in tqdm(memory_vectors.keys()):
    random_indices = numpy.random.choice(range(args.vector_size), args.non_zero, replace=False)
    for index, position in enumerate(random_indices):
        #memory_vectors[k][position] = 1
        if index < args.non_zero:
            memory_vectors[k][position] = 1
        else:
            memory_vectors[k][position] = -1

final_vectors = {k : v for k, v in memory_vectors.items()}

print('Now collecting word co-occurrences...')

for sentence in tqdm(prova):

    co_oc, final_vectors = train_sentence(args, co_oc, memory_vectors, final_vectors, redux, sentence)

def ppmi(matrix):
    """Return a ppmi-weighted CSR sparse matrix from an input CSR matrix."""
    logging.info('Weighing raw count CSR matrix via PPMI')
    words = sparse.csr_matrix(matrix.sum(axis=1))
    contexts = sparse.csr_matrix(matrix.sum(axis=0))
    total_sum = matrix.sum()
    # csr_matrix = csr_matrix.multiply(words.power(-1)) # #(w, c) / #w
    # csr_matrix = csr_matrix.multiply(contexts.power(-1))  # #(w, c) / (#w * #c)
    # csr_matrix = csr_matrix.multiply(total)  # #(w, c) * D / (#w * #c)
    csr_matrix = matrix.multiply(words.power(-1, dtype=float))\
                           .multiply(contexts.power(-1, dtype=float))\
                           .multiply(total_sum)
    csr_matrix.data = numpy.log2(csr_matrix.data)  # PMI = log(#(w, c) * D / (#w * #c))
    csr_matrix = csr_matrix.multiply(csr_matrix > 0)  # PPMI
    csr_matrix.eliminate_zeros()
    return csr_matrix

if args.ppmi == True:

    collection_ppmis = []
    print('Now PPMIing it all...')

    for index_one, dict_one in tqdm(co_oc.items()):

        for index_two, conditional_probability_not_normalized in dict_one.items():

            #probability_index_one = vocab_with_counters[index_one]/total_count
            probability_index_one = vocab_with_counters[index_one]
            #probability_index_two = vocab_with_counters[index_two]/total_count
            probability_index_two = vocab_with_counters[index_two]
            #conditional_probability_normalized = conditional_probability_not_normalized/total_count
            conditional_probability_normalized = conditional_probability_not_normalized
            log_ppmi = max(0, int(numpy.log2(conditional_probability_normalized/(probability_index_one*probability_index_two))))
            #log_ppmi = max(0, int(numpy.log2((conditional_probability_not_normalized/total_count)/((vocab_with_counters[index_one]/total_count)*(vocab_with_counters[index_two]/total_count)))))
            #log_ppmi = numpy.log2((conditional_probability/total_count)/((vocab_with_counters[index_one]/total_count)*(vocab_with_counters[index_two]/total_count)))
            collection_ppmis.append(log_ppmi)

            if args.squared_ppmi == True:
                log_ppmi = (log_ppmi + 1)**2
            else:
               log_ppmi = log_ppmi + 1

            for i in range(round(log_ppmi)):
            #for i in range(conditional_probability_not_normalized):
                final_vectors[index_one] = final_vectors[index_one] + memory_vectors[index_two]

            #ppmis[index_one][index_two] = log_ppmi
                #print(index2word[index_one])
                #print(index2word[index_two], '\n')
                #co_oc_final[index_one][index_two] = log_ppmi 

    #print('Now dumping the PPMI counter...')
    #with open('PPMI_counter_bnc_win_10_min_10_stopwords_removed.pickle', 'wb') as o:
        #dill.dump(ppmis, o)
    print('Now plotting the PPMI clusters for visualization...')
    clusters = range(max([int(round(n)) for n in collection_ppmis]))           
    values = defaultdict(int)

    for v in collection_ppmis:
        values[int(round(v))] += 1

    fig, ax = plt.subplots()
    ax.scatter([k for k, v in values.items()], [v for k, v in values.items()])
    #ax.scatter([1,2],[1,2])
    fig.savefig('RI_plots/{}.png'.format(current_parameters))


#print('Now dumping the co-oc PMIed dictionary...')
#with open('dumped_bnc_cooc_10_min_window_10_stopwords_removed_PPMIed.pickle', 'wb') as d:
    #dill.dump(co_oc_final,d) 

#print('Done!')
#for index_one, dict_one in tqdm(to_dump.items()):
    #for index_two, value_two in dict_one.items():
        #if value_two > 0: 
            #for n in range(value_two):
                #final_vectors[index_one] += memory_vectors[index_two]

#final_vectors_norm = {k : normalize(v) for k, v in final_vectors.items()}
#with open('final_vectors_NORM_10000_min.pickle', 'wb') as v:
    #dill.dump(final_vectors_norm, v)

with open('results/{}.txt'.format(current_parameters), 'w') as r:
    r.write('{}\n\n'.format(current_parameters))
    men, number_men_tested = MEN_scores(final_vectors, redux)
    import pdb; pdb.set_trace()
    r.write('MEN score:\t{}\nNumber of tested words: {}\n\n'.format(men, number_men_tested))
    simlex, number_simlex_tested = SimLex_scores(final_vectors, redux)
    r.write('SimLex999 score:\t{}\nNumber of tested words: {}\n\n'.format(simlex, number_simlex_tested))
    ws353_sim, ws353_rel = ws353_scores(final_vectors, redux)
    r.write('WS353 similarity score:\t{}\nNumber of tested words: {}\n\n'.format(ws353_sim[0], ws353_sim[1]))
    r.write('WS353 relatedness score:\t{}\nNumber of tested words: {}\n\n'.format(ws353_rel[0], ws353_rel[1]))

#print('Now dumping the vectors...')
#with open('pickles/{}_ppmi_{}_stopwords_{}_window_{}_vector_{}_non_zero_{}_min_count_{}'.format(, args.ppmi, args.stopwords, args.window_size, args.non_zero, args.min_count, 'wb') as v:
    #dill.dump([redux, final_vectors], v)
