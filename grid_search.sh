#!/bin/bash

PPMIS=('True' 'False')
SQUARED_PPMI=('True' 'False')
STOPWORDS=('True' 'False')
MIN_COUNT=(10 25 50 100 0)
VECTOR_SIZE=(1800 300 600 1000 1400)
NON_ZERO=(12 18 24 6)

for p in ${PPMIS[@]};
do
    for sq in ${SQUARED_PPMI[@]};
    do
        for s in ${STOPWORDS[@]};
        do
            for m in ${MIN_COUNT[@]};
            do
                for v in ${VECTOR_SIZE[@]};
                do
                    for z in ${NON_ZERO[@]};
                    do
                        python random_indexing_andrea.py --corpus_dir ../dataset/bnc_stemmed_for_bert/ --ppmi ${p} --stopwords ${s} --min_count ${m} --vector_size ${v} --non_zero ${z} --squared_ppmi ${sq} 
#&
                    done 
                #wait
                done
            done
        done
    done
done
