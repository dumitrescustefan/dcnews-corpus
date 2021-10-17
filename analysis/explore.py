import os
import json
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from nltk.tokenize import word_tokenize
from sklearn.metrics.pairwise import cosine_similarity, cosine_distances


# 2. TF-IDF + SVD + Cosine
# code inspired from: https://stackoverflow.com/questions/41743402/svd-on-tfidf-matrix-returns-an-odd-shape
def fit_tfidf_model(corpus, max_features=50000):
    transformer = TfidfVectorizer(input=corpus,
                                  analyzer='word',
                                  max_features=max_features,
                                  ngram_range=(1, 3),
                                  sublinear_tf=False)
    return transformer.fit(corpus)


def transform_tfidf_model(model, corpus):
    matrix = model.transform(corpus)
    svd = TruncatedSVD(n_components=200)
    return svd.fit_transform(matrix)


def tokenize_split_join(raw_txt):
    return ' '.join(word_tokenize(raw_txt.replace('\n', ' ')))


def main(max_features=60000):
    # 1. Data Load
    print('Started loading data')
    if 'all.pkl' not in os.listdir('.'):
        print('Did not find pickled data. Reading from JSON file')
        with open('../all.json') as f:
            all_data = json.loads(f.read())
            all_data_df = pd.DataFrame.from_dict(all_data)
            all_data_df.to_pickle("./all.pkl")
    else:
        print('Found data already pickled.. reading that!')
        all_data_df = pd.read_pickle('all.pkl')

    # all_data = {'title': ['title1', 'title2'], 'text': ['text1', 'text2']}

    print('Done loading data')

    # combine title + text - this should be stored to disk to avoid computing it all the time
    print('Started creating corpuses')
    if 'corpus_both.pkl' not in os.listdir('.'):
        print('Did not find pickled title+text pickle. Creating it..')
        corpus_both_raw = all_data_df.title + ' ' + all_data_df.text
        corpus_both = corpus_both_raw.apply(tokenize_split_join)
        corpus_both.to_pickle('corpus_both.pkl')
        print('Done pickling both preprocessed title+text')
    else:
        print('Found pickled title+text pickle. Loading that!')
        corpus_both = pd.read_pickle('corpus_both.pkl')

    if 'corpus_title.pkl' not in os.listdir('.'):
        corpus_title = all_data_df.title.apply(tokenize_split_join)
        corpus_title.to_pickle('corpus_title.pkl')
        print('Done pickling preprocessed title')
    else:
        print('Found pickled title pickle. Loading that!')
        corpus_title = pd.read_pickle('corpus_title.pkl')

    if 'corpus_text.pkl' not in os.listdir('.'):
        corpus_text = all_data_df.text.apply(tokenize_split_join)
        corpus_text.to_pickle('corpus_text.pkl')
        print('Done pickling preprocessed text')
    else:
        print('Found pickled text pickle. Loading that!')
        corpus_text = pd.read_pickle('corpus_text.pkl')

    print('Done creating all corpuses')

    # create & fit model
    print('Started fitting model for max_features {}'.format(max_features))
    tfidf_model = fit_tfidf_model(list(corpus_both), max_features=max_features)
    print('Done fitting the model for {} max_features'.format(max_features))

    # create reduced vectors
    print('Started creating reduced vectors')
    reduced_vectors_titles = transform_tfidf_model(tfidf_model, corpus_title)
    reduced_vectors_texts = transform_tfidf_model(tfidf_model, corpus_text)
    print('Done creating reduced vectors')

    # compute similarity score
    print('Started computing similarity score for {} max_features'.format(max_features))
    all_data_df['sim_score'] = 0.0
    for cnt, (idx, row) in enumerate(all_data_df.iterrows()):
        sim_score = cosine_similarity([reduced_vectors_titles[cnt]], [reduced_vectors_texts[cnt]])
        all_data_df.at[idx, 'sim_score'] = sim_score
        if (cnt % 5000) == 0:
            print('Processed {} title/text pairs'.format(cnt))
    print('Done computing similarity score')

    all_data_df.to_pickle("./all_data_sim_score_max_features_{}.pkl".format(max_features))

    print('Done computing similarity score for {} max_features'.format(max_features))


def read_cmd_args():
    import argparse
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('--max-features', metavar='mf', type=int,
                        help='an integer for the max # of features')
    ret = parser.parse_args()
    return ret


args = read_cmd_args()
main(args.max_features)
