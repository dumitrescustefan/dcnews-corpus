import os
import json
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from nltk.tokenize import word_tokenize
from sklearn.metrics.pairwise import cosine_similarity, cosine_distances


# TFIDF + SVD + Cosine
# code inspired from: https://stackoverflow.com/questions/41743402/svd-on-tfidf-matrix-returns-an-odd-shape


def fit_tfidf_model(corpus, max_features=50000, ngram_range_max=3, sublinear_tf=False):
    """ TF-IDF train """
    print('Started fitting model for max_features: "{}", '
          'ngram_range_max: "{}", sublinear_tf: "{}"'.format(max_features,
                                                             ngram_range_max,
                                                             sublinear_tf))
    transformer = TfidfVectorizer(input=corpus,
                                  analyzer='word',
                                  max_features=max_features,
                                  ngram_range=(1, ngram_range_max),
                                  sublinear_tf=sublinear_tf)
    ret = transformer.fit(corpus)
    print('Done fitting model for max_features: "{}", '
          'ngram_range_max: "{}", sublinear_tf: "{}"'.format(max_features,
                                                             ngram_range_max,
                                                             sublinear_tf))
    return ret


def transform_tfidf_model(model, corpus):
    """ TF-IDF transform + SVD """
    matrix = model.transform(corpus)
    svd = TruncatedSVD(n_components=200)
    return svd.fit_transform(matrix)


def tokenize_split_join(raw_txt):
    """ tokenize and remove \n chars """
    return ' '.join(word_tokenize(raw_txt.replace('\n', ' ')))


def main(max_features=60000, ngram_range_max=3, sublinear_tf=False):
    # 1. Load data
    all_data_df = load_data()

    # 2. Preprocess data
    corpus_both, corpus_text, corpus_title = preprocess_data(all_data_df)

    # 3. create & fit model
    tfidf_model = fit_tfidf_model(list(corpus_both),
                                  max_features=max_features,
                                  ngram_range_max=ngram_range_max,
                                  sublinear_tf=sublinear_tf)

    # 4. create reduced vectors
    print('Started creating reduced vectors')
    reduced_vectors_titles = transform_tfidf_model(tfidf_model, corpus_title)
    reduced_vectors_texts = transform_tfidf_model(tfidf_model, corpus_text)
    print('Done creating reduced vectors')

    # 5. compute similarity score
    print('Started computing similarity score for max_features: "{}", '
          'ngram_range_max: "{}", sublinear_tf: "{}"'.format(max_features,
                                                             ngram_range_max,
                                                             sublinear_tf))
    all_data_df_sim_score = compute_similarity_score(all_data_df, reduced_vectors_texts, reduced_vectors_titles)
    pickle_path = "./dcnews_max_features_{}_ngram_range_max_{}_sublinear_tf_{}.pkl".format(max_features, ngram_range_max, sublinear_tf)
    all_data_df_sim_score.to_pickle(pickle_path)
    print('Done computing similarity score for max_features: "{}", '
          'ngram_range_max: "{}", sublinear_tf: "{}"'.format(max_features,
                                                             ngram_range_max,
                                                             sublinear_tf))


def preprocess_data(all_data_df):
    """ combine title + text - this should be stored to disk to avoid computing it all the time """
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
    return corpus_both, corpus_text, corpus_title


def load_data():
    """ Data Load from pickled files """
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
    return all_data_df


def compute_similarity_score(all_data_df, reduced_vectors_texts, reduced_vectors_titles):
    """ Compute similarity score based on cosine similarity """
    all_data_df['sim_score'] = 0.0
    for cnt, (idx, row) in enumerate(all_data_df.iterrows()):
        # cosine similarity
        sim_score = cosine_similarity([reduced_vectors_titles[cnt]], [reduced_vectors_texts[cnt]])
        all_data_df.at[idx, 'sim_score'] = sim_score
        if (cnt % 5000) == 0:
            print('Processed {} title/text pairs'.format(cnt))
    return all_data_df


def read_cmd_args():
    import argparse
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('--max-features', metavar='mf', type=int, default=60000,
                        help='an integer for the max # of features')
    parser.add_argument('--ngram-range-max', metavar='nrm', type=int, default=3,
                        help='an integer for the max # of ngram range')
    parser.add_argument('--sublinear-tf', metavar='stf', type=bool, default=False,
                        help='a boolean specifying if to use sublinear_tf')
    return parser.parse_args()


args = read_cmd_args()
main(args.max_features, args.ngram_range_max, args.sublinear_tf)
