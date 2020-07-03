# -*- coding: utf-8 -*-
# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:light
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.3.2
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# # Recommendations with Document Similarity
#
# Recommender systems are one of the popular and most adopted applications of machine learning. They are typically used to recommend entities to users and these entites can be anything like products, movies, services and so on. 
#
# Popular examples of recommendations include,
# - Amazon suggesting products on its website
# - Amazon Prime, Netflix, Hotstar recommending movies\shows
# - YouTube recommending videos to watch
#
# Typically recommender systems can be implemented in three ways:
#
# - Simple Rule-based Recommenders: Typically based on specific global metrics and thresholds like movie popularity, global ratings etc.
# - Content-based Recommenders: This is based on providing similar entities based on a specific entity of interest. Content metadata can be used here like movie descriptions, genre, cast, director and so on
# - Collaborative filtering Recommenders: Here we don't need metadata but we try to predict recommendations and ratings based on past ratings of different users and specific items.
#
# We will be building a book recommendation system here where based on data\metadata pertaining to different books, we try and recommend similar books of interest!
#
# Since our focus in not really recommendation engines but NLP, we will be leveraging the text-based metadata. This falls under content-based recommenders. 
#

# # Load Dataset

# +
import pandas as pd

df = pd.read_csv('data/book_data.csv')
df.info()
# -

df.head()

df['book_desc'][67]

df = df[['book_authors', 'book_desc', 'book_title', 'genres', 'book_rating', 'book_rating_count']]
df = df[df['book_desc'].notna()]
df = df.reset_index(drop=True)
df.info()

df

# # Build a Recommender System
#
# Here you will build your own movie recommender system. We will use the following pipeline:
# - Text pre-processing
# - Feature Engineering
# - Document Similarity Computation
# - Find top similar movies
# - Build a movie recommendation function
#
#
# ## Document Similarity
#
# Recommendations are about understanding the underlying features which make us favour one choice over the other. Similarity between items(in this case movies) is one way to understanding why we choose one movie over another. There are different ways to calculate similarity between two items. One of the most widely used measures is __cosine similarity__ which we have already used in the previous unit.
#
# ### Cosine Similarity
#
# Cosine Similarity is used to calculate a numeric score to denote the similarity between two text documents. Mathematically, it is defined as follows:
#
# $$ cosine(x,y) = \frac{x. y^\intercal}{||x||.||y||} $$

# ## Text pre-processing
#
# We will do some basic text pre-processing on our movie descriptions before we build our features

# +
import nltk
import re
import numpy as np

stop_words = nltk.corpus.stopwords.words('english')

def normalize_document(doc):
    # lower case and remove special characters\whitespaces
    doc = re.sub(r'[^a-zA-Z0-9\s]', '', doc, re.I|re.A)
    doc = doc.lower()
    doc = doc.strip()
    # tokenize document
    tokens = nltk.word_tokenize(doc)
    # filter stopwords out of document
    filtered_tokens = [token for token in tokens if token not in stop_words]
    # re-create document from filtered tokens
    doc = ' '.join(filtered_tokens)
    return doc

normalize_corpus = np.vectorize(normalize_document)

norm_corpus = normalize_corpus(list(df['book_desc']))
len(norm_corpus)
# -

# ### only the description is normalised here and used to recommend books based on their descriptive similarity

# ## Extract TF-IDF Features (unigrams and bigrams)

# +
from sklearn.feature_extraction.text import TfidfVectorizer

tf = TfidfVectorizer(ngram_range=(1, 2), min_df=2)
tfidf_matrix = tf.fit_transform(norm_corpus)
tfidf_matrix.shape
# -

pd.DataFrame(data=tfidf_matrix.toarray())

# ###  can include other information by processing to numerical values and merging genre, author, book rating, etc. to tfidf_matrix.

# ## Compute Pairwise Document Similarity

# +
from sklearn.metrics.pairwise import cosine_similarity

doc_sim = cosine_similarity(tfidf_matrix)
doc_sim_df = pd.DataFrame(doc_sim)
doc_sim_df.head()
# -

# Thus, we should end up getting an N x N matrix where N is equal to the number of books, which is 52970. 

# ## Get List of Book Titles

title_list = df['book_title'].values

# ## Look for books by popularity

df.sort_values(by='book_rating_count', ascending=False)['book_title'].unique()[500:550]

# ## Find Top Similar Book for a Sample Book
#

# #### Find Book ID

# +
book_idx = np.where(title_list == 'Harry Potter and the Sorcerer\'s Stone')[0][0]
book_idx = np.where(title_list == 'Jane Eyre')[0][0]

title = 'Macbeth'
book_idx = np.where(title_list == title)[0][0]
book_idx
# -

# #### Get Book similarities

book_similarities = doc_sim_df.iloc[book_idx].values
book_similarities

# #### Get top 5 similar book IDs

similar_book_idxs = np.argsort(-book_similarities)[1:6]
similar_book_idxs

# #### Get top 5 similar books

df.iloc[similar_book_idxs]

# #### we don't want it to recommend same book (there are loads of duplicates in the data, let's save time by not deleting duplicates). we'll filter books that don't have the same title, and those that haven't appeared yet

# +
no_recs = 5

similar_book_idxs = np.argsort(-book_similarities)
position = 1
count = 0
similar_books = []
similar_ids = []
while count < no_recs:
    if title_list[similar_book_idxs[position]] == title:
        position = position + 1
    else:
        if title_list[similar_book_idxs[position]] in similar_books:
            position = position + 1
        else:
            similar_ids.append(similar_book_idxs[position])
            similar_books.append(title_list[similar_book_idxs[position]])
            count = count + 1
            position = position + 1
        
similar_books, similar_ids
# -

# ####  Let's check the description to see if they sound similar to the selected title! 

df.iloc[similar_ids][['book_authors', 'book_desc', 'book_title']]


# ### Build a book recommender function to recommend top 5 similar books for any book 
#
# The book title, book title list and document similarity matrix dataframe will be given as inputs to the function

# as there are duplicate titles in the data, this function will get the book_id with the largest book_rating_count.
def get_id(title):
    hold = df[df['book_title'] == title]
    hold = hold.sort_values(by='book_rating_count', ascending=False)
    book_id = hold.iloc[0].name
    return book_id


get_id(title)


def book_recommender(no_recs, title, book_sims = doc_sim_df):

    book_idx = get_id(title)
    #     book_idx = np.where(title_list == title)[0][0]
    book_similarities = doc_sim_df.iloc[book_idx].values

    similar_book_idxs = np.argsort(-book_similarities)
    position = 1
    count = 0
    similar_books = []
    similar_ids = []
    while count < no_recs:
        if title_list[similar_book_idxs[position]] == title:
            position = position + 1
        else:
            if title_list[similar_book_idxs[position]] in similar_books:
                position = position + 1
            else:
                similar_ids.append(similar_book_idxs[position])
                similar_books.append(title_list[similar_book_idxs[position]])
                count = count + 1
                position = position + 1

    return similar_books, similar_ids


# +
pd.set_option('display.max_colwidth', -1)

title = 'Freakonomics: A Rogue Economist Explores the Hidden Side of Everything (Freakonomics, #1)'
title = 'Hobbit'
title = 'Jane Eyre'

rec_books, rec_ids = book_recommender(5, title)
print('Book:', title)
results = df.iloc[rec_ids][['book_authors', 'book_desc', 'book_title']]
results
# -





# ## Get Popular Movie Recommendations

popular_books = ['Emma', 'The Girl with the Dragon Tattoo', 'Lord of the Flies',
                 'Harry Potter and the Chamber of Secrets', 'Les Misérables',
                'The Catcher in the Rye', 'Of Mice and Men']

for title in popular_books:
    rec_books, rec_ids = book_recommender(5, title)
    print('Book:', title)
    display(df.iloc[get_id(title)])
    results = df.iloc[rec_ids][['book_authors', 'book_desc', 'book_title']]
    display(results)


# # BM25 Similarity

# There are several techniques that are quite popular in information retrieval and search engines, including PageRank and Okapi BM25. The term BM stands for best matching. This technique is also known just as BM25.
#
#
# There are several steps that we must go through to successfully implement and compute BM25 scores for documents:
# 1.	
# Calculate frequencies of terms in documents and in corpus.
#
#  
# 2.	
# Compute the inverse document frequencies of terms.
#
#  
# 3.	
# Get bag of words-based features for corpus documents and query documents.
#
#  
# 4.	
# Build a function to compute the BM25 score of a given document in relation to a specific document from the corpus.
#
#  
# 5.	
# Build a function that leverages the function from Step 4, which computes and returns BM25 scores of a given document in relation to every other document in the corpus (like a vector of similarities for each document).
#
#  
# 6.	
# Build a function that returns pairwise BM25 similarity scores (weights) for all the documents in the corpus (leverages the function from Step 5).
#
#  
# The code we implement here has actually been adopted from the Gensim framework and we definitely recommend it if you are interested in leveraging BM25 similarity. We show the internals of the similarity framework so you can correlate it with the earlier defined concepts. The following class helps implement all the components from Steps 1 through 5 in our defined workflow.
#
#

# +
"""
Data:
-----
.. data:: PARAM_K1 - Free smoothing parameter for BM25.
.. data:: PARAM_B - Free smoothing parameter for BM25.
.. data:: EPSILON - Constant used for negative idf of document in corpus.
"""

import math
from six import iteritems
from six.moves import xrange

PARAM_K1 = 2.5
PARAM_B = 0.85
EPSILON = 0.2

class BM25(object):
    """Implementation of Best Matching 25 ranking function.
    Attributes
    ----------
    corpus_size : int
        Size of corpus (number of documents).
    avgdl : float
        Average length of document in `corpus`.
    corpus : list of list of str
        Corpus of documents.
    f : list of dicts of int
        Dictionary with terms frequencies for each document in `corpus`. Words used as keys and frequencies as values.
    df : dict
        Dictionary with terms frequencies for whole `corpus`. Words used as keys and frequencies as values.
    idf : dict
        Dictionary with inversed terms frequencies for whole `corpus`. Words used as keys and frequencies as values.
    doc_len : list of int
        List of document lengths.
    """

    def __init__(self, corpus):
        """
        Parameters
        ----------
        corpus : list of list of str
            Given corpus.
        """
        self.corpus_size = len(corpus)
        self.avgdl = sum(float(len(x)) for x in corpus) / self.corpus_size
        self.corpus = corpus
        self.f = []
        self.df = {}
        self.idf = {}
        self.doc_len = []
        self.initialize()

    def initialize(self):
        """Calculates frequencies of terms in documents and in corpus. Also computes inverse document frequencies."""
        for document in self.corpus:
            frequencies = {}
            self.doc_len.append(len(document))
            for word in document:
                if word not in frequencies:
                    frequencies[word] = 0
                frequencies[word] += 1
            self.f.append(frequencies)

            for word, freq in iteritems(frequencies):
                if word not in self.df:
                    self.df[word] = 0
                self.df[word] += 1

        for word, freq in iteritems(self.df):
            self.idf[word] = math.log(self.corpus_size - freq + 0.5) - math.log(freq + 0.5)

    def get_score(self, document, index, average_idf):
        """Computes BM25 score of given `document` in relation to item of corpus selected by `index`.
        Parameters
        ----------
        document : list of str
            Document to be scored.
        index : int
            Index of document in corpus selected to score with `document`.
        average_idf : float
            Average idf in corpus.
        Returns
        -------
        float
            BM25 score.
        """
        score = 0
        for word in document:
            if word not in self.f[index]:
                continue
            idf = self.idf[word] if self.idf[word] >= 0 else EPSILON * average_idf
            score += (idf * self.f[index][word] * (PARAM_K1 + 1)
                      / (self.f[index][word] + PARAM_K1 * (1 - PARAM_B + PARAM_B * self.doc_len[index] / self.avgdl)))
        return score

    def get_scores(self, document, average_idf):
        """Computes and returns BM25 scores of given `document` in relation to
        every item in corpus.
        Parameters
        ----------
        document : list of str
            Document to be scored.
        average_idf : float
            Average idf in corpus.
        Returns
        -------
        list of float
            BM25 scores.
        """
        scores = []
        for index in xrange(self.corpus_size):
            score = self.get_score(document, index, average_idf)
            scores.append(score)
        return scores


def get_bm25_weights(corpus):
    """Returns BM25 scores (weights) of documents in corpus.
    Each document has to be weighted with every document in given corpus.
    Parameters
    ----------
    corpus : list of list of str
        Corpus of documents.
    Returns
    -------
    list of list of float
        BM25 scores.
    Examples
    --------
    >>> from gensim.summarization.bm25 import get_bm25_weights
    >>> corpus = [
    ...     ["black", "cat", "white", "cat"],
    ...     ["cat", "outer", "space"],
    ...     ["wag", "dog"]
    ... ]
    >>> result = get_bm25_weights(corpus)
    """
    bm25 = BM25(corpus)
    average_idf = sum(float(val) for val in bm25.idf.values()) / len(bm25.idf)

    weights = []
    for doc in corpus:
        scores = bm25.get_scores(doc, average_idf)
        weights.append(scores)

    return weights


# -

norm_corpus_tokens = np.array([nltk.word_tokenize(doc) for doc in norm_corpus])
norm_corpus_tokens[:3]

# %%time
wts = get_bm25_weights(norm_corpus_tokens)

bm25_wts_df = pd.DataFrame(wts)
bm25_wts_df.head()

title = 'Jane Eyre'
rec_books, rec_ids = book_recommender(5, title, book_sims = bm25_wts_df)
print('Book:', title)
results = df.iloc[rec_ids][['book_authors', 'book_desc', 'book_title']]
results


