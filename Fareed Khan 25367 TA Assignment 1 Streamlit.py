# ________________________________________ IMPORTING LIBRARIES ________________________________________ #

import streamlit as st
import pandas as pd
import numpy as np
import nltk
import re
import itertools
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from nltk.stem.porter import PorterStemmer
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from gensim.models import KeyedVectors, Word2Vec
import gensim 
import logging
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import KMeans, AgglomerativeClustering, MiniBatchKMeans, MeanShift
from sklearn.decomposition import TruncatedSVD
import pickle

# ________________________________________ LOADING DATA AND CLEANING IT ________________________________________ #

data = pd.read_csv('bbc_news.csv', usecols=['title','description'])
data.drop_duplicates(inplace=True)
data.dropna(inplace=True)

def cleaning(s):
    s = str(s)
    s = re.sub('[!@#$_]', '', s)
    s = s.replace("co","")
    s = s.replace("https","")
    s = s.replace("[\w*"," ")
    s = s.replace('<.*?>', '')
    s = s.replace('strong>', '')
    s = s.replace('\x92', '')
    return s

data['title'] = data['title'].apply(cleaning)
data['description'] = data['description'].apply(cleaning)


# ________________________________________ TF-IDF VECTORIZER (BAG OF WORDS BEST) ________________________________________ #

with open('modelstest/bag_of_words_best/vectorizer.pkl', 'rb') as f:
    vectorizer_tfidf = pickle.load(f)
with open('modelstest/bag_of_words_best/kmeans.pkl', 'rb') as f:
    kmeans_tfidf = pickle.load(f)
# Transform the corpus data into a TF-IDF matrix
data_tfidf = vectorizer_tfidf.transform(data['title'])
# Define a function to find the most similar documents in a cluster
def find_similar_documents_tfidf(new_text, vectorizer, kmeans, data_tfidf, data):
    # Transform the new text and predict its cluster
    new_text_transformed = vectorizer.transform([new_text])
    cluster = kmeans.predict(new_text_transformed)[0]
    # Get the indices of documents in the same cluster
    cluster_indices = (kmeans.labels_ == cluster).nonzero()[0]
    # Calculate cosine similarity between the new text and all documents in the cluster
    similarities = cosine_similarity(new_text_transformed, data_tfidf[cluster_indices]).ravel()
    # Sort the documents by similarity score and return the top three
    most_similar_indices = similarities.argsort()[::-1][:1]
    output_tf = data.iloc[cluster_indices[most_similar_indices]]['title']
    output_tf = output_tf.iloc[0]
    return output_tf


# ________________________________________ PRE-TRAINED WORD2VEC VECTORIZER (WORD2VEC BEST) ________________________________________ #

with open('modelstest/word2vec_best/pretrain_word2vec_assign.pkl', 'rb') as f:
    cluster_data = pickle.load(f)
with open('modelstest/word2vec_best/kmeans_word2vec.pkl', 'rb') as f:
    kmeans_word2vec_pre = pickle.load(f)


def find_similar_documents_word2vec(new_text, kmeans, cluster_data):

    model_W2V = KeyedVectors.load_word2vec_format('GoogleNews-vectors-negative300.bin',binary=True, limit=50000,)

    def sentence_vector(sentence, word_vectors):
        words = sentence.lower().split()
        vectors = []
        for word in words:
            if word in word_vectors:
                vectors.append(word_vectors[word])
            if not vectors:
                vectors.append(np.zeros(300))
        return np.mean(vectors, axis=0)

    new_text_transformed = sentence_vector(new_text, model_W2V)
    cluster = kmeans.predict([new_text_transformed])[0]
    print(cluster)
    clustere_dataframe = pd.DataFrame(cluster_data[cluster], columns=['column1'])
    word2vec_news_clustered = clustere_dataframe['column1'].apply(lambda sentence: sentence_vector(sentence, model_W2V))
    similarities = cosine_similarity([new_text_transformed], word2vec_news_clustered.tolist()).ravel()
    most_similar_indices = similarities.argsort()[::-1][:1]
    return cluster_data[cluster][most_similar_indices[0]]


# ________________________________________ PRE-TRAINED GLOVE VECTORIZER (GLOVE BEST) ________________________________________ #

with open('modelstest/glove_best/pretrain_glove_assign.pkl', 'rb') as f:
    cluster_data_glove = pickle.load(f)
with open('modelstest/glove_best/kmeans_glove.pkl', 'rb') as f:
    kmeans_glove_pre = pickle.load(f)

def find_similar_documents_glove(new_text, kmeans, cluster_data):

    word2vec_output_file = 'glove.6B.50d.txt.word2vec'
    modelg = KeyedVectors.load_word2vec_format(word2vec_output_file, binary=False)

    def sentence_vector(sentence, word_vectors):
        words = sentence.lower().split()
        vectors = []
        for word in words:
            if word in word_vectors:
                vectors.append(word_vectors[word])
            if not vectors:
                vectors.append(np.zeros(50))
        return np.mean(vectors, axis=0)

    new_text_transformed = sentence_vector(new_text, modelg)
    cluster = kmeans.predict([new_text_transformed])[0]
    print(cluster)
    clustere_dataframe = pd.DataFrame(cluster_data[cluster], columns=['column1'])
    word2vec_news_clustered = clustere_dataframe['column1'].apply(lambda sentence: sentence_vector(sentence, modelg))
    similarities = cosine_similarity([new_text_transformed], word2vec_news_clustered.tolist()).ravel()
    most_similar_indices = similarities.argsort()[::-1][:1]
    return cluster_data[cluster][most_similar_indices[0]]


# ________________________________________ CUSTOMIZED SKIPGRAM VECTORIZER (CUSTOMIZED BEST) ________________________________________ #

with open('modelstest/customized_word2vec_best/customized_skipgram_assign.pkl', 'rb') as f:
    cluster_data_custom = pickle.load(f)
with open('modelstest/customized_word2vec_best/kmeans_customized_skipgram.pkl', 'rb') as f:
    kmeans_custom_word2vec = pickle.load(f)

def find_similar_documents_customword2vec(new_text, kmeans, cluster_data):

    word2vec_output_file = 'customized_word2vec.txt'
    modelg = KeyedVectors.load_word2vec_format(word2vec_output_file, binary=False)

    def sentence_vector(sentence, word_vectors):
        words = sentence.lower().split()
        vectors = []
        for word in words:
            if word in word_vectors:
                vectors.append(word_vectors[word])
            if not vectors:
                vectors.append(np.zeros(300))
        return np.mean(vectors, axis=0)

    new_text_transformed = sentence_vector(new_text, modelg)
    cluster = kmeans.predict([new_text_transformed])[0]
    print(cluster)
    clustere_dataframe = pd.DataFrame(cluster_data[cluster], columns=['column1'])
    word2vec_news_clustered = clustere_dataframe['column1'].apply(lambda sentence: sentence_vector(sentence, modelg))
    similarities = cosine_similarity([new_text_transformed], word2vec_news_clustered.tolist()).ravel()
    most_similar_indices = similarities.argsort()[::-1][:1]
    return cluster_data[cluster][most_similar_indices[0]]


# ________________________________________ LSA/SVD REDUCED DATA ________________________________________ #

with open('modelstest/svd_best/svd_lsa_d.pkl', 'rb') as f:
    svd_d = pickle.load(f)

def find_similar_documents_word2vec(new_text, kmeans, cluster_data):

    model_W2V = KeyedVectors.load_word2vec_format('GoogleNews-vectors-negative300.bin',binary=True, limit=50000,)

    def sentence_vector(sentence, word_vectors):
        words = sentence.lower().split()
        vectors = []
        for word in words:
            if word in word_vectors:
                vectors.append(word_vectors[word])
            if not vectors:
                vectors.append(np.zeros(300))
        return np.mean(vectors, axis=0)

    new_text_transformed = sentence_vector(new_text, model_W2V)
    cluster = kmeans.predict([new_text_transformed])[0]
    print(cluster)

    clustere_dataframe = pd.DataFrame(cluster_data[cluster], columns=['column1'])
    word2vec_news_clustered = clustere_dataframe['column1'].apply(lambda sentence: sentence_vector(sentence, model_W2V))

    svd_model = TruncatedSVD(n_components=50, algorithm='randomized', n_iter=100, random_state=122)
    lsa = svd_model.fit_transform(list(word2vec_news_clustered))

    new_doc_veec = svd_model.transform([new_text_transformed])


    similarities = cosine_similarity(new_doc_veec, lsa.tolist()).ravel()
    most_similar_indices = similarities.argsort()[::-1][:1]
    
    return cluster_data[cluster][most_similar_indices[0]]


# Define my Streamlit app
def app():
    st.image('https://seeklogo.com/images/B/bbc-news-logo-CFBBD6FF4D-seeklogo.com.png', width=100)
    st.title('BBC News Text Analytics Assignment 1')
    st.write('**Created by -** Fareed Khan 25367')
    st.write('**GoogleNews-vectors-negative300.bin** and **glove.6B.50d.txt.word2vec** must be present in your current directory')
    st.markdown("""---""")
    st.write('Enter some text to find similar news in the corpus:')
    
    # Add a text input widget
    new_text = st.text_input('Type your text here')
    
    # Add a button widget
    if st.button('Find similar documents'):
        similar_documents_tfidf = find_similar_documents_tfidf(new_text, vectorizer_tfidf, kmeans_tfidf, data_tfidf, data)
        similar_documents_word2vec = find_similar_documents_word2vec(new_text, kmeans_word2vec_pre, cluster_data)
        similar_documents_glove = find_similar_documents_glove(new_text, kmeans_glove_pre, cluster_data_glove)
        similar_documents_custom = find_similar_documents_customword2vec(new_text, kmeans_custom_word2vec, cluster_data_custom)
        similar_documents_lsa = find_similar_documents_word2vec(new_text, kmeans_word2vec_pre, cluster_data)

        df = pd.DataFrame({'Similar News':[similar_documents_tfidf, similar_documents_word2vec, similar_documents_glove, similar_documents_custom,
        similar_documents_lsa]})
        df.index = ['TF-ID', 'Word2Vec', 'GloVe', 'Customized Word2Vec', 'SVD Word2Vec']
        
        st.dataframe(df)

    
# Running my Streamlit app
if __name__ == '__main__':
    app()
