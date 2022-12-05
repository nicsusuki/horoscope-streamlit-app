# -*- coding: utf-8 -*-
"""
Created on Mon Dec  5 10:26:55 2022

@author: susuk002
"""

import numpy as np
import pandas as pd
import string
import time
#import re
#import torch
#import tensorflow as tf
#import matplotlib.pyplot as plt
import pickle as pkl
import streamlit as st
from wordcloud import WordCloud, STOPWORDS
from deepmultilingualpunctuation import PunctuationModel
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import load_model
from tensorflow.python.keras.saving.hdf5_format import save_attributes_to_hdf5_group

images = ["https://i.pinimg.com/736x/8b/60/fa/8b60fa4ce729a0cde32198343360e9a5.jpg"]

st.sidebar.image(images)
st.sidebar.text('MABA 6490 Final Project')
st.sidebar.text('Nicole Susuki')
@st.cache(allow_output_mutation=True)
def load_horoscope_model():
    model=load_model('horoscopeModel.h5')
    return model

model = load_horoscope_model()

@st.cache(allow_output_mutation = True)
def load_punc_model():
    punctuation_model = PunctuationModel()
    return punctuation_model

punctuation_model = load_punc_model()


@st.cache
def load_get_word():
    # open the get_word file
    fileo = open('get_word.pkl' , "rb")
    # loading data
    get_word = pkl.load(fileo)
    return get_word

get_word = load_get_word()

@st.cache
def load_tokenizer():
    # open the horoscope_tokenizer file
    fileo = open('horoscope_tokenizer.pkl' , "rb")
    # loading data
    tokenizer = pkl.load(fileo)
    return tokenizer

tokenizer = load_tokenizer()


# with st.spinner("Loading the cosmos..."):
#     # #load models
#     # punctuation_model = PunctuationModel()

#     # # open the model file
#     # model=load_model('horoscopeModel.h5')
#     # # model.summary()
    
#     # open the get_word file
#     fileo = open('get_word.pkl' , "rb")
#     # loading data
#     get_word = pkl.load(fileo)

#     # open the horoscope_tokenizer file
#     fileo = open('horoscope_tokenizer.pkl' , "rb")
#     # loading data
#     tokenizer = pkl.load(fileo)


# #load data
# url = 'https://raw.githubusercontent.com/nicsusuki/horoscope-streamlit-app/main/horoscopes.csv'
# data = pd.read_csv(url,
#                   error_bad_lines=False, 
#                   sep = "|", header = None, 
#                   names = ["text", "date", "sign"], index_col = 0)










st.title("Horoscope Generator")
st.markdown('This uses NLP on 3 years worth of NYT horoscopes to generate your own horoscope based on user inputted seed text. The algorithm employs an element of randomness so that no two horoscopes are the same.')

st.markdown('The data came from:')
st.markdown('https://github.com/dsnam/markovscope/blob/master/data/horoscopes.csv')
st.markdown('The punctuation model came from Huggingface:')
st.markdown('https://huggingface.co/oliverguhr/fullstop-punctuation-multilang-large')
query = st.selectbox(
    'What is your sign?',
    ('Aries','Taurus','Gemini','Cancer','Leo','Virgo','Libra', 'Scorpio',
      'Sagitarius', 'Capricorn','Aquarius', 'Pisces', 'Generate my own'))

if query == 'Generate my own':
    query = st.text_input("Type horoscope seed text here")
    
search_button = st.button('Search the cosmos!')

# words = ""
# stopwords = set(STOPWORDS)
# for review in data.text.values:
#     text = str(review)
#     text = text.split()
#     words += " ".join([(i.lower() + " ") for i in text])
    
# #cleaning function - lowercase, remove punc
# def clean_text(text):
#     words = str(text).split()
#     words = [i.lower() + " " for i in words]
#     words = " ".join(words)
#     words = words.translate(words.maketrans('', '', string.punctuation))
#     return words

# data['text'] = data['text'].apply(clean_text)

# #tokenize the data
vocab_size = 15000
# max_length = 50
# oov_tok = "<OOV>"

# tokenizer = Tokenizer(num_words = vocab_size, oov_token = oov_tok)
# tokenizer.fit_on_texts(data.text.values)
# word_index = tokenizer.word_index

# get_word = {v: k for k, v in word_index.items()}

# #create n-grams
# sequences = tokenizer.texts_to_sequences(data.text.values[::100])

# n_gram_sequences = []
# for sequence in sequences:
#     for i,j in enumerate(sequence):
#         if i < (len(sequence) - 10):
#             s = sequence[i:i + 10]
#             for k, l in enumerate(s):
#                 n_gram_sequences.append(s[:k + 1])
        
# np.array(n_gram_sequences).shape

# n_gram_sequences = np.array(n_gram_sequences)
max_len = 10 #max([len(i) for i in n_gram_sequences]) ##max len = 10

#predict horoscopes
avg_length = 44 #int(len(words.split())/len(data))  ## average length of horoscope 44

#takes seed text and generates horoscopes using closest matching words
#uses random choice element to change horoscopes returned
#@st.cache
def write_horoscope(seed_text):
    for _ in range(avg_length):
        token_list = tokenizer.texts_to_sequences([seed_text])[0]
        token_list = pad_sequences([token_list], maxlen=max_len-1, padding='pre')
        pred_probs = model.predict(token_list)
        predicted = np.random.choice(np.linspace(0, vocab_size - 1, vocab_size), p = pred_probs[0])
        if predicted == 1: ## if it's OOV, pick the next most likely one.
            pred_probs[0][1] = 0
            predicted = np.argmax(pred_probs)
        output_word = get_word[predicted]
        seed_text += " " + output_word
    return seed_text


if search_button:
    st.markdown("**Searching the cosmos for your horoscope:** " + query)
    with st.spinner("Consulting the oracle..."):
        time.sleep(2)
        horoscope_text = write_horoscope(query)
        horoscope = punctuation_model.restore_punctuation(horoscope_text)
        st.success(horoscope)

