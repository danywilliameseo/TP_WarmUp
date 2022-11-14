#!/usr/bin/env python
# coding: utf-8

# # JALON 3

# In[4]:


import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import NMF

from wordcloud import WordCloud

import nltk
from nltk.tokenize import RegexpTokenizer
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
import contractions


nltk.download('punkt')


# In[7]:


# PREPROCESS
topics = {0:'service',1:'price',2:'pizza',3:'order',4:'food',5:'wait',6:'place',7:'burger',8:'taste',9:'chicken',10:'bar',11:'time',12:'go',13:'lunch',14:'restaurant'}
negative_words = ['not', 'no', 'never', 'nor', 'hardly', 'barely']
negative_prefix = "NOT_"
tokenizer = RegexpTokenizer(r'\w+')

def tokenize_text(text):
    text_processed = " ".join(tokenizer.tokenize(text))
    return text_processed


import en_core_web_sm
nlp = en_core_web_sm.load(disable=['parser', 'tagger', 'ner'])

lemmatizer = WordNetLemmatizer()

def lemmatize_text(text):
    
    tokens_tagged = nltk.pos_tag(nltk.word_tokenize(text))
    lemmatized_text_list = list()
    
    for word, tag in tokens_tagged:
        if tag.startswith('J'):
            lemmatized_text_list.append(lemmatizer.lemmatize(word,'a')) # Lemmatise adjectives. Not doing anything since we remove all adjective
        elif tag.startswith('V'):
            lemmatized_text_list.append(lemmatizer.lemmatize(word,'v')) # Lemmatise verbs
        elif tag.startswith('N'):
            lemmatized_text_list.append(lemmatizer.lemmatize(word,'n')) # Lemmatise nouns
        elif tag.startswith('R'):
            lemmatized_text_list.append(lemmatizer.lemmatize(word,'r')) # Lemmatise adverbs
        else:
            lemmatized_text_list.append(lemmatizer.lemmatize(word)) # If no tags has been found, perform a non specific lemmatisation
    
    return " ".join(lemmatized_text_list)


    
    
    
def normalize_text(text):
    return " ".join([word.lower() for word in text.split()])


def contraction_text(text):
    return contractions.fix(text)


def get_negative_token(text):
    tokens = text.split()
    negative_idx = [i+1 for i in range(len(tokens)-1) if tokens[i] in negative_words]
    for idx in negative_idx:
        if idx < len(tokens):
            tokens[idx]= negative_prefix + tokens[idx]
    
    tokens = [token for i,token in enumerate(tokens) if i+1 not in negative_idx]
    
    return " ".join(tokens)


from spacy.lang.en.stop_words import STOP_WORDS

def remove_stopwords(text):
    english_stopwords = stopwords.words("english") + list(STOP_WORDS) + ["tell", "restaurant"]
    
    return " ".join([word for word in text.split() if word not in english_stopwords])


def preprocess_text(text):
    
    # Tokenize review
    text = tokenize_text(text)
    
    # Lemmatize review
    text = lemmatize_text(text)
    
    # Normalize review
    text = normalize_text(text)
    
    # Remove contractions
    text = contraction_text(text)

    # Get negative tokens
    text = get_negative_token(text)
    
    # Remove stopwords
    text = remove_stopwords(text)
    
    return text
    


# In[106]:





# In[16]:


#PREDICT
import pickle 
from textblob import TextBlob


with open(r"pickle_files_nmfv1","rb") as input_file:
    model = pickle.load(input_file)
    
with open(r"pickle_files_vectoriseurv1","rb") as input_file:
    vectoriseur = pickle.load(input_file)
list_topics_final = []
def predict(text,number_of_topics):
    l = []
   
    blob = TextBlob(text) 
    if blob.sentiment.polarity < 0 :
        l.append(text)
        
        preprocess_text(text)
        a = vectoriseur.transform(l)
        result = model.transform(a) #List 15 Valeur 
        liste_coeff = []
        for i in range(14,0,-1):

            print(result[0][i])
            if result[0][i] != 0:
                liste_coeff.append(result[0][i])




                

        liste_coeff.sort(reverse=True)
        

        list_index = []

        for i in range(0,len(liste_coeff)):
            list_index.append((get_index(liste_coeff[i],result[0])))



        
   
        for i in list_index:
             
            list_topics_final.append(topics[i])
            
            if len(list_topics_final) == number_of_topics:
                print("nombre nécessaire obtenu")
                break
        
    
    return blob.sentiment.polarity
    

        
        
        
    
        
def get_index(value,liste):
    for i in enumerate(liste):
        if i[1] == value:
            index = i[0]
            
            return index 
        
            

    
#APP STREAMLIT
 





# In[ ]:


import streamlit as st

st.title('Topic Provider')
st.write("This app will provide you the topic of your sentence")

st.header('Topic(s)/Polarity')


st.sidebar.header("Enter your sentence here")






       
text =  st.sidebar.text_input("Enter your sentence",
    
        key="placeholder")

number = st.sidebar.number_input('How many keywords you want to display ? ',0,5)

print(text)



if text !="" and number != 0:
    a = predict(text,number)
    
    if len(list_topics_final) != 0:
        
        for i in list_topics_final:
            st.info(i)
    st.info(a)

  
    








        
        
        
        
        
 
