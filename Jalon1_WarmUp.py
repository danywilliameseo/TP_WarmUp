#!/usr/bin/env python
# coding: utf-8

# In[5]:


get_ipython().system('pip install wordcloud')
get_ipython().system('pip install contractions')


# In[3]:


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


# # 1. Exploration du jeu de données

# In[ ]:





# In[ ]:





# In[4]:


DATASET_FILE = "./dataset.csv"


# In[5]:


dataset_df = pd.read_csv(DATASET_FILE)
dataset_df["stars"]





        
    


# ## 1.1 Répartition des avis clients en fonction du nombre d'étoiles

# In[6]:


stars_columns = dataset_df.stars
stars_columns


# In[7]:


stars_columns.value_counts()


# In[8]:


stars_columns.value_counts().plot(kind="pie", figsize=(10, 8), autopct="%1.1f%%", shadow=True)


# ## 1.2 Distribution de la longueur (nombre de mots) des avis clients

# In[9]:


dataset_df["length"] = dataset_df["text"].apply(lambda x: len(x.split()))
dataset_df


# In[10]:


dataset_df.length.plot(kind="hist", bins=100, figsize=(10, 8))


# ## 1.3 Distribution de la longueur des avis clients en fonction du nombre d'étoiles des avis

# In[11]:


plt.figure(figsize=(10, 8))

ax = sns.boxplot(x=dataset_df.stars ,
            y=dataset_df.length,
            showmeans=True,
            )
ax.set_ylim(0, 400)

ax.set_title("Répartition des longueurs des avis en fonction du nombre d'étoiles")


# # 2. Pré-traitement du jeu de données

# In[12]:


df = pd.DataFrame()
df_1_2_stars = dataset_df.loc[dataset_df['stars'] < 3]   #Eliminations des traces posititves dans les reviews à 1 & 2 stars

print(df_1_2_stars)
#il faut tokeniser ses phrases 
#stop word
#raciner
#lemma 
# apres tout ca appliquer, la classification de sentiments



# In[ ]:





# In[ ]:





# In[13]:




def expand_contractions(contraction):
	# take matching contraction in the text
	match = contraction.group(0)
	# first char from matching contraction (D for Doesn't)
	first_char = match[0]
	if contraction_mapping.get(match):
		expanded_contraction = contraction_mapping.get(match)
	else:
		expanded_contraction = contraction_mapping.get(match.lower())
	expanded_contraction = first_char+expanded_contraction[1:]

	return expanded_contraction


# In[14]:



ex_contractions = """
Sometimes our mind doesn't work properly.
"""
contractions.fix(ex_contractions)


# In[15]:


nltk.download('omw-1.4')


# In[18]:


import sys
get_ipython().system('{sys.executable} -m pip install vaderSentiment')

from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
sa=SentimentIntensityAnalyzer()
sa.lexicon


# In[85]:


import nltk
import string
from nltk.tokenize import casual_tokenize
from nltk.tokenize import TreebankWordTokenizer
from nltk.stem.porter import PorterStemmer
from nltk.stem import SnowballStemmer

ps=PorterStemmer()
snow=SnowballStemmer(language='english')

stop_words=nltk.corpus.stopwords.words('english')

tokenizer=TreebankWordTokenizer()
lemmatizer = WordNetLemmatizer()
ponctuations = [";",",",".","?","!"]

def full_neg_sentence(phrase):                    #1 Efface les traces positives d'une phrase
    phrases = phrase.split(".")
    full_negative_phrases = []
    

        
    for phrase in phrases:
        scores=sa.polarity_scores(phrase)
        if scores['compound'] < 0.1  :
            
            full_negative_phrases.append(phrase)
            
   
    final_negative_phrase = '.'.join(full_negative_phrases) 
    
    print("phrase négative:",final_negative_phrase)
        
    return final_negative_phrase
    
def erase_punctuations(words_list):
    remove = string.punctuation
    remove = remove.replace("_", "")
    words = [''.join(letter for letter in word if letter not in remove) for word in words_list if word]
    for i in words:
        if i =="":
            words.remove(i)
    return words
    

for i in df_1_2_stars["text"]:
    
    phrase = contractions.fix(i)                #2 Transforme les 'nt' en not   
    phrase = full_neg_sentence(phrase)
    
    tokens = tokenizer.tokenize(phrase)         #3 Tokenize
    
    
    
    
    tokens = [x.lower() for x in tokens]
    
    
    
    new_tokens = []
    new_tokens2 = []
    for i in enumerate(tokens):                  #4 Transforme les "not adj" en "not_adj"
        if i[1] == "not":        
            id_token = i[0]
            new_token = "not" + "_" + tokens[id_token+1]
            tokens.remove(tokens[id_token+1])
            new_tokens.append(new_token)
            
            
        if i[1] == "never":
            id_token = i[0]
            new_token = "not" + "_" + tokens[id_token+1]
            tokens.remove(tokens[id_token+1])
            new_tokens.append(new_token)
                
            
        else: 
            
            new_tokens.append(tokens[i[0]])
            
    new_tokens = [snow.stem(x) for x in new_tokens if x not in stop_words]  #5 stem
    new_tokens = [lemmatizer.lemmatize(x) for x in new_tokens ] #5 lemmatization
    
    
    new_tokens = erase_punctuations(new_tokens)
    print(new_tokens)       
   
    
    break
   
        
   
   
    
    
            
            
            
            
            
        
            
            
    
    
    
   
        
            
        
            
            
            
            
            
            
        
        
    
    
    
    
        
        
    
    
    
    
    
    
    
    
    


# In[79]:


i = (1,"2")

print(i[1])


# In[ ]:




