# -*- coding: utf-8 -*-
"""
Created on Mon Aug  9 10:41:32 2021

@author: Aryan Sharma
"""

import numpy as np
import pandas as pd
df = pd.read_csv('balancedReviews.csv')

#handling the missing values
df.isnull().any(axis = 0)
df.dropna(inplace = True)

#make a binary clssification problem
df = df[df['overall'] != 3]

#creating a new column (label) based on the overall column to make a binary classification problem
df['Positivity'] = np.where(df['overall'] > 3, 1, 0)

#Stemming and removing stopwords from NLP data
import re
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

corpus = []
for i in range(0, df.shape[0]):
    review = re.sub('[^a-zA-Z]', ' ', df.iloc[i,1])
    review = review.lower()
    review = review.split()
    
    #remove the stopwords
    review = [word for word in review if not word in stopwords.words('english')]
    
    # stemming
    ps = PorterStemmer()
    review = (ps.stem(word) for word in review)
    
    review = " ".join(review)
    corpus.append(review)
    
# Now features --> corpus
# labels --> positivity --> df.iloc[:,-1]

corpus.toarray()
features = corpus
#Train test split
from sklearn.model_selection import train_test_split
features_train, features_test, labels_train, labels_test = train_test_split(features, df['Positivity'], test_size = 0.2, random_state = 0)


#Managing the NLP data(Vectorization)
from sklearn.feature_extraction.text import TfidfVectorizer
vect = TfidfVectorizer(min_df = 5).fit(features_train)
features_train_vect = vect.transform(features_train)

# training the model
from sklearn.linear_model import LogisticRegression
model = LogisticRegression()
model.fit(features_train_vect, labels_train)

#making prediction 
prediction = model.predict(vect.transform(features_test))

# calculating score/accuracy of the model
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(labels_test, prediction)

#or
from sklearn.metrics import roc_auc_score
score = roc_auc_score(labels_test, prediction)

#both score and cm gives us the score of model

import pickle
file = open('model_pickle.pkl','wb')
pickle.dump(model, file)

# to export vocabulary created while vectorization
pickle.dump(vect.vocabulary_, open('vocab.pkl','wb'))


