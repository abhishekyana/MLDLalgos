
# coding: utf-8

# In[1]:

import os
import numpy as np
import re


# In[2]:

# Creates dictionary from all the emails in the directory
def build_dictionary(dir):
    # Read the file names
    emails = os.listdir(dir)
    emails.sort()
    # Array to hold all the words in the emails
    dictionary = []
    # Collecting all words from those emails
    for email in emails:
        m = open(os.path.join(dir, email))
        for i, line in enumerate(m):
            if i == 2: # Body of email is only 3rd line of text file
                words = [w for w in line.lower().split() if w.isalpha() and len(w)>1]
                dictionary += words

    # We now have the array of words which may have duplicate entries
    dictionary = sorted(list(set(dictionary))) # Removes duplicates
    DICTY = {w:i for i,w in enumerate(dictionary)}
    return DICTY

def build_features(dir, dictionary):
    # Read the file names
    emails = os.listdir(dir)
    emails.sort()
    # N-dimensional array to have the features
    features_matrix = np.zeros((len(emails), len(dictionary)))

    # collecting the number of occurances of each of the words in the emails
    for email_index, email in enumerate(emails):
        m = open(os.path.join(dir, email))
        for line_index, line in enumerate(m):
            if line_index == 2:
                words = line.split()
                for word_index, word in enumerate(dictionary):
                    features_matrix[email_index, word_index] = words.count(word)
    return features_matrix

def build_labels(dir):
    # Read the file names
    emails = os.listdir(dir)
    emails.sort()
    # N dimensional array of labels
    labels_matrix = np.zeros(len(emails))
    for index, email in enumerate(emails):
        labels_matrix[index] = 1 if re.search('spms*', email) else 0
    return labels_matrix


# In[3]:

train_dir = './train_data'
print('1. Building the dictionary')
dictionary = build_dictionary(train_dir)


# In[4]:

print('2. Building the training features and labels')
features_train = build_features(train_dir, dictionary)
labels_train = build_labels(train_dir)


# In[143]:

class MultinomialNaiveBayes:
    def __init__(self, alpha=1.0):
        self.alpha = alpha # Used for laplace smoothing the features
        self.class_prior = None # Prior Probabilities
        self.fit_prior = True # Fit Prior
        
    def _update_feature_log_prob(self,alpha):
        smoothed_fc = self.feature_count_ + alpha
        smoothed_cc = smoothed_fc.sum(axis=1).reshape(-1, 1)
        self.feature_log_prob_ = (np.log(smoothed_fc) - np.log(smoothed_cc))
    
    def _update_class_log_prior(self,class_prior=None):
        log_class_count = np.log(self.class_count_)
        self.class_log_prior_ = (log_class_count - np.log(self.class_count_.sum()))
    
    def predict(self, X):
        preds = ((X @ self.feature_log_prob_.T) + self.class_log_prior_)   #Joint log likelihood
        return self.classes_[np.argmax(preds, axis=1)]
    
    def fit(self, X, y):
        _, n_features = X.shape
        Y = np.reshape(y,(-1,1)).astype(np.float32)
        self.classes_ = np.unique(Y)
        Y = np.concatenate((1 - Y, Y), axis=1)
        n_effective_classes = Y.shape[1]
        self.feature_count_ = Y.T@X
        self.class_count_ = Y.sum(axis=0)
        self._update_feature_log_prob(self.alpha)
        self._update_class_log_prior(class_prior=self.class_prior)
        return self


# In[144]:

mnb = MultinomialNaiveBayes()
mnb.fit(features_train,labels_train)


# In[145]:

Preds = mnb.predict(features_train)


# In[146]:

(Preds==labels_train).mean()


# In[147]:

# test_dir = './test_data'
# print('4. Building the test features and labels')
# features_test = build_features(test_dir, dictionary)
# labels_test = build_labels(test_dir)


# In[151]:

print('5. Calculating the accuracy of the trained classifier over the test emails')
accuracy = (mnb.predict(features_test)==labels_test).mean()
print(f'\tAccuracy : {accuracy*100:2.4f}%')

