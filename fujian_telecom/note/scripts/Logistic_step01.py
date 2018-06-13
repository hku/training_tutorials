#encoding=utf-8

import jieba
import jieba.posseg as pseg
import numpy as np
import pandas as pd

df = pd.read_csv('spams.csv', delimiter='\t+',header=None)

corpus_raw = df[1]
y = df[0]


def preprocess_doc(doc):
	words=list(pseg.cut(doc))	
	words_filtered = filter(lambda w: w.flag!= u"x" and w.flag!=u"m" and (w.word not in [u"请", u"的", u"了", u"你", u"我"]), words)
	return " ".join(map(lambda w: w.word, words_filtered))

corpus = map(preprocess_doc, corpus_raw)

for d in corpus:
	print(d)

from sklearn.feature_extraction.text import TfidfVectorizer

vectorizer = TfidfVectorizer()
corpus_sparse_matrix = vectorizer.fit_transform(corpus)
X = corpus_sparse_matrix.todense()

X_train = X[:6]
y_train = y[:6]

X_test = X[6:]
y_test = y[6:]

from sklearn.linear_model.logistic import LogisticRegression



model = LogisticRegression()
model.fit(X_train, y_train)

yp = model.predict(X_test)
