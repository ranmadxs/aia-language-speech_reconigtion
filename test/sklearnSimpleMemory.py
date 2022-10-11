import csv
import pandas as pd
import numpy as np
import string
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score
import sys
 
# setting path
sys.path.append('../')
from svc.NLClassifier import NLClassifier

frases = [
    'a qué hora es',
    'Qué hora es',
    'Vamos a una fiesta',
    'modo análisis',
    'estoy contento',
    'a qué hora nos vamos a ir',
    'detener',
    'ni modo, nada que hacer',
    'Hey Amanda modo Análisis',
    'que tal cómo andamos'
]

nlClassifier = NLClassifier('../resources/memory.csv')
nlClassifier.train()
result = nlClassifier.process(frases)
for expr in result:
    print(expr)
'''
spam_or_ham = pd.read_csv("memory.csv", encoding='utf-8')[["v1", "v2"]]
spam_or_ham.columns = ["label", "text"]
spam_or_ham.head()

print(spam_or_ham["label"].value_counts())
print(spam_or_ham["label"])

punctuation = set(string.punctuation)
def tokenize(sentence):
    tokens = []
    for token in sentence.split():
        new_token = []
        for character in token:
            if character not in punctuation:
                new_token.append(character.lower())
        if new_token:
            tokens.append("".join(new_token))
    return tokens

spam_or_ham.head()["text"].apply(tokenize)


demo_vectorizer = CountVectorizer(
    tokenizer = tokenize,
    binary = True
)
print(spam_or_ham["text"])
from sklearn.model_selection import train_test_split
train_text, test_text, train_labels, test_labels = train_test_split(spam_or_ham["text"], spam_or_ham["label"], stratify=spam_or_ham["label"])
print(f"Training examples: {len(train_text)}, testing examples {len(test_text)}")

real_vectorizer = CountVectorizer(tokenizer = tokenize, binary=True)
train_X = real_vectorizer.fit_transform(train_text)
test_X = real_vectorizer.transform(test_text)
classifier = LinearSVC()
classifier.fit(train_X, train_labels)
LinearSVC(C=1.0, class_weight=None, dual=True, fit_intercept=True,
          intercept_scaling=1, loss='squared_hinge', max_iter=1000,
          multi_class='ovr', penalty='l2', random_state=None, tol=0.0001,
          verbose=0)
predicciones = classifier.predict(test_X)
accuracy = accuracy_score(test_labels, predicciones)
print(f"Accuracy: {accuracy:.4%}")

frases = [
  'a qué hora es',
  'Qué hora es',
  'modo análisis',
  'ni modo, nada que hacer',
  'Hey Amanda modo Análisis',
  'que tal cómo andamos'
]

frases_X = real_vectorizer.transform(frases)
predicciones = classifier.predict(frases_X)

for text, label in zip(frases, predicciones):
    print(f"{label:5} - {text}")

'''