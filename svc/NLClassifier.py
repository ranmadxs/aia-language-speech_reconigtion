import csv
import pandas as pd
import numpy as np
import string
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

class NLClassifier:

    def __init__(self, csvMemFile = 'memory.csv'):
        self.memory = pd.read_csv(csvMemFile, encoding='utf-8')[["v1", "v2"]]
        self.memory.columns = ["label", "text"]
        self.memory.head()
        self.punctuation = set(string.punctuation)
    
    def train(self):
        self.memory.head()["text"].apply(self.tokenize)
        train_text, test_text, train_labels, test_labels = train_test_split(self.memory["text"], self.memory["label"], stratify=self.memory["label"])
        print(f"Training examples: {len(train_text)}, testing examples {len(test_text)}")
        self.real_vectorizer = CountVectorizer(tokenizer = self.tokenize, binary=True)
        train_X = self.real_vectorizer.fit_transform(train_text)
        test_X = self.real_vectorizer.transform(test_text)
        self.classifier = LinearSVC()
        self.classifier.fit(train_X, train_labels)
        LinearSVC(C=1.0, class_weight=None, dual=True, fit_intercept=True,
                intercept_scaling=1, loss='squared_hinge', max_iter=1000,
                multi_class='ovr', penalty='l2', random_state=None, tol=0.0001,
                verbose=0)
        predicciones = self.classifier.predict(test_X)
        accuracy = accuracy_score(test_labels, predicciones)
        print(f"Accuracy: {accuracy:.4%}")

    def tokenize(self, sentence):
        tokens = []
        for token in sentence.split():
            new_token = []
            for character in token:
                if character not in self.punctuation:
                    new_token.append(character.lower())
            if new_token:
                tokens.append("".join(new_token))
        return tokens
    
    def process(self, frases):

        frases_X = self.real_vectorizer.transform(frases)
        print(frases_X)
        predicciones = self.classifier.predict(frases_X)
        print (predicciones)
        #for text, label in zip(frases, predicciones):
        #    print(f"{label:5} - {text}")
        return [list(a) for a in zip(frases, predicciones)]