from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction import text
from sklearn.feature_extraction.text import CountVectorizer
from difflib import SequenceMatcher
from svc.NLUmaincmd import NLUMainCmd
from svc.NLClassifier import NLClassifier

nluCmd = NLUMainCmd()
my_stop_words = text.ENGLISH_STOP_WORDS.union(["spanish"])
cv = CountVectorizer(stop_words=my_stop_words)

text = "lolazo Ella manda encender bomba y luego apagarse"
#text = "Ella manda qué hora es"
#text = "lolazo la pezcada que mas da"

phrase_one = 'Hey Amanda'
phrase_two = text
print("######################################################")
#for token in cv.get_feature_names():
#    print(SequenceMatcher(a="Amanda".upper(), b=token.upper()).ratio())
obj = nluCmd.matrixMatcher(phrase_one, phrase_two)
print (obj)
print("######################################################")
#for token in cv.get_feature_names():
#    print(SequenceMatcher(a="hey".upper(), b=token.upper()).ratio())
