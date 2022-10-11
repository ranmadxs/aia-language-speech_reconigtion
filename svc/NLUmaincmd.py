from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction import text
from sklearn.feature_extraction.text import CountVectorizer
from difflib import SequenceMatcher

import sys
 
# setting path
sys.path.append('svc/')
from NLClassifier import NLClassifier

class NLUMainCmd:

    def __init__(self, sensibilidad = 0.65):
        my_stop_words = text.ENGLISH_STOP_WORDS.union(["spanish"])
        self.cv = CountVectorizer(stop_words=my_stop_words)
        self.sensibilidad = sensibilidad
        self.nlClassifier = NLClassifier('resources/memory.csv')
        self.nlClassifier.train()
    
    def txtMatcher(self, token, txt):
        return SequenceMatcher(a=token, b=txt).ratio()
    
    def matrixMatcher(self, str1, str2):
        phrase_one = str1.lower()
        phrase_two = str2.lower()
        #tokens1 = phrase_one.split()
        tokens1 = [phrase_one]
        #self.cv.fit_transform([phrase_two])
        #tokens2 = self.cv.get_feature_names()    
        tokens2 = phrase_two.split()
        tokens2 = [' '.join(tokens2[i:i+2]) for i in range(0, len(tokens2), 2)]
        #list1 = list(map(txtMatcher, tokens2))
        listM = []
        listIdx = []
        for token in tokens1:
            listAux = [self.txtMatcher(token, i) for i in tokens2]
            listIdx.append(listAux.index(max(listAux)))
            listM.append(listAux)
        #list2 = [txtMatcher('Amanda', i) for i in tokens2]
        print(tokens2)
        print(listM)
        print("-------------------------------------------------------")
        print(listIdx)
        mainCmd = []
        for idx in listIdx:
            mainCmd.append(tokens2[idx])
        print(mainCmd)
        mainCmdTxt = ' '.join(mainCmd)
        print(mainCmdTxt)
        print("-------------------------------------------------------")
        print(phrase_two)
        posInTxt = phrase_two.find(mainCmdTxt)
        cmdAmanda = None
        isAiaMsg = False
        classification = None
        if(posInTxt >= 0):
            print (phrase_two.find(mainCmdTxt))
            calcSimilar = self.txtMatcher(phrase_one, mainCmdTxt)
            print(calcSimilar)
            cmdAmanda = phrase_two[posInTxt + len(mainCmdTxt):].strip()
            print(cmdAmanda)
            if (self.sensibilidad <= calcSimilar):
                isAiaMsg = True
                result = self.nlClassifier.process([cmdAmanda])
                classification = result[0][1]
                print(result)             
        return {
            'cmd': cmdAmanda,
            'msg': str2,
            'classification': classification,
            'isAia': isAiaMsg
        }
