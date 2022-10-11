import speech_recognition as sr
from dotenv import load_dotenv
import os
from kafka.Queue import QueueProducer
from svc.NLUmaincmd import NLUMainCmd
import time

load_dotenv()
queueProducer = QueueProducer(os.environ['CLOUDKARAFKA_TOPIC'])
nluCmd = NLUMainCmd()
r = sr.Recognizer() 
mainCmd = 'Hey Amanda'

def startSpeechReconigtion():
    print(mainCmd)
    print("CLOUDKARAFKA_TOPIC=" + os.environ['CLOUDKARAFKA_TOPIC'])
    with sr.Microphone() as source:
        print('Speak Anything : ')
        audio = r.listen(source)
        try:
            text = r.recognize_google(audio, language="es-ES")
            bodyObj = nluCmd.matrixMatcher(mainCmd, text)
            print('You said: {}'.format(text))
            print(bodyObj)
            queueProducer.sendMsg(bodyObj)        
            queueProducer.flush()
            time.sleep(5)
        except:
            print('Sorry could not hear')
