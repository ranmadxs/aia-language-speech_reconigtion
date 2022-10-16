import speech_recognition as sr
from dotenv import load_dotenv
import os
from kafka.Queue import QueueProducer
from repositories.aiaRepo import AIAMessageRepository
from svc.NLUmaincmd import NLUMainCmd
import time
from . import __version__

load_dotenv()
queueProducer = QueueProducer(os.environ['CLOUDKARAFKA_TOPIC'], __version__)
aiaMsgRepo = AIAMessageRepository(os.environ['MONGODB_URI'])
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
            sendObject = queueProducer.msgBuilder(bodyObj)
            id = aiaMsgRepo.insertAIAMessage(sendObject)
            print(id) 
            sendObject['id'] = str(id)
            queueProducer.send(sendObject)
            queueProducer.flush()
            time.sleep(2)
        except:
            print('Sorry could not hear')
