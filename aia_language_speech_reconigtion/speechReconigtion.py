import speech_recognition as sr
from dotenv import load_dotenv
import os
from aia_utils.Queue import QueueProducer
from aia_utils.mqtt import MqttConsumer
from repositories.aiaRepo import AIAMessageRepository
from svc.NLUmaincmd import NLUMainCmd
import time
from . import __version__
from aia_utils.logs_cfg import config_logger
import logging
config_logger()

# Configura el nivel de logging para pymongo
logging.getLogger("pymongo").setLevel(logging.ERROR)
logger = logging.getLogger(__name__)
load_dotenv()
queueProducer = QueueProducer(os.environ['CLOUDKARAFKA_TOPIC'], __version__, "aia_language_speech_reconigtion")
client_id = "aia_language_speech_reconigtion"
aiaMsgRepo = AIAMessageRepository(os.environ['MONGODB_URI'])
nluCmd = NLUMainCmd()
r = sr.Recognizer() 
mainCmd = 'Hey Amanda'

def sendMsg(text: str ):
    bodyObj = nluCmd.matrixMatcher(mainCmd, text)
    logger.debug('You said: {}'.format(text))
    logger.info(bodyObj)
    sendObject = queueProducer.msgBuilder(bodyObj)
    aiaMsg = queueProducer.msgBuilder(bodyObj)
    id = aiaMsgRepo.insertAIAMessage(aiaMsg)
    logger.info(f"ObjId == {id}") 
    sendObject['_id'] = str(id)
    logger.info(sendObject)
    queueProducer.send(sendObject)
    queueProducer.flush()

MqttConsumer(os.environ['MQTT_TOPIC'], client_id, [sendMsg])



def startSpeechReconigtion():
    logger.info("startSpeechReconigtion")
    

def startSpeechReconigtionSvcMic():
    print(mainCmd)
    print("CLOUDKARAFKA_TOPIC=" + os.environ['CLOUDKARAFKA_TOPIC'])
    with sr.Microphone() as source:
        print('Speak Anything : ')
        audio = r.listen(source)
        try:
            text = r.recognize_google(audio, language="es-ES")
            print('You said: {}'.format(text))
            sendMsg(text)
            time.sleep(1)
        except:
            print('Sorry could not hear')
