# test_capitalize.py

#poetry run pytest tests/test_sendmsg.py::test_send_msg -s
#poetry run pytest -s
from dotenv import load_dotenv
import os
from aia_utils.Queue import QueueProducer
load_dotenv()
queueProducer = QueueProducer(os.environ['CLOUDKARAFKA_TOPIC'], 'testVersion')
from aia_language_speech_reconigtion.speechReconigtion import sendMsg
import speech_recognition as sr

#poetry run pytest tests/test_sendmsg.py::test_send -s
def test_send():
    print ("send msg")
    AUDIO_FILE =  "resources/test/audio_sampling/wh40k_tactical.wav"
    r = sr.Recognizer()
    with sr.AudioFile(AUDIO_FILE) as source:
        # read the entire audio file
        audio = r.record(source)
        # recognize speech using Google Speech Recognition
        text = r.recognize_google(audio, language="es-ES")
        sendMsg(text)

def test_send_msg():
    print("Test send msg to kafka")
    sendObject = {
        "id": "6385019273a9bcf7c1ee3dce",
        "body": {
            "cmd": "leer correo de yahoo",
            "msg": "Ella manda leer correo de Yahoo",
            "isAia": True,
            "classification": "CMD"
        },
        "status": {
            "creationDate": "2022-11-28T18:44:36.832Z",
            "code": "PROCESSED",
            "description": "AIA Semantic Graph Generated"
        },
        "head": {
            "producer": "aia_language_speech_recognition",
            "creationDate": "2022-01-28T18:44:34.000Z",
            "version": "0.1.1"
        },
        "breadcrumb": [
            {
            "creationDate": "2022-01-28T18:44:34.000Z",
            "name": "aia_language_speech_reconigtion"
            },
            {
            "creationDate": "2022-11-28T18:44:36.894Z",
            "name": "amanda-ia.NLProcessorComponent"
            }
        ],
        }
    queueProducer.send(sendObject)
    queueProducer.flush()
    assert True