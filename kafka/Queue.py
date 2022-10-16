from dotenv import load_dotenv
import sys
import os
from confluent_kafka import Producer
import json
import time
from datetime import datetime

load_dotenv()

class QueueProducer:
    def __init__(self, topic, version = None):
        self.version = version
        self.topic = topic
        self.conf = {
            'bootstrap.servers': os.environ['CLOUDKARAFKA_BROKERS'],
            'session.timeout.ms': 6000,
            'default.topic.config': {'auto.offset.reset': 'smallest'},
            'security.protocol': 'SASL_SSL',
            'sasl.mechanisms': 'SCRAM-SHA-256',
            'sasl.username': os.environ['CLOUDKARAFKA_USERNAME'],
            'sasl.password': os.environ['CLOUDKARAFKA_PASSWORD']            
        }
        self.producer = Producer(**self.conf)

    def msgBuilder(self, bodyObject):
        now = datetime.now()
        objMessage = {
            "head": {
                "producer": "aia-language-speech_recognition",
                "creationDate": now.strftime("%Y-%m-%d %H:%M:%S"),
                "version": self.version
            }, 
            "body": bodyObject,
            "breadcrumb": [{
                "creationDate": now.strftime("%Y-%m-%d %H:%M:%S"),
                "name": "aia-language-speech_reconigtion"
            }],
            "status": {
                "creationDate": now.strftime("%Y-%m-%d %H:%M:%S"),
                "code": "SEND",
                "description": "AIA new message arrived"
            }
        }
        return objMessage

    def sendMsg(self, bodyObject, callback_queue = None):
        objStr = self.msgBuilder(bodyObject)
        self.send(objStr, callback_queue)

    def send(self, msg, callback_queue = None):
        print(msg)
        msg = str(msg)
        self.producer.produce(self.topic, msg.rstrip(), callback=callback_queue)
        self.producer.poll(0)
    
    def flush(self):
        self.producer.flush()

