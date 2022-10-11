#!/usr/bin/env python3

# NOTE: this example requires PyAudio because it uses the Microphone class

import time
import os
import speech_recognition as sr
from kafka.Queue import QueueProducer
from dotenv import load_dotenv
import json
from datetime import datetime
from svc.NLUmaincmd import NLUMainCmd

load_dotenv()
queueProducer = QueueProducer(os.environ['CLOUDKARAFKA_TOPIC'])
nluCmd = NLUMainCmd()
mainCmd = 'Hey Amanda'
# this is called from the background thread
def callback(recognizer, audio):
    # received audio data, now we'll recognize it using Google Speech Recognition
    try:
        print("Llegó un mensaje!")
        # for testing purposes, we're just using the default API key
        # to use another API key, use `r.recognize_google(audio, key="GOOGLE_SPEECH_RECOGNITION_API_KEY")`
        # instead of `r.recognize_google(audio)`
        txt = recognizer.recognize_google(audio, language="es-ES")
        print("aia $ " + txt)
        bodyObj = nluCmd.matrixMatcher(mainCmd, txt)
        queueProducer.sendMsg(bodyObj)
        
    except sr.UnknownValueError:
        print("Google Speech Recognition could not understand audio")
    except sr.RequestError as e:
        print("Could not request results from Google Speech Recognition service; {0}".format(e))


r = sr.Recognizer()
m = sr.Microphone()
with m as source:
    r.adjust_for_ambient_noise(source)  # we only need to calibrate once, before we start listening

# start listening in the background (note that we don't have to do this inside a `with` statement)
stop_listening = r.listen_in_background(m, callback)
# `stop_listening` is now a function that, when called, stops background listening

# do some unrelated computations for 5 seconds
#for _ in range(50): time.sleep(0.1)  # we're still listening even though the main thread is doing other things
while True: time.sleep(0.1) 
#print("luego se ejecuta stop")
# calling this function requests that the background listener stop listening
#stop_listening(wait_for_stop=True)
#print("se paro me parece")
# do some more unrelated things
#while True: time.sleep(0.1)  # we're not listening anymore, even though the background thread might still be running for a second or two while cleaning up and stopping