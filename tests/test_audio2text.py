import speech_recognition as sr

# obtain path to "english.wav" in the same folder as this script
from os import path
AUDIO_FILE =  "resources/test/audio_sampling/wh40k_sm.wav"

#poetry run pytest tests/test_audio2text.py::test_audio2text -s
def test_audio2text():
    # use the wav file as the audio source
    r = sr.Recognizer()
    with sr.AudioFile(AUDIO_FILE) as source:
        # read the entire audio file
        audio = r.record(source)
        # recognize speech using Google Speech Recognition
        print("Google Speech Recognition thinks you said " + r.recognize_google(audio, language="es-ES"))
    
    # recognize speech using Sphinx
    #try:
    #    print("Sphinx thinks you said " + r.recognize_sphinx(audio, language="es-ES"))
    #except sr.UnknownValueError:
    #    print("Sphinx could not understand audio")
    #except sr.RequestError as e:
    #    print("Sphinx error; {0}".format(e))