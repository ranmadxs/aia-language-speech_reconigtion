#!/usr/bin/env python3

# NOTE: this example requires PyAudio because it uses the Microphone class

import speech_recognition as sr

FOLDER_AUDIO = "resources/test/audio_sampling/"
FILE_AUDIO = "test"
#poetry run pytest tests/test_write_audio.py::test_save -s
def test_save():
    # obtain audio from the microphone
    r = sr.Recognizer()
    with sr.Microphone() as source:
        r.adjust_for_ambient_noise(source) 
        print("Say something!")
        audio = r.listen(source)

    # write audio to a RAW file
    #with open(f"{FOLDER_AUDIO}{FILE_AUDIO}.raw", "wb") as f:
    #    f.write(audio.get_raw_data())

    # write audio to a WAV file
    with open(f"{FOLDER_AUDIO}{FILE_AUDIO}.wav", "wb") as f:
        f.write(audio.get_wav_data())

    # write audio to an AIFF file
    #with open(f"{FOLDER_AUDIO}{FILE_AUDIO}.aiff", "wb") as f:
    #    f.write(audio.get_aiff_data())

    # write audio to a FLAC file
    with open(f"{FOLDER_AUDIO}{FILE_AUDIO}.flac", "wb") as f:
        f.write(audio.get_flac_data())