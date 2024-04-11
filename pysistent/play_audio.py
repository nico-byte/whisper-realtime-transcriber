import sounddevice as sd
import soundfile as sf
import numpy as np

def play_audio(path_to_file: str):
    fs1, x = sf.read(path_to_file, dtype='float32')
    sd.play(fs1, x)
    sd.wait()
    sd.stop()
