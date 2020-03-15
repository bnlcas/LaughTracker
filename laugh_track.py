import numpy as np
import voice_activity_detector as vad
import sounddevice as sd
import soundfile as sf

laugh_sound, sample_rate = sf.read('laughter.wav', dtype='float32')
sample_rate = 44100
sample_duration = 0.2  # seconds
voice_detector = vad.VoiceActivityDetector(sample_rate)

def PlayLaughter():
    sd.play(laugh_sound, sample_rate)
    status = sd.wait()

while(True):
    sound_recording = sd.rec(int(sample_duration * sample_rate), samplerate=sample_rate, channels=1)
    sound_recording = np.transpose(sound_recording)[0]
    sd.wait()
    was_voice = voice_detector.speech_on
    voice_detector.CheckActivation(sound_recording)
    if(voice_detector.speech_on):
        print('...')
    if(voice_detector.DetectSpeechEnd()):
        PlayLaughter();
