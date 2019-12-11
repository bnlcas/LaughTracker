import numpy as np
import voice_activity_detector as vad
import sounddevice as sd
import soundfile as sf

#laugh_sound, sample_rate = sf.read('laugh.wav', dtype='float32')
sample_rate = 44100
sample_duration = 0.25  # seconds
voice_detector = vad.VoiceActivityDetector(sample_rate)

def PlayLaughter():
    #sd.play(laugh_sound, sample_rate)
    #status = sd.wait()
    print 'ha\n'

while(True):
    sound_recording = sd.rec(int(sample_duration * sample_rate), samplerate=sample_rate, channels=1)
    sound_recording = np.transpose(sound_recording)[0]
    sd.wait()
    was_voice = voice_detector.speech_on
    voice_detector.CheckActivation(sound_recording)
    if(was_voice and not voice_detector.speech_on):
        PlayLaughter()
