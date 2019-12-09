import numpy as np
import scipy.signal as signal

class VoiceActivityDetector():
    def __init__(self, fs):
        self.sample_rate = fs
        self.sample_rate = 44100
        self.speech_on = False

        #Spectrum Based:
        self.speech_energy_threshold = 0.6 #60% of energy in voice band
        self.speech_start_band = 75
        self.speech_end_band = 280

    def CheckActivation(self, data):
        speech_band_energy_fraction = self.GetSpeechBandEnergyFraction(data)
        self.speech_on = self.ClassifySpeech(speech_band_energy_fraction)

    def FreqInRange(self, f):
        return freq[i] > self.speech_start_band and freq[i] < self.speech_end_band

    def GetSpeechBandEnergyFraction(self,data):
        freq, power = signal.periodogram(data, self.sample_rate)
        total_energy = np.sum(power)
        in_range = lambda (f) : (f > self.speech_start_band) and (f < self.speech_end_band)
        band_energy = sum([power[i] for i in range(len(power)) if in_range(freq[i])])
        return band_energy / total_energy

    def ClassifySpeech(self, speech_band_energy_fraction):
        print(speech_band_energy_fraction )
        return (speech_band_energy_fraction > self.speech_energy_threshold)
