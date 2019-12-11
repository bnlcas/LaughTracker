import cPickle
import numpy as np
import scipy.signal as signal
from sklearn.svm import SVC

class VoiceActivityDetector():
    def __init__(self, fs):
        self.sample_rate = fs
        #self.sample_rate = 44100
        self.speech_on = False

        #Spectrum Based:
        self.speech_energy_threshold = 0.6 #60% of energy in voice band
        self.speech_start_band = 75
        self.speech_end_band = 280

        # Classifier:
        self.band_bins = [180, 250, 360, 500, 1000, 2000, 5000, 10000, 25000]
        self.prior_bands = [0.0]*len(self.band_bins)
        with open('svm_classifier.pkl', 'rb') as fid:
            self.clf = cPickle.load(fid)

    def CheckActivation(self, data):
        band_energy = self.ExtractPowerBands(data)
        self.speech_on = self.ClassifySpeech(band_energy)

    def ExtractPowerBands(self, x):
        freq, power = signal.periodogram(x, self.sample_rate)
        total_energy = np.sum(power)
        band_energy_fraction = [0.0] * (len(self.band_bins))
        band_ind = 0
        for i in range(len(power)):
            if(freq[i]>self.band_bins[band_ind]):
                band_ind += 1
            band_energy_fraction[band_ind] += power[i]
        normalized_band_energy = [b / total_energy for b in band_energy_fraction]
        return normalized_band_energy

    def ClassifySpeech(self, band_energy):
        band_energy.extend(self.prior_bands)
        prior_bands = band_energy[0:len(self.band_bins)]
        x = np.reshape(band_energy, (1,len(band_energy)))
        return self.clf.predict(x)[0]
