import numpy as np
import librosa
import hparams as hp


# ref : https://github.com/erogol/WaveRNN/blob/master/utils/audio.py 
# ref : https://github.com/fatchord/WaveRNN/blob/master/utils/dsp.py

def linear_to_mel(spectrogram):
    return librosa.feature.melspectrogram(S=spectrogram, 
        sr=hp.sample_rate, n_fft=hp.filter_length, n_mels=hp.num_mels, fmin=hp.fmin)

def stft(y):
    return librosa.stft(y=y,
        n_fft=hp.n_fft, hop_length=hp.hop_length, win_length=hp.win_length)

def normalize(S):
    return np.clip((S - hp.min_level_db) / -hp.min_level_db, 0, 1)

def amp_to_db(x):
    return 20 * np.log10(np.maximum(1e-5, x))

def melspectrogram(y, return_energy=False):
    # === mel spectrogram ===#
    spectrum                   = stft(y)
    linear_spectrogram         = np.abs(spectrum)
    mel_spectrogram            = linear_to_mel(linear_spectrogram)
    db_mel_spectrogram         = amp_to_db(mel_spectrogram)
    normalized_mel_spectrogram = normalize(db_mel_spectrogram)
    
    if not return_energy:
        return normalized_mel_spectrogram
    else:
        # === energy === #
        magnitude, phase = librosa.magphase(spectrum)
        energy = np.sqrt(np.sum(magnitude ** 2, axis=0))
        
        return normalized_mel_spectrogram, energy

