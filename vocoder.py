import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib
import matplotlib
matplotlib.use("Agg")
from matplotlib import pyplot as plt
from scipy.io import wavfile
import os
import librosa
import text
import hparams as hp
import soundfile

## Vocoders ##
def get_waveglow():
    waveglow = torch.hub.load('nvidia/DeepLearningExamples:torchhub', 'nvidia_waveglow')
    waveglow = waveglow.remove_weightnorm(waveglow)
    waveglow.eval()
    for m in waveglow.modules():
        if 'Conv' in str(type(m)):
            setattr(m, 'padding_mode', 'zeros')

    return waveglow

def waveglow_infer(mel, waveglow, path):
    with torch.no_grad():
        wav = waveglow.infer(mel, sigma=1.0) * hp.max_wav_value
        wav = wav.squeeze().cpu().numpy()
    soundfile.write(path, wav, hp.sampling_rate)
    

def get_melgan():
    melgan = torch.hub.load('descriptinc/melgan-neurips', 'load_melgan')
    return melgan

def melgan_infer(mel, melgan, path):
    wav = melgan.inverse(mel).squeeze(0).detach().cpu().numpy()
    soundfile.write(path, wav, hp.sampling_rate)
    
def melgan_infer_batch(mel, melgan):
    return melgan.inverse(mel).cpu().numpy()

        