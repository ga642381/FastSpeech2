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
def get_melgan():
    melgan = torch.hub.load('descriptinc/melgan-neurips', 'load_melgan')
    return melgan

def melgan_infer(mel, melgan, path):
    wav = melgan.inverse(mel).squeeze(0).detach().cpu().numpy()
    soundfile.write(path, wav, hp.sampling_rate)
    
def melgan_infer_batch(mel, melgan):
    return melgan.inverse(mel).cpu().numpy()

        