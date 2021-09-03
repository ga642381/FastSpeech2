import torch
import numpy as np

"""
MelGAN NeurIPS 2019 
ref : https://github.com/descriptinc/melgan-neurips
"""


class Vocoder():
    def __init__(self, name="melgan"):
        self.name = name
        if name == "melgan":
            """
            mel spectrogram config:
            * sampling_rate : 22050
            * n_fft : 1024
            * win_length : 1024
            * hop_length : 256
            * n_mels : 80
            * mel_fmin : 0.0
            * mel_fmax : None
            """
            self.model = torch.hub.load(
                'descriptinc/melgan-neurips', 'load_melgan', 'multi_speaker')

    def wav2mel(self, wav: torch.Tensor) -> torch.Tensor:
        return self.model(wav)

    def mel2wav(self, mel: torch.Tensor) -> torch.Tensor:
        return self.model.inverse(mel)


def wav2mel_np(wav: np.ndarray, vocoder) -> np.ndarray:
    """
    Args:
    * wav : numpy array, shape (n_samples, )
    * vocoder : Vocoder (melgan)

    Return:
    * mel spectrogram : numpy array, shape (n_mels, time)
    """
    mel_tensor = vocoder.wav2mel(torch.tensor(wav).unsqueeze(0)).squeeze(0)
    return mel_tensor.squeeze(0).cpu().numpy()


def mel2wav_np(mel: np.ndarray, vocoder) -> np.ndarray:
    pass
