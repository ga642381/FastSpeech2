import numpy as np
import pyworld as pw
import torch
import torch.nn as nn
from torchaudio.transforms import MelSpectrogram, Spectrogram


class LogMelSpectrogram(nn.Module):
    # 2021.08.07 Redundant, we used melgan's mel in wav_mel.py
    # However this class can be used for user defined mel spectrogram
    def __init__(
        self,
        sample_rate: int,
        n_fft: int,
        win_length: int,
        hop_length: int,
        f_min: float,
        n_mels: int,
    ):
        super().__init__()

        self.sample_rate = sample_rate
        self.n_fft = n_fft
        self.win_length = win_length
        self.hop_length = hop_length
        self.f_min = f_min
        self.n_mels = n_mels

        self.melspectrogram = MelSpectrogram(
            sample_rate=sample_rate,
            n_fft=n_fft,
            win_length=win_length,
            hop_length=hop_length,
            f_min=f_min,
            n_mels=n_mels,
            power=1,
            center=False,
        )

    def forward(self, wav_tensor: torch.Tensor) -> torch.Tensor:
        # 1. mel-spectrogram
        mel_tensor = self.melspectrogram(wav_tensor)

        # 2. log mel-spectrogram
        log_mel_tensor = torch.log10(torch.clamp(mel_tensor, min=1e-5))
        return log_mel_tensor


"""
TODO : possible to use espnet's energy extractor and dio 
* https://github.com/espnet/espnet/blob/master/espnet2/tts/feats_extract/energy.py
* https://github.com/espnet/espnet/blob/master/espnet2/tts/feats_extract/dio.py
"""


class Energy(nn.Module):
    def __init__(
        self, n_fft: int, win_length: int, hop_length: int,
    ):
        super().__init__()
        self.n_fft = n_fft
        self.win_length = win_length
        self.hop_length = hop_length

        # default is power of 2
        # we use power=1 to get a smaller range of energy.
        self.spectrogram = Spectrogram(
            n_fft=n_fft, hop_length=hop_length, win_length=win_length, power=1
        )

    def forward(self, wav: torch.Tensor) -> torch.Tensor:
        spectrogram = self.spectrogram(wav)
        return torch.norm(spectrogram, dim=1)


def get_feature_from_wav(wav, feature_nn: nn.Module) -> np.ndarray:
    wav = torch.tensor(wav).unsqueeze(0)
    with torch.no_grad():
        feature = feature_nn(wav).squeeze(0)
    return feature.numpy().astype(np.float32)


def get_f0_from_wav(wav, sample_rate, hop_length):
    """
    pyworld expectes wav in type double(float64)
    """
    f0, t = pw.dio(
        wav.astype(np.float64),
        sample_rate,
        frame_period=hop_length / sample_rate * 1000,
    )
    f0 = pw.stonemask(wav.astype(np.float64), f0, t, sample_rate).astype(np.float32)
    return f0
