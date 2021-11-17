import numpy as np
import torch

"""
MelGAN NeurIPS 2019 
ref : https://github.com/descriptinc/melgan-neurips
"""


class Vocoder:
    def __init__(self, name="melgan"):
        self.name = name
        self.hop_length = 256
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
                "descriptinc/melgan-neurips", "load_melgan", "multi_speaker"
            )

    def wav2mel(self, wav, return_type: str = "tensor"):
        """
        Args:
        * wav : torch.Tensor, shape ()
        """
        if type(wav) == np.ndarray:
            wav = torch.tensor(wav)

        mel_tensor = self.model(wav.unsqueeze(0))

        if return_type == "tensor":
            """
            mel spectrogram : tensor
            """
            return mel_tensor

        elif return_type in ["np", "numpy"]:
            """
            Return:
            mel spectrogram : numpy array, shape (n_mels, time)
            """
            return mel_tensor.squeeze(0).cpu().numpy()

    def mel2wav(self, mel: torch.Tensor) -> torch.Tensor:
        return self.model.inverse(mel)
