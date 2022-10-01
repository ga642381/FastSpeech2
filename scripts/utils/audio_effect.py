"""Wav2Mel for processing audio data."""

import torch
from torchaudio.sox_effects import apply_effects_tensor


class SoxEffects(torch.nn.Module):
    """Transform waveform tensors."""

    def __init__(
        self,
        resample: bool = True,
        norm_vad: bool = True,
        norm: bool = False,
        sample_rate: int = 16000,
        norm_db: float = -3,
    ):
        super().__init__()
        self.effects = [
            ["channels", "1"],  # convert to mono
        ]

        if resample:
            self.effects.extend([["rate", f"{sample_rate}"]])

        if norm_vad:
            # remove silence throughout the file; vad only trim beginning of the utternece
            # 1. use "reverse" to trim the end of the utterence
            # 2. norm before vad is recommended
            self.effects.extend([["norm"], ["vad"], ["reverse"], ["vad"], ["reverse"]])

        if norm:
            self.effects.extend([["norm", f"{norm_db}"]])

        # === print effect chains === #
        print(f"[INFO] sox_effects : {self.effects}")

    def forward(self, wav_tensor: torch.Tensor, sample_rate: int) -> torch.Tensor:
        wav_tensor, _ = apply_effects_tensor(wav_tensor, sample_rate, self.effects)
        if wav_tensor.numel() == 0:
            return None
        return wav_tensor
