from typing import List
import os
import numpy as np
from pathlib import Path
import librosa
from tqdm import tqdm
import shutil
from scipy.interpolate import interp1d
from functools import partial

from dlhlp_lib.parsers.Interfaces import BaseDataParser
from dlhlp_lib.audio.features import Energy, LogMelSpectrogram, get_feature_from_wav, get_f0_from_wav
from dlhlp_lib.audio.tools import wav_normalization

from utils.alignment import get_textgird_contents, get_alignment
from config import hparams as hp


ENERGY_NN = Energy(
    n_fft=hp.n_fft,
    win_length=hp.win_length,
    hop_length=hp.hop_length
)
MEL_NN = LogMelSpectrogram(
    sample_rate=hp.sampling_rate,
    n_fft=hp.n_fft,
    win_length=hp.win_length,
    hop_length=hp.hop_length,
    n_mels=hp.n_mels,
    pad=(hp.n_fft - hp.hop_length) // 2,
    power=1,
    norm="slaney",
    mel_scale="slaney"
)


wav2mel = partial(get_feature_from_wav, feature_nn=MEL_NN)
wav2energy = partial(get_feature_from_wav, feature_nn=ENERGY_NN)
wav2f0 = partial(
    get_f0_from_wav, sample_rate=hp.sampling_rate, hop_length=hp.hop_length
)


def denoise(data_dir: Path, save_dir: Path) -> None:
    # dir
    assert data_dir != save_dir, f"data_dir and save_dir should not be the same!"
    assert data_dir.exists(), f"{data_dir} does not exist!"
    print(f"[INFO] Task : Denoise with Facebook Demucs")
    print(f"[INFO] data_dir : {data_dir}")
    print(f"[INFO] save_dir : {save_dir}")

    # denoise
    tmp_dir = save_dir / "tmp"
    denoise_cmd = f"MKL_THREADING_LAYER=GNU python -m denoiser.enhance --master64 --noisy_dir {data_dir} --out_dir {tmp_dir} --batch_size 8 --device cuda"
    print(f"[INFO] Performing Facebook denoiser (demucs)")
    print(f"[INFO] Denoising command : {denoise_cmd}")
    os.system(denoise_cmd)

    # move results to correct locations
    wav_files = librosa.util.find_files(tmp_dir)
    enhanced_files = [Path(x) for x in wav_files if "enhanced" in Path(x).stem]
    for wav_path in tqdm(enhanced_files):
        output_path = save_dir / f"{wav_path.name.replace('_enhanced', '')}"
        shutil.move(wav_path, output_path)


def resample(
    dataset: BaseDataParser, query,
    input_featname: str,
    output_featname: str,
    sr: int,
) -> None:
    input_feat = dataset.get_feature(input_featname)
    output_feat = dataset.get_feature(output_featname)
    wav, _ = librosa.load(input_feat.read_filename(query, raw=True), sr=sr)
    output_feat.save(wav, query)


def process_utterance(
    dataset: BaseDataParser, query,
    wav_featname: str,
) -> None:
    textgrid_feat = dataset.get_feature("textgrid")
    wav_feat = dataset.get_feature(wav_featname)
    segment_feat = dataset.get_feature("mfa_segment")
    duration_feat = dataset.get_feature("mfa_duration")
    phoneme_feat = dataset.get_feature("phoneme")
    mel_feat = dataset.get_feature("mel")
    energy_feat = dataset.get_feature("energy")
    pitch_feat = dataset.get_feature("pitch")
    interp_pitch_feat = dataset.get_feature("interpolate_pitch")

    # === TextGrid ==== #
    textgrid = textgrid_feat.read_from_query(query)
    textgrid_phones = textgrid.get_tier_by_name("phones")
    textgrid_words = textgrid.get_tier_by_name("words")
    phones, segments = get_alignment(textgrid_phones)
    # words : ['but', 'after', 'a', 'time', ...]
    words = get_textgird_contents(textgrid_words)
    words = " ".join(words)

    # phones : ['B', 'AH1', 'T', 'AE1', 'F', 'T' ...]
    # durations : [3, 4, 4, 14, 3, 8 ...]
    # '{A}{B}{$}{C}', $ represents silent phones
    text = "{" + "}{".join(phones) + "}"
    text = text.replace("{$}", " ")  # '{A}{B} {C}'
    text = text.replace("}{", " ")  # '{A B} {C}'

    trim_start, trim_end = segments[0][0], segments[-1][1]
    assert trim_start < trim_end

    # === Wav ==== #
    inv_frame_period = hp.sampling_rate / hp.hop_length
    durations = []
    for (s, e) in segments:
        durations.append(
            int(
                round(round(e * inv_frame_period, 4))
                - round(round(s * inv_frame_period, 4))
            )
        )
    
    # === Duration ==== #
    total_length = sum(durations)
    sample_rate = hp.sampling_rate
    wav = wav_normalization(wav_feat.read_from_query(query))
    wav = wav[int(sample_rate * trim_start) : int(sample_rate * trim_end)]

    pitch = wav2f0(wav)[:total_length]  # (time, )
    mel = wav2mel(wav)[:, :total_length]  # (n_mels, time)
    energy = wav2energy(wav)[:total_length]  # (time, )

    # interpolate
    if np.sum(pitch != 0) <= 1:
        raise ValueError(f"Zero pitch detected {query}")
    nonzero_ids = np.where(pitch != 0)[0]
    interp_fn = interp1d(
        nonzero_ids,
        pitch[nonzero_ids],
        fill_value=(pitch[nonzero_ids[0]], pitch[nonzero_ids[-1]]),
        bounds_error=False,
    )
    interp_pitch = interp_fn(np.arange(0, len(pitch)))

    # Phoneme-level average
    avg_pitch_feat = dataset.get_feature(f"mfa_duration_avg_pitch")
    avg_energy_feat = dataset.get_feature(f"mfa_duration_avg_energy")
    avg_pitch, avg_energy = pitch[:], energy[:]
    avg_pitch = representation_average(avg_pitch, durations)
    avg_energy = representation_average(avg_energy, durations)

    # Saving
    phoneme_feat.save(text, query)
    segment_feat.save(segments, query)
    duration_feat.save(durations, query)
    mel_feat.save(mel, query)
    energy_feat.save(energy, query)
    pitch_feat.save(pitch, query)
    interp_pitch_feat.save(interp_pitch, query)
    avg_pitch_feat.save(avg_pitch, query)
    avg_energy_feat.save(avg_energy, query)


def representation_average(representation, durations, pad=0):
    pos = 0
    for i, d in enumerate(durations):
        if d > 0:
            representation[i] = np.mean(
                representation[pos: pos + d], axis=0)
        else:
            representation[i] = pad
        pos += d
    return representation[: len(durations)]
