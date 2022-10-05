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
    f_min=hp.mel_fmin,
    n_mels=hp.n_mels,
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


SILENCE = ["sil", "sp", "spn"]


def textgrid2segment_and_phoneme(
    dataset: BaseDataParser, query,
    textgrid_featname: str,
    segment_featname: str,
    phoneme_featname: str
) -> None:
    textgrid_feat = dataset.get_feature(textgrid_featname)

    segment_feat = dataset.get_feature(segment_featname)
    phoneme_feat = dataset.get_feature(phoneme_featname)

    tier = textgrid_feat.read_from_query(query).get_tier_by_name("phones")
        
    phones = []
    durations = []
    end_idx = 0
    for t in tier._objects:
        s, e, p = t.start_time, t.end_time, t.text
        # Handle empty intervals
        if p == "":
            if s == 0:
                p = "sil"
            else:
                p = "sp"

        # Trim leading silences
        if phones == []:
            if p in SILENCE:
                continue
            else:
                pass
        phones.append(p)
        durations.append((s, e))
        if p not in SILENCE:
            end_idx = len(phones)

    durations = durations[:end_idx]
    phones = phones[:end_idx]

    segment_feat.save(durations, query)
    phoneme_feat.save(" ".join(phones), query)


def trim_wav_by_segment(
    dataset: BaseDataParser, query, sr: int,
    wav_featname: str,
    segment_featname: str,
    wav_trim_featname: str
) -> None:
    wav_feat = dataset.get_feature(wav_featname)
    segment_feat = dataset.get_feature(segment_featname)

    wav_trim_feat = dataset.get_feature(wav_trim_featname)

    wav = wav_feat.read_from_query(query)
    segment = segment_feat.read_from_query(query)

    wav_trim_feat.save(wav[int(sr * segment[0][0]) : int(sr * segment[-1][1])], query)


def wav_to_mel_energy_pitch(
    dataset: BaseDataParser, query,
    wav_featname: str,
    mel_featname: str,
    energy_featname: str,
    pitch_featname: str,
    interp_pitch_featname: str
) -> None:
    wav_feat = dataset.get_feature(wav_featname)

    mel_feat = dataset.get_feature(mel_featname)
    energy_feat = dataset.get_feature(energy_featname)
    pitch_feat = dataset.get_feature(pitch_featname)
    interp_pitch_feat = dataset.get_feature(interp_pitch_featname)

    wav = wav_normalization(wav_feat.read_from_query(query))

    pitch = wav2f0(wav)  # (time, )
    mel = wav2mel(wav)  # (n_mels, time)
    energy = wav2energy(wav)  # (time, )

    # interpolate
    nonzero_ids = np.where(pitch != 0)[0]
    interp_fn = interp1d(
        nonzero_ids,
        pitch[nonzero_ids],
        fill_value=(pitch[nonzero_ids[0]], pitch[nonzero_ids[-1]]),
        bounds_error=False,
    )
    interp_pitch = interp_fn(np.arange(0, len(pitch)))

    if np.sum(pitch != 0) <= 1:
        raise ValueError("Zero pitch detected")

    mel_feat.save(mel, query)
    energy_feat.save(energy, query)
    pitch_feat.save(pitch, query)
    interp_pitch_feat.save(interp_pitch, query)


def segment2duration(
    dataset: BaseDataParser, query, inv_frame_period: float,
    segment_featname: str, 
    duration_featname: str
) -> None:
    segment_feat = dataset.get_feature(segment_featname)

    duration_feat = dataset.get_feature(duration_featname)

    segment = segment_feat.read_from_query(query)
    durations = []
    for (s, e) in segment:
        durations.append(int(np.round(e * inv_frame_period) - np.round(s * inv_frame_period)))
    
    duration_feat.save(durations, query)


def duration_avg_pitch_and_energy(
    dataset: BaseDataParser, query,
    duration_featname: str,
    pitch_featname: str,
    energy_featname: str
) -> None:
    duration_feat = dataset.get_feature(duration_featname)
    pitch_feat = dataset.get_feature(pitch_featname)
    energy_feat = dataset.get_feature(energy_featname)

    avg_pitch_feat = dataset.get_feature(f"{duration_featname}_avg_pitch")
    avg_energy_feat = dataset.get_feature(f"{duration_featname}_avg_energy")

    durations = duration_feat.read_from_query(query)
    pitch = pitch_feat.read_from_query(query)
    energy = energy_feat.read_from_query(query)

    avg_pitch, avg_energy = pitch[:], energy[:]
    avg_pitch = representation_average(avg_pitch, durations)
    avg_energy = representation_average(avg_energy, durations)

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
