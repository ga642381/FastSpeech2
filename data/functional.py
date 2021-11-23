import json
import os
import random
from functools import partial
from pathlib import Path

import librosa
import numpy as np
import tgt
from audio.feature import Energy, get_f0_from_wav, get_feature_from_wav
from audio.wavmel import Vocoder
from config import hparams as hp
from text import clean_text
from tqdm import tqdm
from utils.alignment import get_alignment, get_textgird_contents

energy_nn = Energy(n_fft=hp.n_fft, win_length=hp.win_length, hop_length=hp.hop_length,)

vocoder = Vocoder("melgan")
# wav2mel = partial(wav2mel_np, vocoder=vocoder)
wav2energy = partial(get_feature_from_wav, feature_nn=energy_nn)
wav2f0 = partial(
    get_f0_from_wav, sample_rate=hp.sampling_rate, hop_length=hp.hop_length
)


def _get_spker_table(spkers: list) -> dict:
    spk_table = {}
    for i, s in enumerate(spkers):
        spk_table[str(s)] = i
    return spk_table


def _save_text(path, text):
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)


def prepare_mfa(wav_dir: Path, txt_dir: Path, mfa_data_dir: Path):
    """
    * Use [filename].normalized.txt ot generate [filename].txt
    * [filename].txt is what we want
    """
    spker_dirs = [d for d in wav_dir.iterdir() if d.is_dir()]
    print(f'[INFO] {len(spker_dirs)} speakers were found in wav dir : "{wav_dir}"')
    for spker_dir in tqdm(spker_dirs):
        # 1. create a similar structure in "mfa_data_dir" as in "wav_dir"
        target_dir = mfa_data_dir / spker_dir.relative_to(wav_dir)
        target_dir.mkdir(parents=True, exist_ok=True)

        for wav_file in spker_dir.rglob("*.wav"):
            # 2. create symlink for wav file
            link_file = target_dir / wav_file.name
            if not link_file.exists():
                link_file.symlink_to(wav_file.resolve())

            # 3. clean text
            if hp.dataset in ["LibriTTS", "LJSpeech"]:
                txt_file = (txt_dir / wav_file.relative_to(wav_dir)).with_suffix(
                    ".normalized.txt"
                )
            else:
                txt_file = (txt_dir / wav_file.relative_to(wav_dir)).with_suffix(".txt")

            if not txt_file.exists():
                continue

            with open(txt_file, "r", encoding="utf-8") as f:
                lines = f.readlines()
                assert len(lines) == 1
                text = clean_text(lines[0], hp.text_cleaners)
            # make whether two or single suffixes ".txt"
            with open(
                target_dir / txt_file.with_suffix("").with_suffix(".txt").name,
                "w",
                encoding="utf-8",
            ) as f:
                f.write(text)


def process_utterance(
    in_dir: Path, out_dir: Path, spk: Path, subdir: Path, basename: str
):
    """
    * The most important function in preprocessing.

    * You can set some constrains by returning None.
    """
    # === Paths === #
    wav_path = in_dir / spk / subdir / f"{basename}.wav"
    textgrid_path = out_dir / "TextGrid" / spk / f"{basename}.TextGrid"
    if not textgrid_path.exists() or not wav_path.exists():
        return None

    # === Text === #
    textgrid = tgt.io.read_textgrid(textgrid_path)
    textgrid_phones = textgrid.get_tier_by_name("phones")
    textgrid_words = textgrid.get_tier_by_name("words")
    phones, durations, trim_start, trim_end = get_alignment(
        textgrid_phones, hp.sampling_rate, hp.hop_length
    )
    # words : ['but', 'after', 'a', 'time', ...]
    words = get_textgird_contents(textgrid_words)
    words = " ".join(words)

    # phones : ['B', 'AH1', 'T', 'AE1', 'F', 'T' ...]
    # durations : [3, 4, 4, 14, 3, 8 ...]
    # '{A}{B}{$}{C}', $ represents silent phones
    text = "{" + "}{".join(phones) + "}"
    text = text.replace("{$}", " ")  # '{A}{B} {C}'
    text = text.replace("}{", " ")  # '{A B} {C}'

    if trim_start >= trim_end:
        return None

    # === Wav ==== #
    total_length = sum(durations)
    sample_rate = hp.sampling_rate

    wav, _ = librosa.load(wav_path, sr=sample_rate)
    wav = wav[int(sample_rate * trim_start) : int(sample_rate * trim_end)]

    f0 = wav2f0(wav)[:total_length]  # (time, )
    mel = vocoder.wav2mel(wav, return_type="np")[:, :total_length]  # (n_mels, time)
    energy = wav2energy(wav)[:total_length]  # (time, )

    if mel.shape[1] >= hp.max_seq_len or mel.shape[1] <= hp.min_seq_len:
        return None
    try:
        # if the shape is not right, you can check get_alignment function
        assert f0.shape[0] == energy.shape[0] == mel.shape[1]
    except AssertionError as e:
        print("duration problem: {}".format(wav_path))
        return None

    try:
        f_max = max(f0)
        f_min = min([f for f in f0 if f > 0])
        e_max = max(energy)
        e_min = min(energy)
    except Exception as e:
        print(f"[INFO] error occured while processing {basename} : {e}")
        return None

    # === Save Data === #
    # 1. Save alignment
    ali_filename = f"{basename}.npy"
    ali_path = out_dir / "alignment" / spk / ali_filename
    ali_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(ali_path, durations, allow_pickle=False)

    # 2. Save fundamental prequency (time, )
    f0_filename = f"{basename}.npy"
    f0_path = out_dir / "f0" / spk / f0_filename
    f0_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(f0_path, f0, allow_pickle=False)

    # 3. Save energy (time, )
    energy_filename = f"{basename}.npy"
    energy_path = out_dir / "energy" / spk / energy_filename
    energy_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(energy_path, energy, allow_pickle=False)

    # 4. Save mel spectrogram : (time, n_mels)
    mel_filename = f"{basename}.npy"
    mel_path = out_dir / "mel" / spk / mel_filename
    mel_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(mel_path, mel.T, allow_pickle=False)

    # 5. Save text
    text_filename = f"{basename}.txt"
    text_path = out_dir / "text" / spk / text_filename
    text_path.parent.mkdir(parents=True, exist_ok=True)
    _save_text(text_path, text)

    # 6. Write Metadata
    n_frames = mel.shape[1]
    utt_data = {
        "data_id": basename,
        "spker": spk,
        "text": {"path": str(text_path), "phones": text, "words": words},
        "wav": {"path": str(wav_path)},
        "mel": {"path": str(mel_path), "length": str(n_frames)},
        "alignment": {"path": str(ali_path)},
        "f0": {"path": str(f0_path), "max": str(f_max), "min": str(f_min)},
        "energy": {"path": str(energy_path), "max": str(e_max), "min": str(e_min)},
    }
    return utt_data


def build_dataset(in_dir, out_dir, n_val_per_spker=1):
    """
    * The in_dir should contain speaker's dir
    """
    train = list()
    valid = list()

    spker_dirs = [d for d in in_dir.iterdir() if d.is_dir()]
    spkers = [x.stem for x in spker_dirs]
    spker_dirs.sort()
    spker_table = _get_spker_table(spkers)
    print(f"[INFO] Total Speakers : {len(spkers)}")

    if not Path(out_dir / "TextGrid").exists():
        raise FileNotFoundError(f'[ERROR] "TextGird" not found in {out_dir}')

    # !TODO multiprocessing is possible here
    random.seed(829)
    for spker_dir in tqdm(spker_dirs):
        file_paths = []
        for file_path in spker_dir.rglob("*.wav"):
            # * subdir : PosixPath('100/121674')
            # * basename : 100_121674_000046_000001
            spk = spker_dir.stem
            subdir = file_path.parent.relative_to(in_dir).relative_to(spk)
            basename = file_path.stem
            file_paths.append((spk, subdir, basename))

        random.shuffle(file_paths)
        valid_num = 0
        for file_path in file_paths:
            # ! core of processing utterence
            spk = file_path[0]
            subdir = file_path[1]
            basename = file_path[2]

            utt_result = process_utterance(in_dir, out_dir, spk, subdir, basename)

            if utt_result is not None:
                if valid_num < n_val_per_spker:
                    valid.append(utt_result)
                    valid_num += 1
                else:
                    train.append(utt_result)

    ### Save data ###
    with open(out_dir / "metadata.json", "w", encoding="utf-8") as f:
        data = {"spker_table": spker_table, "train": train, "valid": valid}
        json.dump(data, f, indent=4)
