from posixpath import basename
import numpy as np
import os
import tgt
from tqdm import tqdm
import pyworld as pw
import torch
import librosa
import hparams as hp
import random
from functools import partial
from pathlib import Path

from audio.feature import Energy
from audio.feature import get_feature_from_wav, get_f0_from_wav
from audio.wav_mel import Vocoder
from audio.wav_mel import wav2mel_np
from utils import get_alignment
from text import _clean_text

energy_nn = Energy(
    n_fft=hp.n_fft,
    win_length=hp.win_length,
    hop_length=hp.hop_length,
)

vocoder = Vocoder("melgan")
wav2mel = partial(wav2mel_np, vocoder=vocoder)
wav2energy = partial(get_feature_from_wav, feature_nn=energy_nn)
wav2f0 = partial(get_f0_from_wav, sample_rate=hp.sampling_rate,
                 hop_length=hp.hop_length)


# def get_spk_table():
#     '''
#     spk_table     : {'14' :0, '16': 1, ...}
#     inv_spk_table : { 0:'14', 1: '16', ...}
#     '''
#     spk_table = {}
#     spk_id = 0

#     spks = os.listdir(hp.data_path)
#     spks.sort()
#     for spk in spks:
#         spk_table[spk] = spk_id
#         spk_id += 1
#     inv_spk_table = {v: k for k, v in spk_table.items()}
#     return spk_table, inv_spk_table


def prepare_mfa(wav_dir: Path, txt_dir: Path, mfa_data_dir: Path):
    """
    * Use [filename].normalized.txt ot generate [filename].txt
    * [filename].txt is what we want
    """
    spker_dirs = [d for d in wav_dir.iterdir() if d.is_dir()]
    print(f"{len(spker_dirs)} speakers were found in wav dir : \"{wav_dir}\"")
    for spker_dir in tqdm(spker_dirs):
        # 1. create a similar structure in "mfa_data_dir" as in "wav_dir"
        target_dir = mfa_data_dir / spker_dir.relative_to(wav_dir)
        target_dir.mkdir(parents=True, exist_ok=True)

        for wav_file in spker_dir.rglob("*.wav"):
            # 2. create symlink for wav file
            link_file = target_dir / wav_file.name
            if not link_file.exists():
                link_file.symlink_to(wav_file.absolute())

            # 3. clean text
            if hp.dataset == "LibriTTS":
                txt_file = (txt_dir / wav_file.relative_to(wav_dir)
                            ).with_suffix('.normalized.txt')
            else:
                txt_file = (txt_dir / wav_file.relative_to(wav_dir)
                            ).with_suffix('.txt')

            if not txt_file.exists():
                continue

            with open(txt_file, 'r', encoding='utf-8') as f:
                lines = f.readlines()
                assert(len(lines) == 1)
                text = _clean_text(lines[0], hp.text_cleaners)
            # make whether two or single suffixes ".txt"
            with open(target_dir / txt_file.with_suffix('').with_suffix(".txt").name, 'w', encoding='utf-8') as f:
                f.write(text)


def build_from_path(in_dir, out_dir):
    """
    * The in_dir should contain speaker's dir
    """
    train = list()
    val = list()
    f0_max = energy_max = 0
    f0_min = energy_min = 1000000
    n_frames = 0

    spker_dirs = [d for d in in_dir.iterdir() if d.is_dir()]
    spker_dirs.sort()
    print(f"Total Speakers : {len(spker_dirs)}")

    if not os.path.exists(os.path.join(out_dir, "TextGrid")):
        raise FileNotFoundError("\"TextGird\" not found in {}".format(out_dir))

    random.seed(829)
    for spker_dir in tqdm(spker_dirs):
        file_paths = []

        for file_path in spker_dir.rglob("*.wav"):
            # subdir, basename
            # * subdir : PosixPath('100/121674')
            # * basename : 100_121674_000046_000001
            spk = spker_dir.relative_to(in_dir)
            subdir = file_path.parent.relative_to(in_dir).relative_to(spk)
            basename = file_path.stem
            file_paths.append((spk, subdir, basename))

        random.shuffle(file_paths)
        for i, file_path in enumerate(file_paths):
            # ! core of processing utterence
            spk = file_path[0]
            subdir = file_path[1]
            basename = file_path[2]

            ret = process_utterance(
                in_dir, out_dir, spk, subdir, basename)

            if ret is None:
                continue
            else:
                info, f_max, f_min, e_max, e_min, n = ret

            if i == 0:
                val.append(info)
            else:
                train.append(info)

            f0_max = max(f0_max, f_max)
            f0_min = min(f0_min, f_min)
            energy_max = max(energy_max, e_max)
            energy_min = min(energy_min, e_min)
            n_frames += n

    ### Write Stats ###
    with open(os.path.join(out_dir, 'stat.txt'), 'w', encoding='utf-8') as f:
        strs = ['Total time: {} hours'.format(n_frames * hp.hop_length / hp.sampling_rate / 3600),
                'Total frames: {}'.format(n_frames),
                'Min F0: {}'.format(f0_min),
                'Max F0: {}'.format(f0_max),
                'Min energy: {}'.format(energy_min),
                'Max energy: {}'.format(energy_max)]
        for s in strs:
            print(s)
            f.write(s+'\n')

    return [r for r in train if r is not None], [r for r in val if r is not None]


def process_utterance(in_dir: Path, out_dir: Path, spk: Path, subdir: Path, basename: str):
    """
    * The most important function in preprocessing.

    * You can set some constrains by returning None.
    """

    wav_path = in_dir / spk / subdir / f'{basename}.wav'
    textgrid_path = out_dir / 'TextGrid' / spk / f'{basename}.TextGrid'
    if not textgrid_path.exists() or not wav_path.exists():
        return None

    # === Text === #
    textgrid = tgt.io.read_textgrid(textgrid_path)
    phone, duration, trim_start, trim_end = get_alignment(
        textgrid.get_tier_by_name('phones'), hp.sampling_rate, hp.hop_length)

    # '{A}{B}{$}{C}', $ represents silent phones
    text = '{' + '}{'.join(phone) + '}'
    text = text.replace('{$}', ' ')    # '{A}{B} {C}'
    text = text.replace('}{', ' ')     # '{A B} {C}'

    if trim_start >= trim_end:
        return None

    # === Wav ==== #
    total_length = sum(duration)
    sample_rate = hp.sampling_rate

    wav, _ = librosa.load(wav_path, sr=sample_rate)
    wav = wav[int(sample_rate*trim_start):int(sample_rate*trim_end)]

    f0 = wav2f0(wav)[:total_length]  # (time, )
    mel = wav2mel(wav)[:, :total_length]  # (n_mels, time)
    energy = wav2energy(wav)[:total_length]  # (time, )

    if mel.shape[1] >= hp.max_seq_len or mel.shape[1] <= hp.min_seq_len:
        return None

    try:
        # if the shape is not right, you can check get_alignment function
        assert(f0.shape[0] == energy.shape[0] == mel.shape[1])
    except AssertionError as e:
        print("duration problem: {}".format(wav_path))
        return None

    # === Save Data === #
    # 1. Save alignment
    ali_filename = f'{hp.dataset}-ali-{basename}.npy'
    np.save(os.path.join(out_dir, 'alignment', ali_filename),
            duration, allow_pickle=False)

    # 2. Save fundamental prequency
    f0_filename = f'{hp.dataset}-f0-{basename}.npy'
    np.save(os.path.join(out_dir, 'f0', f0_filename),
            f0, allow_pickle=False)

    # 3. Save energy
    energy_filename = f'{hp.dataset}-energy-{basename}.npy'
    np.save(os.path.join(out_dir, 'energy', energy_filename),
            energy, allow_pickle=False)

    # 4. Save mel spectrogram : (time, n_mels)
    mel_filename = f'{hp.dataset}-mel-{basename}.npy'
    np.save(os.path.join(out_dir, 'mel', mel_filename),
            mel.T, allow_pickle=False)

    try:
        return ('|'.join([basename, text]),
                max(f0),
                min([f for f in f0 if f > 0]),
                max(energy),
                min(energy),
                mel.shape[1])
    except:
        # print(basename)
        return None
