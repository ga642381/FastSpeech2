import argparse
import os
from functools import partial
from multiprocessing import Pool, cpu_count
from pathlib import Path
from random import sample

import librosa
import torch
import torchaudio
from tqdm import tqdm

from utils.wav2mel import SoxEffects

"""
The reason we seperate vad from sox effect and make it an independent script is that
after adding the noise, vad might not detect human voice then return None. Therefore,
we would like to apply vad as a preprocessing step and we will not ever do vad again.

The pipeline is :
wav -> norm -> vad -> add_noise (-> mel, other features)

* Should do norm before vad!

The RULE is:
DON'T DO VAD AFTER ADDING NOISE
"""


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("data_dir", type=str)
    parser.add_argument("save_dir", type=str)
    parser.add_argument("--out_sample_rate", type=int, default=16000)
    return vars(parser.parse_args())


def process_save_wav(wav_file, processor, data_dir, save_dir, sr):
    output_path = wav_file.replace(str(data_dir), str(save_dir))
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    wav, _ = librosa.load(wav_file, sr=sr)
    output_wav = processor(torch.tensor(wav).unsqueeze(0), sr)
    if output_wav is not None:
        torchaudio.save(output_path, output_wav, sample_rate=sr)
    else:
        print(f"[WARNING] vad failed : {wav_file}")


def main(data_dir, save_dir, out_sample_rate):
    # dir
    data_dir = Path(data_dir).resolve()
    save_dir = Path(save_dir).resolve()
    assert data_dir != save_dir, f"data_dir and save_dir should not be the same!"
    assert data_dir.exists(), f"{data_dir} does not exist!"
    print(f"[INFO] Task : resample and vad")
    print(f"[INFO] data_dir : {data_dir}")
    print(f"[INFO] save_dir : {save_dir}")
    save_dir.mkdir(parents=True, exist_ok=True)

    # find wav files
    wav_files = librosa.util.find_files(data_dir)
    print(f"[INFO] {len(wav_files)} wav files found in {data_dir}")

    # add noise
    sox_effects = SoxEffects(resample=True, norm_vad=True, norm=False, sample_rate=out_sample_rate)
    file_to_processed_file = partial(
        process_save_wav,
        processor=sox_effects,
        data_dir=data_dir,
        save_dir=save_dir,
        sr=out_sample_rate,
    )

    N_processes = cpu_count()
    print(f"[INFO] Start multiprocessing with {N_processes} processes")
    with Pool(processes=N_processes) as pool:
        for _ in tqdm(
            pool.imap(file_to_processed_file, wav_files), total=len(wav_files)
        ):
            pass

    # if you have trouble in multiprocessing, use this :
    # for wav_file in tqdm(wav_files):
    #     file_to_processed_file(wav_file)

    # copy text files
    txt_in_dir = data_dir / "txt"
    txt_out_dir = save_dir / "txt"
    cmd = f"cp -r {txt_in_dir} {txt_out_dir}"
    print(f'[INFO] Copying text files with command : "{cmd}"')
    os.system(cmd)


if __name__ == "__main__":
    torchaudio.set_audio_backend("sox_io")
    main(**parse_args())
