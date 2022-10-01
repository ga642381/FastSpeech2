import argparse
import os
import shutil
from pathlib import Path
from random import sample

import librosa
import torchaudio
from tqdm import tqdm

os.environ["MKL_THREADING_LAYER"] = "GNU"


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("data_dir", type=str)
    parser.add_argument("save_dir", type=str)
    return vars(parser.parse_args())


def main(data_dir, save_dir):
    # dir
    data_dir = Path(data_dir).resolve()
    save_dir = Path(save_dir).resolve()
    assert data_dir != save_dir, f"data_dir and save_dir should not be the same!"
    assert data_dir.exists(), f"{data_dir} does not exist!"
    print(f"[INFO] Task : Denoise with Facebook Demucs")
    print(f"[INFO] data_dir : {data_dir}")
    print(f"[INFO] save_dir : {save_dir}")
    save_dir.mkdir(parents=True, exist_ok=True)

    # denoise
    tmp_dir = save_dir / "tmp"
    denoise_cmd = f"python -m denoiser.enhance --master64 --noisy_dir {data_dir} --out_dir {tmp_dir}"
    print(f"[INFO] Performing Facebook denoiser (demucs)")
    print(f"[INFO] Denoising command : {denoise_cmd}")
    os.system(denoise_cmd)

    # file structure
    wav_files = librosa.util.find_files(tmp_dir)
    enhanced_files = [Path(x) for x in wav_files if "enhanced" in Path(x).stem]
    for wav_path in tqdm(enhanced_files):
        spk, utt, _ = wav_path.stem.split("_")
        output_dir = save_dir / "wav48" / spk
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / f"{spk}_{utt}.wav"
        shutil.copy(wav_path, output_path)

    # copy text files
    txt_in_dir = data_dir / "txt"
    txt_out_dir = save_dir / "txt"
    cmd = f"cp -r {txt_in_dir} {txt_out_dir}"
    print(f'[INFO] Copying text files with command : "{cmd}"')
    os.system(cmd)


if __name__ == "__main__":
    torchaudio.set_audio_backend("sox_io")
    main(**parse_args())
