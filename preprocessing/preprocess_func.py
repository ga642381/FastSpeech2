import os
from pathlib import Path
import librosa
from tqdm import tqdm
import shutil


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