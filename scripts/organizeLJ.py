import argparse
import csv
import os
from pathlib import Path

from genericpath import exists


def main(args):
    LJ_dir = Path(args.LJ_dir).resolve()
    out_dir = Path(args.out_dir).resolve()
    out_wav_dir = out_dir / "wavs" / "LJ"
    out_txt_dir = out_dir / "txts" / "LJ"
    out_wav_dir.mkdir(parents=True, exist_ok=True)
    out_txt_dir.mkdir(parents=True, exist_ok=True)

    # === organize wav === #
    for wav_file in (LJ_dir / "wavs").rglob("*.wav"):
        link_file = out_wav_dir / wav_file.name
        if not link_file.exists():
            link_file.symlink_to(wav_file.resolve())

    # === organize txt === #
    with open(LJ_dir / "metadata.csv", "r") as f:
        data = f.readlines()
        # data = csv.reader(f, delimiter="|")
        for row in data:
            row = row.split("|")
            file_name = row[0] + ".normalized.txt"
            normalized_text = row[2]
            with open(out_txt_dir / file_name, "w") as f:
                f.write(normalized_text)


if __name__ == "__main__":
    """
    e.g. python organizeLJ.py [path]/LJSpeech-1.1 [path]/LJSpeech_organized
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("LJ_dir", type=str)
    parser.add_argument("out_dir", type=str)
    args = parser.parse_args()

    main(args)
