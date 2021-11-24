import argparse
import os
from pathlib import Path

from config import hparams as hp

if hp.dataset in ["LibriTTS", "VCTK", "LJSpeech"]:
    from data import functional as F
else:
    raise NotImplementedError(
        "You should specify the dataset in hparams.py\
                               and write a corresponding file in data/"
    )


class Preprocessor:
    def __init__(self, args):
        self.args = args
        self.dataset = hp.dataset
        self.mfa_path = hp.mfa_path
        self.wav_dir = Path(args.wav_dir).resolve()
        self.txt_dir = Path(args.txt_dir).resolve()
        self.out_dir = Path(args.save_dir).resolve()

    def exec(self):
        self.print_message()
        key_input = ""
        while key_input not in ["y", "Y", "n", "N"]:
            key_input = input("Proceed? ([y/n])? ")

        if key_input in ["y", "Y"]:
            self.make_output_dirs(force=True)
            # 1. Prepare MFA
            if self.args.prepare_mfa:
                print("[INFO] Preparing data for Montreal Force Alignmnet...")
                self.prepare_mfa()
            # 2. MFA
            if self.args.mfa:
                print("[INFO] Performing Montreal Force Alignment...")
                self.mfa()
            # 3. Create Dataset
            if self.args.create_dataset:
                print("[INFO] Creating Training and Validation Dataset...")
                self.create_dataset()

    def print_message(self):
        print("\n")
        print("------ Preprocessing ------")
        print(f"* Dataset     : {self.dataset}")
        print(f"* Data(wav) path   : {self.wav_dir}")
        print(f"* Data(txt) path   : {self.txt_dir}")
        print(f"* Output path : {self.out_dir}")
        print("\n")
        print(" [INFO] The following will be executed:")
        if self.args.prepare_mfa:
            print("* Preparing data for Montreal Force Alignment")
        if self.args.mfa:
            print("* Montreal Force Alignmnet")
        if self.args.create_dataset:
            print("* Creating Training Dataset")
        print("\n")

    def make_output_dirs(self, force=False):
        out_dir = self.out_dir
        if self.args.mfa:
            self.mfa_out_dir = os.path.join(out_dir, "TextGrid")
            os.makedirs(self.mfa_out_dir, exist_ok=force)

        self.mfa_data_dir = os.path.join(out_dir, "mfa_data")
        os.makedirs(self.mfa_data_dir, exist_ok=force)

        self.mel_out_dir = os.path.join(out_dir, "mel")
        os.makedirs(self.mel_out_dir, exist_ok=force)

        self.ali_out_dir = os.path.join(out_dir, "alignment")
        os.makedirs(self.ali_out_dir, exist_ok=force)

        self.f0_out_dir = os.path.join(out_dir, "f0")
        os.makedirs(self.f0_out_dir, exist_ok=force)

        self.energy_out_dir = os.path.join(out_dir, "energy")
        os.makedirs(self.energy_out_dir, exist_ok=force)

    # === 1. Preapare Algin === #
    def prepare_mfa(self):
        F.prepare_mfa(self.wav_dir, self.txt_dir, self.mfa_data_dir)

    # === 2. MFA  === #
    def mfa(self):
        out_dir = self.out_dir
        mfa_path = self.mfa_path

        mfa_in_dir = self.mfa_data_dir
        mfa_out_dir = os.path.join(out_dir, "TextGrid")
        mfa_bin_path = os.path.join(mfa_path, "bin", "mfa_align")
        mfa_pretrain_path = os.path.join(
            mfa_path, "pretrained_models", "librispeech-lexicon.txt"
        )
        cmd = f"{mfa_bin_path} {mfa_in_dir} {mfa_pretrain_path} english {mfa_out_dir} -j 8 -v"
        os.system(cmd)

    # === 3. Create Dataset === #
    def create_dataset(self):
        """
        * metadata.json will be created
        * mel, energy, f0,... will be created
        """
        in_dir = self.wav_dir
        out_dir = self.out_dir

        F.build_dataset(in_dir, out_dir)


def main(args):
    P = Preprocessor(args)
    P.exec()


if __name__ == "__main__":
    """
    e.g.
    # LJSpeech #
        * run ./script/organizeLJ.py first
        * python preprocess.py /storage/tts2021/LJSpeech-organized/wavs /storage/tts2021/LJSpeech-organized/txts ./processed/LJSpeech --prepare_mfa --mfa --create_dataset
    
    # LibriTTS #
        * python preprocess.py /storage/tts2021//LibriTTS/train-clean-360 /storage/tts2021//LibriTTS/train-clean-360 ./processed/LibriTTS --prepare_mfa --mfa --create_dataset
    
    # VCTK #
        * python preprocess.py /storage/tts2021/VCTK-Corpus/wav48/ /storage/tts2021/VCTK-Corpus/txt ./processed/VCTK --prepare_mfa --mfa --create_dataset
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("wav_dir", type=str)
    parser.add_argument("txt_dir", type=str)
    parser.add_argument("save_dir", type=str)

    parser.add_argument("--prepare_mfa", action="store_true", default=False)
    parser.add_argument("--mfa", action="store_true", default=False)
    parser.add_argument("--create_dataset", action="store_true", default=False)
    args = parser.parse_args()

    main(args)
