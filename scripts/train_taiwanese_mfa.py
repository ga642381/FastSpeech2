"""
Script for training an taiwanese mfa aligner from scratch.
This is how we generate lexicons and pretrained mfa acoustic models on github.

Note that preprocessor's prepare_mfa() require modifications when running this script, since we 
train mfa with raw audios, but client prepares denoised audios in TAT corpus. To train mfa, 
you should link to wav_16000 instead of wav_16000_enhanced in TATPreprocessor.prepare_mfa.
"""
import os
from pathlib import Path
from multiprocessing import set_start_method

from Parsers import get_raw_parser, get_preprocessor


if __name__ == "__main__":
    from sys import platform
    if platform == "linux" or platform == "linux2":
        set_start_method("spawn", force=True)

    mfa_data_dir = Path("MFA/TAT/mfa_data")
    raw_parser = get_raw_parser("TATTTS")(Path("/mnt/d/Data/TAT-TTS"), Path("preprocessed/TAT-TTS"))
    prepocessor = get_preprocessor(Path("preprocessed/TAT-TTS"))
    raw_parser.parse(n_workers=8)
    prepocessor.prepare_mfa(mfa_data_dir)

    raw_parser = get_raw_parser("TAT")(Path("/mnt/d/Data/TAT"), Path("preprocessed/TAT"))
    prepocessor = get_preprocessor(Path("preprocessed/TAT"))
    raw_parser.parse(n_workers=8)
    prepocessor.prepare_mfa(mfa_data_dir)

    dictionary_path = "lexicon/taiwanese.txt"
    output_paths = "MFA/TAT/taiwanese_acoustic_model.zip"
    cmd = f"mfa train {str(mfa_data_dir)} {dictionary_path} {output_paths} -j 8 -v --clean"
    os.system(cmd)
