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

from Parsers.TAT import TATPreprocessor
from Parsers.TAT_TTS import TATTTSPreprocessor
from preprocessing.preprocess_raw import preprocess_raw


if __name__ == "__main__":
    from sys import platform
    if platform == "linux" or platform == "linux2":
        set_start_method("spawn", force=True)

    # prepare initial features
    preprocess_raw("TATTTS", Path("/mnt/d/Data/TAT-TTS"), Path("preprocessed/TAT-TTS"), n_workers=4)
    preprocess_raw("TAT", Path("/mnt/d/Data/TAT"), Path("preprocessed/TAT"), n_workers=4)

    # mfa
    mfa_data_dir = Path("MFA/TAT/mfa_data")
    prepocessor = TATTTSPreprocessor(Path("preprocessed/TAT-TTS"))
    prepocessor.prepare_mfa(mfa_data_dir)
    prepocessor = TATPreprocessor(Path("preprocessed/TAT"))
    prepocessor.prepare_mfa(mfa_data_dir)

    os.makedirs("MFA/TAT", exist_ok=True)
    corpus_directory = "MFA/TAT/mfa_data"
    dictionary_path = "lexicon/taiwanese.txt"
    output_paths = "MFA/TAT/taiwanese_acoustic_model.zip"
    cmd = f"mfa train {corpus_directory} {dictionary_path} {output_paths} -j 8 -v --clean"
    os.system(cmd)