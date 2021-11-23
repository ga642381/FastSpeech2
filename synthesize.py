import argparse
import os
import random
import re
from pathlib import Path
from string import punctuation

import numpy as np
import soundfile
import torch
import torch.nn as nn
from g2p_en import G2p

import audio as Audio
import utils
from audio.wavmel import Vocoder
from config import hparams as hp
from model.fastspeech2 import FastSpeech2
from text import clean_text, sequence_to_text, text_to_sequence
from utils import get_mask_from_lengths
from utils.pad import pad_1D


class Synthesizer:
    def __init__(self, args):
        ckpt_path = Path(args.ckpt_path).resolve()
        self.tts_model = self.__init_tts(ckpt_path)
        self.vocoder = self.__init_vocoder("melgan")
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.g2p = G2p()
        pass

    def __init_tts(self, ckpt_path):
        state = torch.load(ckpt_path)
        n_spkers = state["model"]["module.spker_embeds.weight"].shape[0]
        model = nn.DataParallel(FastSpeech2(n_spkers))
        model.load_state_dict(state["model"])
        model.requires_grad_ = False
        model.eval()
        return model

    def __init_vocoder(self, name="melgan"):
        return Vocoder(name)

    def __process_text(self, text):
        text_cleaned = clean_text(text, hp.text_cleaners)
        phone = self.g2p(text_cleaned)
        phone = [p for p in phone if p != " "]
        phone = "{" + "}{".join(phone) + "}"
        phone = re.sub(r"\{[^\w\s]?\}", "{sp}", phone)
        phone = phone.replace("}{", " ")
        text = text_to_sequence(phone, hp.text_cleaners)
        return text

    def synthesize(self, texts: list):
        texts = [self.__process_text(text) for text in texts]
        texts = [np.array(text) for text in texts]
        text_lens = np.array([text.shape[0] for text in texts])
        text_lens = torch.from_numpy(text_lens).long().to(self.device)
        texts = pad_1D(texts)
        texts = torch.from_numpy(texts).long().to(self.device)
        spker_id = torch.tensor([0]).to(self.device)
        with torch.no_grad():
            (model_pred, text_mask, mel_mask, mel_lens) = self.tts_model(
                spker_id, texts, text_lens
            )
            wavs = self.vocoder.mel2wav(model_pred[1].transpose(1, 2))
            wav_lens = [m * self.vocoder.hop_length for m in mel_lens]
            utils.save_audios(
                wavs,
                wav_lens=wav_lens,
                data_ids=["test1", "test2", "test3", "test4", "test5"],
                save_dir=Path("./"),
            )


if __name__ == "__main__":
    model_path = "/fortress/tts2021/FastSpeech2/records/LJSpeech_2021-11-22-22:42/ckpt/checkpoint_125000.pth.tar"
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt_path", type=str, default=model_path)
    parser.add_argument("--output_path", type=str, default="./")
    parser.add_argument("--vocoder", type=str, default="melgan")
    args = parser.parse_args()

    sentences = [
        "Weather forecast for tonight: dark.",
        "I put a dollar in a change machine. Nothing changed.",
        "“No comment” is a comment.",
        "So far, this is the oldest I’ve been.",
        "I am in shape. Round is a shape.",
    ]

    tts = Synthesizer(args)
    tts.synthesize(sentences)
