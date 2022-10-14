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

from dlhlp_lib.vocoders import get_vocoder

import audio as Audio
import utils
from config import hparams as hp
from model.fastspeech2 import FastSpeech2
from text import clean_text, sequence_to_text, text_to_sequence
from utils import get_mask_from_lengths
from utils.pad import pad_1D


from typing import List
from tqdm import tqdm

from dlhlp_lib.utils.generators import batchify

class LexiconBasedG2p(object):
    """
    Rule based G2p using fix lexicons, can not handle OOV.
    """
    def __init__(self, lexicon_path: str):
        self.lexicon = {}
        with open(lexicon_path, 'r', encoding='utf-8') as f:
            for line in f:
                if line == '\n':
                    continue
                word, phns = line.split('\t')
                self.lexicon[word] = phns.strip().split(" ")

    def check_oov(self, text: List[str]) -> List[str]:
        return [word for word in text.split(" ") if word not in self.lexicon]
    
    def __call__(self, text: List[str]) -> List[str]:
        res = []
        words = text.split(" ")
        for word in words:
            res.extend(self.lexicon.get(word, ["sp"]))
        return res


class Synthesizer:
    def __init__(self, args):
        self.ckpt_path = Path(args.ckpt_path).resolve()
        self.output_dir = Path(args.output_dir).resolve()
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        self.__init_tts(self.ckpt_path)
        self._init_g2p()
        self._init_vocoder()

    def __init_tts(self, ckpt_path):
        state = torch.load(ckpt_path)
        n_spkers = state["model"]["module.spker_embeds.weight"].shape[0]
        model = nn.DataParallel(FastSpeech2(n_spkers))
        model.load_state_dict(state["model"])
        model.requires_grad_ = False
        model.eval()

        self.tts_model = model
        self.tts_model.to(self.device)

    def _init_vocoder(self):
        if hp.lang_id == "twn":
            self.vocoder = get_vocoder("HifiGAN")(dirpath="../model-lib/hifigan/TAT")
        elif hp.lang_id == "en":
            self.vocoder = get_vocoder("MelGAN")()
        else:
            raise NotImplementedError
        self.vocoder.to(self.device)
    
    def _init_g2p(self):
        if hp.lang_id == "twn":
            self.g2p = LexiconBasedG2p("lexicon/taiwanese.txt")
        elif hp.lang_id == "en":
            self.g2p = G2p()
        else:
            raise NotImplementedError

    def __process_text(self, text):
        text_cleaned = clean_text(text, hp.text_cleaners)
        phone = self.g2p(text_cleaned)
        phone = [p for p in phone if p != " "]
        phone = "{" + "}{".join(phone) + "}"
        phone = re.sub(r"\{[^\w\s]?\}", "{sp}", phone)
        phone = phone.replace("}{", " ")
        text = text_to_sequence(phone, hp.text_cleaners, hp.lang_id)
        return text

    def synthesize(self, sentences: List[List[str]], batch_size=16):
        oov_info = {}
        for i, texts in tqdm(enumerate(batchify(sentences, batch_size=batch_size))):
            output_names = [str(o) for o in range(i * batch_size, i * batch_size + len(texts))]

            # If G2p support ICheckOOV interface
            if 1:
                for text, filename in zip(texts, output_names):
                    oov = self.g2p.check_oov(text)
                    if len(oov) > 0:
                        oov_info[filename] = oov

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
                wav_lens = [m * hp.hop_length for m in mel_lens]
                wavs = self.vocoder.infer(model_pred[1].transpose(1, 2), wav_lens)

                utils.save_audios(
                    wavs,
                    wav_lens=wav_lens,
                    data_ids=output_names,
                    save_dir=self.output_dir,
                )
        
        if len(oov_info) > 0:
            with open(self.output_dir / "oov.json", 'w', encoding='utf-8') as f:
                json.dump(oov_info, f, indent=4)


if __name__ == "__main__":
    """
    e.g. python synthesize.py --ckpt_path ./records/LJSpeech_2021-11-22-22:42/ckpt/checkpoint_125000.pth.tar --output_dir ./output
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt_path", type=str)
    parser.add_argument("--output_dir", type=str, default="./output")
    # parser.add_argument("--vocoder", type=str, default="melgan")
    args = parser.parse_args()

    import json
    with open("aishell_test.json", 'r', encoding='utf-8') as f:
        info = json.load(f)
    sentences = [x["臺語"] for x in info]

    # sentences = [
    #     "Weather forecast for tonight: dark.",
    #     "I put a dollar in a change machine. Nothing changed.",
    #     "“No comment” is a comment.",
    #     "So far, this is the oldest I’ve been.",
    #     "I am in shape. Round is a shape.",
    # ]

    tts = Synthesizer(args)
    tts.synthesize(sentences, batch_size=16)
