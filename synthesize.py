import argparse
import os
import random
import re
from string import punctuation

import numpy as np
import soundfile
import torch
import torch.nn as nn
from g2p_en import G2p

import audio as Audio
import hparams as hp
import utils
from fastspeech2 import FastSpeech2
from text import sequence_to_text, text_to_sequence
from utils.mask import get_mask_from_lengths

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
if hp.use_spk_embed:
    if hp.dataset == "VCTK":
        from data import vctk
        spk_table, inv_spk_table = vctk.get_spk_table()
        
    if hp.dataset == "LibriTTS":
        from data import libritts
        spk_table, inv_spk_table = libritts.get_spk_table()

def preprocess(text):        
    text = text.rstrip(punctuation)

    g2p = G2p()
    phone = g2p(text)
    phone = list(filter(lambda p: p != ' ', phone))
    phone = '{'+ '}{'.join(phone) + '}'
    phone = re.sub(r'\{[^\w\s]?\}', '{sp}', phone)
    phone = phone.replace('}{', ' ')
    print(text)
    print('|' + phone + '|')
    print("\n")
    #print(text_to_sequence(phone, hp.text_cleaners))
    sequence = np.array(text_to_sequence(phone, hp.text_cleaners))
    sequence = np.stack([sequence])

    return torch.from_numpy(sequence).long().to(device)

def get_FastSpeech2(num):
    checkpoint_path = os.path.join(hp.checkpoint_path, "checkpoint_{}.pth.tar".format(num))
    n_spkers = torch.load(checkpoint_path)['model']['module.embed_speakers.weight'].shape[0]
    
    if hp.use_spk_embed:    
        model = nn.DataParallel(FastSpeech2(True, n_spkers))
    else:
        model = nn.DataParallel(FastSpeech2())
        
    model.load_state_dict(torch.load(checkpoint_path)['model'])
    model.requires_grad = False
    model.eval()
    return model

def synthesize(model, waveglow, melgan, text, sentence, prefix=''):
    sentence = sentence[:150] # long filename will result in OS Error
    src_len = torch.from_numpy(np.array([text.shape[1]])).to(device)
    
    # create dir
    if not os.path.exists(os.path.join(hp.test_path, hp.dataset)):
        os.makedirs(os.path.join(hp.test_path, hp.dataset))    
    
    # generate wav
    if hp.use_spk_embed:
        hp.batch_size = 3
        # select speakers
        # TODO
        spk_ids = torch.tensor(list(inv_spk_table.keys())[5:5+hp.batch_size]).to(torch.int64).to(device)
        text = text.repeat(hp.batch_size, 1)
        src_len = src_len.repeat(hp.batch_size)
        mel, mel_postnet, log_duration_output, f0_output, energy_output, _, _, mel_len = model(text, src_len, speaker_ids=spk_ids)
        
        mel_mask = get_mask_from_lengths(mel_len, None)
        mel_mask = mel_mask.unsqueeze(-1).expand(mel_postnet.size())
        silence = (torch.ones(mel_postnet.size()) * -5).to(device)
        mel = torch.where(~mel_mask, mel, silence)
        mel_postnet = torch.where(~mel_mask, mel_postnet, silence)
        
        mel_torch = mel.transpose(1, 2).detach()
        mel_postnet_torch = mel_postnet.transpose(1, 2).detach()

        if waveglow is not None:
            wavs = utils.waveglow_infer_batch(mel_postnet_torch, waveglow)
        if melgan is not None:
            wavs = utils.melgan_infer_batch(mel_postnet_torch, melgan)        
            
        for i, spk_id in enumerate(spk_ids):
            spker = inv_spk_table[int(spk_id)]
            mel_postnet_i = mel_postnet[i].cpu().transpose(0, 1).detach()
            f0_i = f0_output[i].detach().cpu().numpy()
            energy_i = energy_output[i].detach().cpu().numpy()
            mel_mask_i = mel_mask[i]
            wav_i = wavs[i]
            
            # output
            base_dir_i = os.path.join(hp.test_path, hp.dataset, "step {}".format(args.step), spker)
            os.makedirs(base_dir_i, exist_ok=True)
            path_i = os.path.join(base_dir_i, '{}_{}_{}.wav'.format(prefix, hp.vocoder, sentence))
            soundfile.write(path_i, wav_i, hp.sampling_rate)
            utils.plot_data([(mel_postnet_i.numpy(), f0_i, energy_i)], 
                            ['Synthesized Spectrogram'], 
                            filename=os.path.join(base_dir_i, '{}_{}.png'.format(prefix, sentence)))
            
    else:
        spk_ids = None
        mel, mel_postnet, log_duration_output, f0_output, energy_output, _, _, mel_len = model(text, src_len, speaker_ids=spk_ids)
        mel_torch = mel.transpose(1, 2).detach()
        mel_postnet_torch = mel_postnet.transpose(1, 2).detach()
        mel = mel[0].cpu().transpose(0, 1).detach()
        mel_postnet = mel_postnet[0].cpu().transpose(0, 1).detach()
        f0_output = f0_output[0].detach().cpu().numpy()
        energy_output = energy_output[0].detach().cpu().numpy()
        
        Audio.tools.inv_mel_spec(mel_postnet, os.path.join(hp.test_path, '{}_griffin_lim_{}.wav'.format(prefix, sentence)))
        if waveglow is not None:
            utils.waveglow_infer(mel_postnet_torch, waveglow, os.path.join(hp.test_path, hp.dataset, '{}_{}_{}_{}.wav'.format(prefix, hp.vocoder, spker, sentence)))
        if melgan is not None:
            utils.melgan_infer(mel_postnet_torch, melgan, os.path.join(hp.test_path, hp.dataset, '{}_{}_{}_{}.wav'.format(prefix, hp.vocoder, spker, sentence)))
        
        utils.plot_data([(mel_postnet.numpy(), f0_output, energy_output)], ['Synthesized Spectrogram'], filename=os.path.join(hp.test_path, '{}_{}.png'.format(prefix, sentence)))


if __name__ == "__main__":
    # Test
    parser = argparse.ArgumentParser()
    parser.add_argument('--step', type=int, default=160000)
    parser.add_argument('--input', action="store_true", default=False)
    args = parser.parse_args()
    
    if args.input:
        sentence = input("Please enter an English sentence : ")
        sentences = [sentence]
        
    else:
        sentences = ["Weather forecast for tonight: dark.",
                     "I put a dollar in a change machine. Nothing changed.",
                     "“No comment” is a comment.",
                     "So far, this is the oldest I’ve been.",
                     "I am in shape. Round is a shape."
                ]
        
    model = get_FastSpeech2(args.step).to(device)
    melgan = waveglow = None
    if hp.vocoder == 'melgan':
        melgan = utils.get_melgan()
        
    elif hp.vocoder == 'waveglow':
        waveglow = utils.get_waveglow()
        waveglow.to(device)
        
    print("Synthesizing...")
    for sentence in sentences:
        text = preprocess(sentence)
        synthesize(model, waveglow, melgan, text, sentence, prefix='step_{}'.format(args.step))
