import torch
from torch.utils.data import Dataset, DataLoader

import numpy as np
import math
import os
import random
import hparams as hp
from utils import pad_1D, pad_2D
from text import text_to_sequence, sequence_to_text

def average_to_phone_level(mel_level_attribute, phones_len):
    result = np.zeros(phones_len.shape)
    
    for i, _ in enumerate(phones_len):
        start = 0
        for j, d in enumerate(phones_len[i]):
            # calculate average value, if phone len is 0, then average is 0
            if start == start + d:
                average = 0
            else:
                average = np.mean(mel_level_attribute[i][start: start+d])
                
            result[i][j] = average
            start += d
    return result

class Dataset(Dataset):
    def __init__(self, filename="train.txt", sort=True):
        self.basename, self.text = self.process_meta(os.path.join(hp.preprocessed_path, filename))
        self.sort = sort
        
        if hp.dataset == "VCTK":
            from data import vctk
            self.spk_table, self.inv_spk_table = vctk.get_spk_table()
        if hp.dataset == "LibriTTS":
            from data import libritts
            self.spk_table, self.inv_spk_table = libritts.get_spk_table()
            
            
    def __len__(self):
        return len(self.text)

    def __getitem__(self, idx):
        basename = self.basename[idx]
        phone = np.array(text_to_sequence(self.text[idx], []))
        
        mel_path = os.path.join(hp.preprocessed_path, "mel", "{}-mel-{}.npy".format(hp.dataset, basename))
        mel_target = np.load(mel_path)
        
        D_path = os.path.join(hp.preprocessed_path, "alignment", "{}-ali-{}.npy".format(hp.dataset, basename))
        D = np.load(D_path)
        
        f0_path = os.path.join(hp.preprocessed_path, "f0", "{}-f0-{}.npy".format(hp.dataset, basename))
        f0 = np.load(f0_path)
        
        energy_path = os.path.join(hp.preprocessed_path, "energy", "{}-energy-{}.npy".format(hp.dataset, basename))
        energy = np.load(energy_path)
        
        sample = {"id"        : basename,
                  "text"      : phone,
                  "mel_target": mel_target,
                  "D"         : D,
                  "f0"        : f0,
                  "energy"    : energy}

        return sample
    
    def collate_fn(self, batch):
        # batchsize      : 256 samples
        # real_batchsize : 16  samples
        # cut_list       : 16 x 16
        # output         : 16 batches (256 samples)
        batchsize = len(batch)
        real_batchsize = int(math.sqrt(batchsize))
        
        len_arr = np.array([b["text"].shape[0] for b in batch]) # lens of 256 samples
        iex_arr = np.argsort(-len_arr)                        # [(i_max_len) ... (i_min_len)]

        # index of samples sorted by lens
        cut_lists = list()
        for i in range(real_batchsize):
            if self.sort:
                cut_lists.append(iex_arr[i * real_batchsize: (i+1) * real_batchsize])
            else:
                cut_lists.append(np.arange(i * real_batchsize, (i+1) * real_batchsize))
        
        output = list()
        for i in range(real_batchsize):
            output.append(self.reprocess(batch, cut_lists[i]))
        
        # shuffle batch of batchs to solve the problem that
        # during synth, it always synthesizes short(long) sentences
        # 2020.10.03 KaiWei
        
        random.shuffle(output)
        return output    

    def reprocess(self, batch, cut_list):
        # batch    : 256 samples
        # cut_list : 16 iices
        ids = [batch[i]["id"] for i in cut_list]
        if hp.use_spk_embed:
            if hp.dataset == "VCTK" or hp.dataset=="LibriTTS":
                spk_ids = [self.spk_table[_id.split("_")[0]] for _id in ids]
            else:
                raise NotImplementedError("Looking up datset {} speaker table not implemented".format(hp.dataset))
                
        texts       = [batch[i]["text"]       for i in cut_list]
        mel_targets = [batch[i]["mel_target"] for i in cut_list]
        Ds          = [batch[i]["D"]          for i in cut_list]
        f0s         = [batch[i]["f0"]         for i in cut_list]
        energies    = [batch[i]["energy"]     for i in cut_list]
        
        for text, D, id_ in zip(texts, Ds, ids):
            if len(text) != len(D):
                print(text, text.shape, D, D.shape, id_)
                
        length_text = np.array(list())
        for text in texts:
            length_text = np.append(length_text, text.shape[0])

        length_mel = np.array(list())
        for mel in mel_targets:
            length_mel = np.append(length_mel, mel.shape[0])
        
        texts       = pad_1D(texts)
        Ds          = pad_1D(Ds)
        mel_targets = pad_2D(mel_targets)
        f0s         = pad_1D(f0s)
        energies    = pad_1D(energies)
        log_Ds      = np.log(Ds + hp.log_offset)
        
        # pitch and energy from mel level to phone level 2020.11.26 kaiwei
        f0s      = average_to_phone_level(f0s, Ds)
        energies = average_to_phone_level(energies, Ds)
        #
        
        out = {"id"        : ids,
               "text"      : texts,
               "mel_target": mel_targets,
               "D"         : Ds,
               "log_D"     : log_Ds,
               "f0"        : f0s,
               "energy"    : energies,
               "src_len"   : length_text,
               "mel_len"   : length_mel}
        if hp.use_spk_embed:
            out.update({"spk_ids": spk_ids})
    
        return out
    
    def process_meta(self, meta_path):
        with open(meta_path, "r", encoding="utf-8") as f:
            text = []
            name = []
            for line in f.readlines():
                #100_122655_000027_000001|{HH IY1 R IH0 Z UW1 M D}
                n, t = line.strip('\n').split('|')
                name.append(n)
                text.append(t)
            return name, text

if __name__ == "__main__":
    # Test
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    dataset = Dataset('val.txt')
    training_loader = DataLoader(dataset, batch_size=1, shuffle=False, collate_fn=dataset.collate_fn,
        drop_last=True, num_workers=0)
    total_step = hp.epochs * len(training_loader) * hp.batch_size

    cnt = 0
    for i, batchs in enumerate(training_loader):
        for j, data_of_batch in enumerate(batchs):
            mel_target = torch.from_numpy(
                data_of_batch["mel_target"]).float().to(device)
            D = torch.from_numpy(data_of_batch["D"]).int().to(device)
            if mel_target.shape[1] == D.sum().item():
                cnt += 1

    print(cnt, len(dataset))
