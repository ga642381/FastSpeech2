import json
import math
import os
import random
from pathlib import Path

from config import hparams as hp
import numpy as np
import torch
from text import sequence_to_text, text_to_sequence
from torch.utils.data import DataLoader, Dataset
from utils.pad import pad_1D, pad_2D
from Parsers.parser import DataParser


# Handled by preprocess_func
# def average_to_phone_level(mel_level_attribute, phones_len):
#     result = np.zeros(phones_len.shape)

#     for i, _ in enumerate(phones_len):
#         start = 0
#         for j, d in enumerate(phones_len[i]):
#             # calculate average value, if phone len is 0, then average is 0
#             if start == start + d:
#                 average = 0
#             else:
#                 average = np.mean(mel_level_attribute[i][start : start + d])

#             result[i][j] = average
#             start += d
#     return result


class Dataset(Dataset):
    def __init__(self, data_dir, split, sort=True):
        self.data_parser = DataParser(data_dir)
        self.data_dir = data_dir
        self.split = split
        self.sort = sort
        self.metadata = self.get_metadata(data_dir)
        self.dataset = self.metadata[split]
        self.set_speakers(self.data_parser.get_all_speakers())

    def set_speakers(self, spkers):
        # support run time setting in order to train on multiple datasets
        self.spkers = spkers
        self.spker_table = {s: i for i, s in enumerate(spkers)}  # {spker: id}
    
    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        query = self.dataset[idx]
        phone_seq = text_to_sequence(self.data_parser.phoneme.read_from_query(query), [], hp.lang_id)
        data_id = query["basename"]
        spker = query["spk"]

        spker_id = self.spker_table[spker]
        phone = np.array(phone_seq)
        mel = self.data_parser.mel.read_from_query(query)
        D = self.data_parser.mfa_duration.read_from_query(query)
        f0 = self.data_parser.mfa_duration_avg_pitch.read_from_query(query)
        energy = self.data_parser.mfa_duration_avg_energy.read_from_query(query)
        sample = {
            "data_id": data_id,
            "spker_id": spker_id,
            "text": phone,
            "mel": mel.T,
            "D": D,
            "f0": f0,
            "energy": energy,
        }

        return sample

    # def __getitem__(self, idx):
    #     data = self.dataset[idx]
    #     phone_seq = text_to_sequence(data["text"]["phones"], [])
    #     data_id = data["data_id"]
    #     spker = data["spker"]
    #     mel_path = data["mel"]["path"]
    #     D_path = data["alignment"]["path"]
    #     f0_path = data["f0"]["path"]
    #     energy_path = data["energy"]["path"]

    #     spker_id = self.spker_table[spker]
    #     phone = np.array(phone_seq)
    #     mel = np.load(mel_path)
    #     D = np.load(D_path)
    #     f0 = np.load(f0_path)
    #     energy = np.load(energy_path)
    #     sample = {
    #         "data_id": data_id,
    #         "spker_id": spker_id,
    #         "text": phone,
    #         "mel": mel,
    #         "D": D,
    #         "f0": f0,
    #         "energy": energy,
    #     }

    #     return sample

    def collate_fn(self, batch):
        # samples_num   : 256 samples
        # batch_size    : 16  samples
        # seg_list      : 16 x 16
        # output        : 16 batches (256 samples)
        samples_num = len(batch)

        # last batch, contain few samples (during evaluation)
        # e.g. last 8 samples, and sorting is not required.
        if samples_num < hp.batch_size ** 2:
            batches_size = math.ceil(samples_num / hp.batch_size)
            idxs = np.arange(samples_num)
            seg_lists = np.array_split(idxs, batches_size)

        # collect batches with sorting
        else:
            batch_size = hp.batch_size
            batches_size = int(samples_num / batch_size)
            len_arr = np.array(
                [b["text"].shape[0] for b in batch]
            )  # lens of 256 samples
            iex_arr = np.argsort(-len_arr)  # [(i_max_len) ... (i_min_len)]

            # index of samples sorted by lens
            seg_lists = list()
            for i in range(batch_size):
                if self.sort:
                    seg_lists.append(iex_arr[i * batch_size : (i + 1) * batch_size])
                else:
                    seg_lists.append(np.arange(i * batch_size, (i + 1) * batch_size))

        output = list()
        for i in range(batches_size):
            output.append(self.reprocess(batch, seg_lists[i]))
        random.shuffle(output)
        return output

    def reprocess(self, batch: list, seg_list: list):
        # batch    : 256 samples
        # seg_list : 16 lists of lists
        data_ids = [batch[i]["data_id"] for i in seg_list]
        spker_ids = [batch[i]["spker_id"] for i in seg_list]
        texts = [batch[i]["text"] for i in seg_list]
        mels = [batch[i]["mel"] for i in seg_list]
        Ds = [batch[i]["D"] for i in seg_list]
        f0s = [batch[i]["f0"] for i in seg_list]
        energies = [batch[i]["energy"] for i in seg_list]

        for text, D in zip(texts, Ds):
            if len(text) != len(D):
                print(text, text.shape, D, D.shape)

        text_lens = np.array([text.shape[0] for text in texts])
        mel_lens = np.array([mel.shape[0] for mel in mels])

        spker_ids = np.array(spker_ids)
        texts = pad_1D(texts)
        Ds = pad_1D(Ds)
        mels = pad_2D(mels)
        f0s = pad_1D(f0s)
        energies = pad_1D(energies)
        log_Ds = np.log(Ds + hp.log_offset)

        # pitch and energy from mel level to phone level 2020.11.26 kaiwei
        # f0s = average_to_phone_level(f0s, Ds)
        # energies = average_to_phone_level(energies, Ds)

        out = {
            "data_id": data_ids,
            "spker_id": spker_ids,
            "text_seq": texts,
            "text_len": text_lens,
            "d": Ds,
            "log_d": log_Ds,
            "f0": f0s,
            "energy": energies,
            "mel": mels,
            "mel_len": mel_lens,
        }

        return out

    def get_metadata(self, data_dir):
        with open(f"{data_dir}/metadata.json", "r", encoding="utf-8") as f:
            metadata = json.load(f)
        return metadata


if __name__ == "__main__":
    """
    write some tests here
    """
    pass
