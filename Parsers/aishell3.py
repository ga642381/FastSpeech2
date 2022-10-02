import os
from tqdm import tqdm


class AISHELL3RawParser(object):
    def __init__(self, root):
        self.root = root
        self.data = None

    def parse(self):
        self.data = {"data": [], "data_info": [], "all_speakers": []}

        # train
        path = f"{self.root}/train/label_train-set.txt"
        with open(path, 'r', encoding='utf-8') as f:
            for i, line in tqdm(enumerate(f)):
                if i < 5 or line == '\n':
                    continue
                wav_name, text, _ = line.strip().split('|')
                speaker = wav_name[:-4]
                if speaker not in self.data["all_speakers"]:
                    self.data["all_speakers"].append(speaker)
                wav_path = f"{self.root}/train/wav/{speaker}/{wav_name}.wav"
                if os.path.isfile(wav_path):
                    data = {
                        "wav_path": wav_path,
                        "text": text,
                    }
                    data_info = {
                        "spk": speaker,
                        "basename": wav_name,
                        "dset": "train",
                    }
                    self.data["data"].append(data)
                    self.data["data_info"].append(data_info)
                else:
                    print("transcript.txt should not contain non-exist wav files, data might be corrupted.")
                    print(f"Can not find {wav_path}.")
        #TODO: test
