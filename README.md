## Multi-speaker FastSpeech 2 - PyTorch Implementation :zap:

* This is a PyTorch implementation of Microsoft's [FastSpeech 2: Fast and High-Quality End-to-End Text to Speech](https://arxiv.org/abs/2006.04558). 

* Now supporting about **900 speakers** in :fire: **LibriTTS** for multi-speaker text-to-speech.

<p align="center">
    <br>
    <img src="https://github.com/ga642381/FastSpeech2/blob/main/FastSpeech2.png" width="700"/>
    <br>
</p>

## Datasets :elephant:
This project supports 2 muti-speaker datasets:

### :fire: Single-Speaker
- **LJSpeech**

### :fire: Multi-Speaker
- **LibriTTS**

- **VCTK**

## Config

Configurations are in:
* config/dataset.yaml
* config/hparams.py

Please modify the dataest and mfa_path in hparams.

In this repo, we're using [MFA](https://github.com/MontrealCorpusTools/Montreal-Forced-Aligner) v1. Migrating to MFA v2 is a TODO item. 
    
## Steps
1. preprocess.py
2. train.py
3. synthesize.py

## 1. Preprocess
### File Structures:

[DATASET] / wavs / speaker / wav_files
[DATASET] / txts / speaker / txt_files

* wav_dir : the folder containing speaker dirs  ( [DATASET] / wavs )
* txt_dir : the folder containing speaker dirs ( [DATASET] / txts )
* save_dir : the output directory (e.g. "./processed" )
* -\-prepare_mfa : create mfa_data
* -\-mfa : create textgrid files
* -\-create_dataset : generate mel, phone, f0 ....., metadata.json

### Example commands:
* LJSpeech:
``` shell
#run the script for organizing LJSpeech first
python ./script/organizeLJ.py

python preprocess.py /storage/tts2021/LJSpeech-organized/wavs /storage/tts2021/LJSpeech-organized/txts ./processed/LJSpeech --prepare_mfa --mfa --create_dataset
```

* LibriTTS:
``` shell 
python preprocess.py /storage/tts2021//LibriTTS/train-clean-360 /storage/tts2021//LibriTTS/train-clean-360 ./processed/LibriTTS --prepare_mfa --mfa --create_dataset
```

* VCTK:
``` shell
python preprocess.py /storage/tts2021/VCTK-Corpus/wav48/ /storage/tts2021/VCTK-Corpus/txt ./processed/VCTK --prepare_mfa --mfa --create_dataset
```
### metadata.json includes:
1. spker table
2. traning data
3. validation data

## 2. Train
* data_dir : the preprocessed data directory
* -\-comment: some comments

### Example commands:
* LJSpeech:
``` shell
python train.py ./processed/LJSpeech --comment "Hello LJSpeech" 
```
* LibriTTS:
``` shell 
python train.py ./processed/LibriTTS --comment "Hello LibriTTS" 
```

* VCTK:
``` shell
python train.py ./processed/VCTK --comment "Hello VCTK"
```
## 3. Synthesize
* -\-ckpt_path: the checkpoint path
* -\-output_dir: the directory to put the synthesized audios

### Example commands:
``` shell
python synthesize.py --ckpt_path ./records/LJSpeech_2021-11-22-22:42/ckpt/checkpoint_125000.pth.tar --output_dir ./output
```

## References :notebook_with_decorative_cover:
- [FastSpeech 2: Fast and High-Quality End-to-End Text to Speech](https://arxiv.org/abs/2006.04558), Y. Ren, *et al*.
- [FastSpeech: Fast, Robust and Controllable Text to Speech](https://arxiv.org/abs/1905.09263), Y. Ren, *et al*.
- [xcmyz's FastSpeech implementation](https://github.com/xcmyz/FastSpeech)
- [rishikksh20's FastSpeech2 implementation](https://github.com/rishikksh20/FastSpeech2)
- [TensorSpeech's FastSpeech2 implementation](https://github.com/TensorSpeech/TensorflowTTS)
- [NVIDIA's WaveGlow implementation](https://github.com/NVIDIA/waveglow)
- [seungwonpark's MelGAN implementation](https://github.com/seungwonpark/melgan)
