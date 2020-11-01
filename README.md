# FastSpeech 2 - Pytorch Implementation

* This is a Pytorch implementation of Microsoft's text-to-speech system [**FastSpeech 2: Fast and High-Quality End-to-End Text to Speech**](https://arxiv.org/abs/2006.04558). 

* This project is based on [ming024's implementation](https://github.com/ming024/FastSpeech2). Any suggestion for improvement is appreciated.

## Datasets
This project supports 4 datasets, including muti-speaker datasets and single-speaker datasets:

### Multi speaker
* VCTK
* LibriTTS

### Single speaker
* LJSpeech
* Blizzard2013

After downloading the dataset, extract the compressed files, you have to modify the ``hp.data_path`` and some other parameters in ``hparams.py``. Default parameters are for the LibriTTS dataset.

## Preprocessing
Preprocessing contains 3 stages:
1. Preparing Alignment Data 
2. Montreal Force Alignmnet
3. Creating Training Dataset

For Montreal Force Alignment
```
wget https://github.com/MontrealCorpusTools/Montreal-Forced-Aligner/releases/download/v1.1.0-beta.2/montreal-forced-aligner_linux.tar.gz
tar -zxvf montreal-forced-aligner_linux.tar.gz
```
After downloading MFA, you should specify the path to MA in ``hparams.py``.

```
python preprocess.py --prepare_align --mfa --create_dataset
```

After preprocessing, you will get a ``stat.txt`` file in your ``hp.preprocessed_path/``, recording the maximum and minimum values of the fundamental frequency and energy values throughout the entire corpus. You have to modify the f0 and energy parameters in the ``data/dataset.yaml`` according to the content of ``stat.txt``.

## Training

Train your model with
```
python3 train.py
```
The training output, including log message, checkpoint, and synthesized audios will be put in ``./log``
# References
- [FastSpeech 2: Fast and High-Quality End-to-End Text to Speech](https://arxiv.org/abs/2006.04558), Y. Ren, *et al*.
- [FastSpeech: Fast, Robust and Controllable Text to Speech](https://arxiv.org/abs/1905.09263), Y. Ren, *et al*.
- [xcmyz's FastSpeech implementation](https://github.com/xcmyz/FastSpeech)
- [rishikksh20's FastSpeech2 implementation](https://github.com/rishikksh20/FastSpeech2)
- [TensorSpeech's FastSpeech2 implementation](https://github.com/TensorSpeech/TensorflowTTS)
- [NVIDIA's WaveGlow implementation](https://github.com/NVIDIA/waveglow)
- [seungwonpark's MelGAN implementation](https://github.com/seungwonpark/melgan)
