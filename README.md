<p align="center">
    <br>
    <img src="https://raw.githubusercontent.com/ga642381/ga642381.github.io/main/assets/KW_Speech_logo.png" width="250"/>
    <br>
</p>
<p>
  <h3 align="center">
  Multi-Speaker FastSpeech 2 - PyTorch
  </h3>
</p>

</br>

## FastSpeech 2 - PyTorch Implementation :zap:

* This is a PyTorch implementation of Microsoft's [FastSpeech 2: Fast and High-Quality End-to-End Text to Speech](https://arxiv.org/abs/2006.04558). 

* This project is based on [ming024's implementation](https://github.com/ming024/FastSpeech2). Any suggestion for improvement is appreciated.

* Now supporting about **900 speakers** in :fire: **LibriTTS** for multi-speaker text-to-speech.

## Datasets :elephant:
This project supports 4 datasets, including muti-speaker datasets and single-speaker datasets:

### Multi-Speaker
:fire: **LibriTTS**

:fire: **VCTK**

### Single-Speaker
:fire: **LJSpeech**

:fire: **Blizzard2013**

After downloading the dataset, extract the compressed files. You have to modify the ``hp.data_path`` and some other parameters in ``hparams.py``. Default parameters are for the LibriTTS dataset.

## Preprocessing :pencil2:
Preprocessing contains 3 stages:
1. Preparing Alignment Data 
2. Montreal Force Alignmnet
3. Creating Training Dataset

For Montreal Force Alignment
```
wget https://github.com/MontrealCorpusTools/Montreal-Forced-Aligner/releases/download/v1.1.0-beta.2/montreal-forced-aligner_linux.tar.gz
tar -zxvf montreal-forced-aligner_linux.tar.gz
```
After downloading MFA, you should specify the path to MFA in ``hparams.py``.

```
python preprocess.py --prepare_align --mfa --create_dataset
```

After preprocessing, you will get a ``stat.txt`` file in your ``hp.preprocessed_path/``, recording the maximum and minimum values of the fundamental frequency and energy values throughout the entire corpus. You have to modify the f0 and energy parameters in the ``data/dataset.yaml`` according to the content of ``stat.txt``.

## Training :snake:

Train your model with
```
python train.py
```
The training output, including log message, checkpoint, and synthesized audios will be put in ``./log``
## References :notebook_with_decorative_cover:
- [FastSpeech 2: Fast and High-Quality End-to-End Text to Speech](https://arxiv.org/abs/2006.04558), Y. Ren, *et al*.
- [FastSpeech: Fast, Robust and Controllable Text to Speech](https://arxiv.org/abs/1905.09263), Y. Ren, *et al*.
- [xcmyz's FastSpeech implementation](https://github.com/xcmyz/FastSpeech)
- [rishikksh20's FastSpeech2 implementation](https://github.com/rishikksh20/FastSpeech2)
- [TensorSpeech's FastSpeech2 implementation](https://github.com/TensorSpeech/TensorflowTTS)
- [NVIDIA's WaveGlow implementation](https://github.com/NVIDIA/waveglow)
- [seungwonpark's MelGAN implementation](https://github.com/seungwonpark/melgan)
