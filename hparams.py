import os


# Dataset
#dataset = "LJSpeech"
#data_path = "/dataset/LJSpeech-1.1"

#dataset = "Blizzard2013"
#data_path = "./Blizzard-2013/train/segmented/"

#dataset = "VCTK"
#data_path = "/home/kaiwei/SpeechNet/FastSpeech2/dataset/VCTK-Corpus"

dataset = "LibriTTS"
data_path = "/home/kaiwei/SpeechNet/FastSpeech2/dataset/LibriTTS/train-clean-360"

# Text
### g2p_en ###
text_cleaners = ['english_cleaners']


# Audio and mel
### for LJSpeech ###
#sampling_rate = 22050
#filter_length = 1024
#hop_length = 256
#win_length = 1024

### for Blizzard2013 ###
#sampling_rate = 16000
#filter_length = 800
#hop_length = 200
#win_length = 800

### for VCTK ###
#sampling_rate = 48000
#filter_length = 2048
#hop_length = 512
#win_length = 2048
#sampling_rate = 22050
#filter_length = 1024
#hop_length = 256
#win_length = 1024

### for LibriTTS ###
sampling_rate = 22050
filter_length = 1024
hop_length = 256
win_length = 1024


max_wav_value = 32768.0
n_mel_channels = 80
mel_fmin = 0.0
mel_fmax = None


# FastSpeech 2
encoder_layer = 4
encoder_head = 2
encoder_hidden = 256
decoder_layer = 4
decoder_head = 2
#decoder_hidden = 256
#decoder_hidden = 256 + 32 + 32 # encoder_hidden + 2+ spk_embed_dim
decoder_hidden = 256 + 128 + 128
fft_conv1d_filter_size = 1024
fft_conv1d_kernel_size = (9, 1)
encoder_dropout = 0.2
decoder_dropout = 0.2

variance_predictor_filter_size = 256
variance_predictor_kernel_size = 3
variance_predictor_dropout = 0.5

use_spk_embed = True
#spk_embed_integration_type = "concat"

max_seq_len = 1000


# Quantization for F0 and energy
# these stats can be known from preprocessing
### for LJSpeech ###
#f0_min = 71.0
#f0_max = 795.8
#energy_min = 0.0
#energy_max = 315.0

### for Blizzard2013 ###
#f0_min = 71.0
#f0_max = 786.7
#energy_min = 21.23
#energy_max = 101.02

### for VCTK ###
#spk_embed_dim = 32
#n_spkers = 108
#spk_embed_weight_std = 0.01
#f0_min = 70.0
#f0_max = 793.0
#energy_min = 0.22
#energy_max = 1048.7

## for LibriTTS ###
spk_embed_dim = 256
n_spkers = 904
spk_embed_weight_std = 0.01
f0_min = 70.0
f0_max = 800.0
energy_min = 0.0
energy_max = 570.0

n_bins = 256


# Checkpoints and synthesis path
preprocessed_path = os.path.join("./preprocessed/", dataset)
checkpoint_path = os.path.join("./ckpt/", dataset)
synth_path = os.path.join("./synth/", dataset)
eval_path = os.path.join("./eval/", dataset)
log_path = os.path.join("./log/", dataset)
test_path = os.path.join("./results/")


# Optimizer
batch_size = 16
epochs = 1000
n_warm_up_step = 4000
grad_clip_thresh = 1.0
acc_steps = 1

betas = (0.9, 0.98)
eps = 1e-9
weight_decay = 0.


# Vocoder
vocoder = 'melgan' # 'waveglow' or 'melgan'


# Log-scaled duration
log_offset = 1.


# Save, log and synthesis
save_step = 20000
synth_step = 2000
eval_step = 100000000
eval_size = 256
log_step = 1000
clear_Time = 20
