import os
import yaml


### Dataset(LJSpeech, Blizzard2013, VCTK, LibriTTS) ###
#dataset = "LJSpeech"
#data_path = "/dataset/LJSpeech-1.1"

#dataset = "Blizzard2013"
#data_path = "./Blizzard-2013/train/segmented/"

#dataset = "VCTK"
#data_path = "/home/kaiwei/SpeechNet/FastSpeech2/dataset/VCTK-Corpus"

dataset = "LibriTTS"
data_path = "./dataset/LibriTTS/train-clean-360"
mfa_path = "./MFA"

### Text ###
# g2p_en
text_cleaners = ['english_cleaners']


### FastSpeech 2 ###
encoder_layer = 4
encoder_head = 2
encoder_hidden = 256

decoder_layer = 4
decoder_head = 2
decoder_hidden = 256

fft_conv1d_filter_size = 1024
fft_conv1d_kernel_size = (9, 1)

encoder_dropout = 0.2
decoder_dropout = 0.2

variance_predictor_filter_size = 256
variance_predictor_kernel_size = 3
variance_predictor_dropout = 0.5

max_seq_len = 1000

### dataset ###
with open('./data/dataset.yaml', 'r') as f:
    dataset_config = yaml.load(f, Loader=yaml.FullLoader)

max_wav_value = 32768.0
n_mel_channels = 80
mel_fmin = 0.0
mel_fmax = None

# Audio and mel
sampling_rate = dataset_config[dataset]['sampling_rate']
filter_length = dataset_config[dataset]['filter_length']
hop_length = dataset_config[dataset]['hop_length']
win_length = dataset_config[dataset]['win_length']


# Quantization for F0 and energy
f0_min = dataset_config[dataset]['f0_min']
f0_max = dataset_config[dataset]['f0_max']
energy_min = dataset_config[dataset]['energy_min']
energy_max = dataset_config[dataset]['energy_max']
n_bins = 256

# Speaker embedding
use_spk_embed = dataset_config[dataset]['use_spk_embed']
spk_embed_dim = 256
spk_embed_weight_std = 0.01


### Checkpoints and synthesis path ###
preprocessed_path = os.path.join("./preprocessed/", dataset)
checkpoint_path   = os.path.join("./log/ckpt/", dataset)
synth_path        = os.path.join("./log/synth/", dataset)
eval_path         = os.path.join("./log/eval/", dataset)
log_path          = os.path.join("./log/log", dataset)
test_path         = os.path.join("./log/results/")


### Optimizer ###
batch_size = 16
epochs = 1000
n_warm_up_step = 4000
grad_clip_thresh = 1.0
acc_steps = 1

betas = (0.9, 0.98)
eps = 1e-9
weight_decay = 0.

# Vocoder
vocoder = 'melgan'

# Log-scaled duration
log_offset = 1.

# Save, log and synthesis
save_step = 20000
synth_step = 2000
eval_step = 2000
eval_size = 256
log_step = 1000
clear_Time = 20
