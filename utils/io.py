from pathlib import Path

import matplotlib
import soundfile
import torch
import torchaudio
from config import hparams as hp
from matplotlib import pyplot as plt

# make use to use "Agg", otherwised memory leakage will occur!!
matplotlib.use("Agg")

# === dirs === #
def make_paths(paths: list):
    for path in paths:
        path.mkdir(parents=True, exist_ok=True)


def add_comment(file_path: Path, index: str, text: str):
    with open(file_path, "a") as f:
        f.write(f"{index}: {text}\n")


# === files === #
def save_audios(
    wav_batch: torch.Tensor, wav_lens: torch.Tensor, data_ids: list, save_dir: Path
) -> int:
    saved_num = 0
    wav_batch = wav_batch.cpu()
    wavs = [w[:l] for w, l in zip(wav_batch, wav_lens)]
    for wav, file_name in zip(wavs, data_ids):
        save_path = save_dir / (file_name + ".wav")
        torchaudio.save(save_path, wav.unsqueeze(0), hp.sampling_rate)
        # soundfile.write(save_path, wav.numpy(), hp.sampling_rate)
        saved_num += 1
    return saved_num


# === plot === #
def plot_mel(mel_gt, mel_pred, data_ids: list, save_dir: Path):
    """
    data : [batch of gt, batch of pred]
    
    len(data) == 2
    
    """
    assert len(mel_gt) == len(mel_pred)
    for i, (mel1, mel2) in enumerate(
        zip(
            mel_gt.transpose(1, 2).cpu().numpy(), mel_pred.transpose(1, 2).cpu().numpy()
        )
    ):
        fig, axes = plt.subplots(2, 1, squeeze=False)
        save_path = save_dir / (data_ids[i] + ".png")
        fig.suptitle(data_ids[i])
        axes[0][0].imshow(mel1, origin="lower")
        axes[1][0].imshow(mel2, origin="lower")

        fig.savefig(save_path)
        fig.clear()
        plt.close(fig)
        plt.close("all")


# === stdio === #
def simple_table(item_tuples):

    border_pattern = "+---------------------------------------"
    whitespace = "                                            "

    headings, cells, = [], []

    for item in item_tuples:

        heading, cell = str(item[0]), str(item[1])

        pad_head = True if len(heading) < len(cell) else False

        pad = abs(len(heading) - len(cell))
        pad = whitespace[:pad]

        pad_left = pad[: len(pad) // 2]
        pad_right = pad[len(pad) // 2 :]

        if pad_head:
            heading = pad_left + heading + pad_right
        else:
            cell = pad_left + cell + pad_right

        headings += [heading]
        cells += [cell]

    border, head, body = "", "", ""

    for i in range(len(item_tuples)):

        temp_head = f"| {headings[i]} "
        temp_body = f"| {cells[i]} "

        border += border_pattern[: len(temp_head)]
        head += temp_head
        body += temp_body

        if i == len(item_tuples) - 1:
            head += "|"
            body += "|"
            border += "+"

    print(border)
    print(head)
    print(border)
    print(body)
    print(border)
    print(" ")
