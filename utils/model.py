import numpy as np
import torch


def get_param_num(model):
    num_param = sum(param.numel() for param in model.parameters())
    return num_param


def data_to_device(batch, device):
    spker_ids = torch.from_numpy(batch["spker_id"]).to(device)
    text_seq = torch.from_numpy(batch["text_seq"]).long().to(device)
    text_len = torch.from_numpy(batch["text_len"]).long().to(device)
    d_gt = torch.from_numpy(batch["d"]).long().to(device)
    log_d_gt = torch.from_numpy(batch["log_d"]).float().to(device)
    p_gt = torch.from_numpy(batch["f0"]).float().to(device)
    e_gt = torch.from_numpy(batch["energy"]).float().to(device)
    mel_gt = torch.from_numpy(batch["mel"]).float().to(device)
    mel_len = torch.from_numpy(batch["mel_len"]).long().to(device)
    max_text_len = np.max(batch["text_len"]).astype(np.int32)
    max_mel_len = np.max(batch["mel_len"]).astype(np.int32)

    model_batch = (
        spker_ids,
        text_seq,
        text_len,
        d_gt,
        p_gt,
        e_gt,
        mel_len,
        max_text_len,
        max_mel_len,
    )

    gt_batch = (log_d_gt, p_gt, e_gt, mel_gt)
    return (model_batch, gt_batch)
