import numpy as np
from tgt.core import IntervalTier


def get_textgird_contents(tier : IntervalTier) -> list:
    intervals  = tier.intervals
    contents = [i.text for i in intervals]
    return contents


def duration_warp(real_d, int_d):
    total_diff = sum(real_d) - sum(int_d)
    drop_diffs = np.array(real_d) - np.array(int_d)
    drop_order = np.argsort(-drop_diffs)
    for i in range(int(total_diff)):
        index = drop_order[i]
        int_d[index] += 1

    return int_d


def get_alignment(tier, sample_rate, hop_length):
    sil_phones = ["sil", "sp", "spn"]
    phones = []
    durations_real = []
    durations_int = []
    start_time = 0
    end_time = 0
    end_idx = 0
    for t in tier._objects:
        s, e, p = t.start_time, t.end_time, t.text

        # Trimming leading silences
        if phones == []:
            if p in sil_phones:
                continue
            else:
                start_time = s
        if p not in sil_phones:
            phones.append(p)
            end_time = e
            end_idx = len(phones)
        else:
            phones.append(p)

        d = e * sample_rate / hop_length - s * sample_rate / hop_length
        durations_real.append(d)
        durations_int.append(int(d))

    # Trimming tailing silences
    durations_real = durations_real[:end_idx]
    durations_int = durations_int[:end_idx]
    phones = phones[:end_idx]
    durations = duration_warp(durations_real, durations_int)

    return phones, durations, start_time, end_time
