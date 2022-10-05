from multiprocessing import Pool

from .preprocess_func import *


class ImapWrapper(object):
    """
    Function object wrapper.
    """
    def __init__(self, func) -> None:
        self.f = func
    
    def __call__(self, task) -> bool:
        *args, ignore_errors = task
        try:
            self.f(*args)
        except:
            if ignore_errors:
                return False
            raise
        return True


def imap_textgrid2segment_and_phoneme(task):
    *args, ignore_errors = task
    try:
        textgrid2segment_and_phoneme(*args)
    except:
        if ignore_errors:
            return False
        raise
    return True


def textgrid2segment_and_phoneme_mp(
    dataset, queries, 
    textgrid_featname: str,
    segment_featname: str,
    phoneme_featname: str,
    n_workers: int=os.cpu_count()-2, chunksize: int=64,
    ignore_errors: bool=False
) -> None:
    print("[textgrid2segment_and_phoneme_mp]:")
    n = len(queries)
    tasks = list(zip(
        [dataset] * n, queries,
        [textgrid_featname] * n,
        [segment_featname] * n,
        [phoneme_featname] * n,
        [ignore_errors] * n
    ))
    
    fail_cnt = 0
    with Pool(processes=n_workers) as pool:
        # for res in tqdm(pool.imap(imap_textgrid2segment_and_phoneme, tasks, chunksize=chunksize), total=n):
        for res in tqdm(pool.imap(ImapWrapper(textgrid2segment_and_phoneme), tasks, chunksize=chunksize), total=n):
            fail_cnt += 1 - res
    print("[textgrid2segment_and_phoneme_mp]: Skipped: ", fail_cnt)


def imap_trim_wav_by_mfa_segment(task):
    *args, ignore_errors = task
    try:
        trim_wav_by_segment(*args)
    except:
        if ignore_errors:
            return False
        raise
    return True


def trim_wav_by_segment_mp(
    dataset: BaseDataParser, queries, sr: int,
    wav_featname: str,
    segment_featname: str,
    wav_trim_featname: str,
    refresh: bool=False,
    n_workers: int=1, chunksize: int=256,
    ignore_errors: bool=False
) -> None:
    print("[trim_wav_by_segment_mp]:")
    n = len(queries)
    tasks = list(zip(
        [dataset] * n, queries, [sr] * n,
        [wav_featname] * n,
        [segment_featname] * n,
        [wav_trim_featname] * n,
        [ignore_errors] * n
    ))

    fail_cnt = 0
    if n_workers == 1:
        segment_feat = dataset.get_feature(segment_featname)
        segment_feat.read_all(refresh=refresh)
        for i in tqdm(range(n)):
            res = trim_wav_by_segment(
                dataset, queries[i], sr,
                wav_featname,
                segment_featname,
                wav_trim_featname,
                ignore_errors
            )
            fail_cnt += 1 - res
    else:
        with Pool(processes=n_workers) as pool:
            for res in tqdm(pool.imap(imap_trim_wav_by_mfa_segment, tasks, chunksize=chunksize), total=n):
                fail_cnt += 1 - res
    print("[trim_wav_by_segment_mp]: Skipped: ", fail_cnt)


def imap_wav_to_mel_energy_pitch(task):
    *args, ignore_errors = task
    try:
        wav_to_mel_energy_pitch(*args)
    except:
        if ignore_errors:
            return False
        raise
    return True


def wav_to_mel_energy_pitch_mp(
    dataset: BaseDataParser, queries,
    wav_featname: str,
    mel_featname: str,
    energy_featname: str,
    pitch_featname: str,
    interp_pitch_featname: str,
    n_workers: int=4, chunksize: int=32,
    ignore_errors: bool=False
) -> bool:
    print("[wav_to_mel_energy_pitch_mp]:")
    n = len(queries)
    tasks = list(zip(
        [dataset] * n, queries,
        [wav_featname] * n,
        [mel_featname] * n,
        [energy_featname] * n,
        [pitch_featname] * n,
        [interp_pitch_featname] * n,
        [ignore_errors] * n
    ))

    fail_cnt =0
    if n_workers == 1:
        for i in tqdm(range(n)):
            res = wav_to_mel_energy_pitch(
                dataset, queries[i],
                wav_featname,
                mel_featname,
                energy_featname,
                pitch_featname,
                interp_pitch_featname,
                ignore_errors
            )
            fail_cnt += 1 - res
    else:
        with Pool(processes=n_workers) as pool:
            for i in tqdm(pool.imap(imap_wav_to_mel_energy_pitch, tasks, chunksize=chunksize), total=n):
                fail_cnt += 1 - res
    print("[wav_to_mel_energy_pitch_mp]: Skipped: ", fail_cnt)


def imap_segment2duration(task):
    *args, ignore_errors = task
    try:
        segment2duration(*args)
    except:
        if ignore_errors:
            return False
        raise
    return True


def segment2duration_mp(
    dataset: BaseDataParser, queries, inv_frame_period: float,
    segment_featname: str, 
    duration_featname: str,
    refresh: bool=False,
    n_workers=1, chunksize=256,
    ignore_errors: bool=False
) -> None:
    print("[segment2duration_mp]:")
    n = len(queries)
    tasks = list(zip(
        [dataset] * n, queries, [inv_frame_period] * n,
        [segment_featname] * n,
        [duration_featname] * n,
        [ignore_errors] * n
    ))

    fail_cnt = 0
    if n_workers == 1:
        segment_feat = dataset.get_feature(segment_featname)
        segment_feat.read_all(refresh=refresh)
        for i in tqdm(range(n)):
            res = segment2duration(
                dataset, queries[i], inv_frame_period,
                segment_featname,
                duration_featname,
                ignore_errors
            )
            fail_cnt += 1- res
    else:
        with Pool(processes=n_workers) as pool:
            for res in tqdm(pool.imap(imap_segment2duration, tasks, chunksize=chunksize), total=n):
                fail_cnt += 1 - res
    print("[segment2duration_mp]: Skipped: ", fail_cnt)


def imap_duration_avg_pitch_and_energy(task):
    *args, ignore_errors = task
    try:
        duration_avg_pitch_and_energy(*args)
    except:
        if ignore_errors:
            return False
        raise
    return True


def duration_avg_pitch_and_energy_mp(
    dataset: BaseDataParser, queries,
    duration_featname: str,
    pitch_featname: str,
    energy_featname: str,
    refresh: bool=False,
    n_workers: int=1, chunksize: int=256,
    ignore_errors: bool=False
) -> None:
    print("[duration_avg_pitch_and_energy_mp]:")
    n = len(queries)
    tasks = list(zip(
        [dataset] * n, queries,
        [duration_featname] * n,
        [pitch_featname] * n,
        [energy_featname] * n,
        [ignore_errors] * n,
    ))

    fail_cnt = 0
    if n_workers == 1:
        duration_feat = dataset.get_feature(duration_featname)
        pitch_feat = dataset.get_feature(pitch_featname)
        energy_feat = dataset.get_feature(energy_featname)
        duration_feat.read_all(refresh=refresh)
        pitch_feat.read_all(refresh=refresh)
        energy_feat.read_all(refresh=refresh)
        for i in tqdm(range(n)):
            res = duration_avg_pitch_and_energy(
                dataset, queries[i],
                duration_featname,
                pitch_featname,
                energy_featname,
                ignore_errors
            )
            fail_cnt += 1 - res
    else:
        with Pool(processes=n_workers) as pool:
            for i in tqdm(pool.imap(imap_duration_avg_pitch_and_energy, tasks, chunksize=chunksize), total=n):
                fail_cnt += 1- res
        print("[duration_avg_pitch_and_energy_mp]: Skipped: ", fail_cnt)
