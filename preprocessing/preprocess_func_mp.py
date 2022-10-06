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
                print(*args[2])
                return False
            raise
        return True


def resample_mp(
    dataset: BaseDataParser, queries,
    input_featname: str,
    output_featname: str,
    sr: int,
    n_workers: int=4, chunksize: int=64,
    ignore_errors: bool=False
) -> None:
    print("[resample_mp]:")
    n = len(queries)
    tasks = list(zip(
        [dataset] * n, queries,
        [input_featname] * n,
        [output_featname] * n,
        [sr] * n,
        [ignore_errors] * n
    ))
    
    fail_cnt = 0
    with Pool(processes=n_workers) as pool:
        for res in tqdm(pool.imap(ImapWrapper(resample), tasks, chunksize=chunksize), total=n):
            fail_cnt += 1 - res
    print("[process_utterance_mp]: Skipped: ", fail_cnt)


def process_utterance_mp(
    dataset: BaseDataParser, queries,
    wav_featname: str,
    n_workers: int=4, chunksize: int=64,
    ignore_errors: bool=False
) -> None:
    print("[process_utterance_mp]:")
    n = len(queries)
    tasks = list(zip(
        [dataset] * n, queries,
        [wav_featname] * n,
        [ignore_errors] * n
    ))
    
    fail_cnt = 0
    with Pool(processes=n_workers) as pool:
        for res in tqdm(pool.imap(ImapWrapper(process_utterance), tasks, chunksize=chunksize), total=n):
            fail_cnt += 1 - res
    print("[process_utterance_mp]: Skipped: ", fail_cnt)
