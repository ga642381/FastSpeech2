import argparse
import os
from datetime import datetime
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

import utils
from dlhlp_lib.vocoders import MelGAN, HifiGAN
from config import hparams as hp
from torch.utils.data import ConcatDataset
from data.dataset import Dataset
from model import FastSpeech2, FastSpeech2Loss, ScheduledOptim


if hp.CUDA_LAUNCH_BLOCKING:
    os.environ['CUDA_LAUNCH_BLOCKING'] = '1'


# gt : ground truth
# pred : prediction
class Trainer:
    def __init__(self, paths, restore_step: int = 0):
        # === Init === #
        self.paths = paths
        self.restore_step = restore_step
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        (
            self.train_loader,
            self.valid_loader,
            self.n_spkers,
            self.speaker_table,
            self.inv_speaker_table,
        ) = self.__init_dataset()
        self.train_logger, self.eval_logger = self.__init_logger()
        self.vocoder = HifiGAN(dirpath="../model-lib/hifigan/TAT")
        self.vocoder.cuda()

        # === nn === #
        self.model = self.__init_model(n_spkers=self.n_spkers).to(self.device)
        self.loss = FastSpeech2Loss().to(self.device)
        self.optimizer, self.scheduled_optim = self.__init_optimizer(restore_step)
        if restore_step > 0:
            self.__load_model(self.paths["checkpoint_path"], restore_step)
            self.train_step = restore_step
            print(f"[Training] Model Restored at Step {self.train_step}")
        else:
            self.train_step = 0
            print("[Training] Start New Training")

    def train(self):
        self.model.train()
        while self.train_step < hp.total_steps:
            for batches in self.train_loader:
                for batch in batches:
                    print(f"[Training] step: {self.train_step}", end="\r", flush=True)

                    # === 1. Load data === #
                    model_batch, gt_batch = utils.data_to_device(batch, self.device)

                    # === 2. Forward === #
                    (model_pred, text_mask, mel_mask, mel_lens) = self.model(
                        *model_batch
                    )

                    # === 3. Cal Loss ===#
                    mel_loss, mel_postnet_loss, d_loss, f_loss, e_loss = self.loss(
                        *model_pred, *gt_batch, ~text_mask, ~mel_mask,
                    )
                    total_loss = mel_loss + mel_postnet_loss + d_loss + f_loss + e_loss

                    # === 4. Backward === #
                    total_loss = total_loss / hp.acc_steps
                    total_loss.backward()
                    if self.train_step % hp.acc_steps == 0:
                        nn.utils.clip_grad_norm_(
                            self.model.parameters(), hp.grad_clip_thresh
                        )
                        self.scheduled_optim.step_and_update_lr()
                        self.scheduled_optim.zero_grad()

                    # === 5. Logging === #
                    if self.train_step % hp.log_step == 0:
                        self.__log(
                            "train",
                            total_loss,
                            mel_loss,
                            mel_postnet_loss,
                            d_loss,
                            f_loss,
                            e_loss,
                        )
                    # === 6. Eval and Synth === #
                    if self.train_step % hp.eval_step == 0:
                        self.__eval_and_synth()

                    # # === 7. Save === #
                    if self.train_step % hp.save_step == 0:
                        self.__save_model_and_optimizer()

                    self.train_step += 1

    def __init_dataset(self):
        # Multiple datasets
        all_spkers = set()
        train_datasets, valid_datasets = [], []
        for d in self.paths["data_dir"]:
            train_dataset = Dataset(data_dir=d, split="train")
            valid_dataset = Dataset(data_dir=d, split="valid")
            all_spkers.update(train_dataset.spkers)
            all_spkers.update(valid_dataset.spkers)
            train_datasets.append(train_dataset)
            valid_datasets.append(valid_dataset)

        all_spkers = list(all_spkers)
        all_spkers.sort()
        train_dataset, valid_dataset = ConcatDataset(train_datasets), ConcatDataset(valid_datasets)
        for dset in train_dataset.datasets:
            dset.set_speakers(all_spkers)
        for dset in valid_dataset.datasets:
            dset.set_speakers(all_spkers)
        
        train_loader = DataLoader(
            train_dataset,
            batch_size=hp.batch_size ** 2,
            shuffle=True,
            collate_fn=train_datasets[0].collate_fn,
            drop_last=True,
            num_workers=4,
        )
        valid_loader = DataLoader(
            valid_dataset,
            batch_size=hp.batch_size ** 2,
            shuffle=False,
            collate_fn=train_datasets[0].collate_fn,
            drop_last=False,
            num_workers=4,
        )
        n_spkers = len(all_spkers)
        spker_table = train_datasets[0].spker_table  # spker table is shared for all datasets
        inv_spker_table = {v: k for k, v in spker_table.items()}
        return train_loader, valid_loader, n_spkers, spker_table, inv_spker_table

    def __init_model(self, n_spkers):
        model = nn.DataParallel(FastSpeech2(n_spkers=n_spkers))
        num_param = utils.get_param_num(model)
        print(f"[INFO] Number of FastSpeech2 Parameters: {num_param:,}")

        return model

    def __init_optimizer(self, restore_step):
        optimizer = torch.optim.Adam(
            self.model.parameters(),
            betas=hp.betas,
            eps=hp.eps,
            weight_decay=hp.weight_decay,
        )
        scheduled_optim = ScheduledOptim(
            optimizer, hp.decoder_hidden, hp.n_warm_up_step, restore_step
        )
        return optimizer, scheduled_optim

    def __init_logger(self):
        train_log_path = self.paths["log_path"] / "train"
        eval_log_path = self.paths["log_path"] / "eval"
        utils.make_paths([train_log_path, eval_log_path])

        train_logger = SummaryWriter(train_log_path)
        eval_logger = SummaryWriter(eval_log_path)
        return train_logger, eval_logger

    def __log(
        self, mode, total_loss, mel_loss, mel_postnet_loss, d_loss, f_loss, e_loss
    ):
        total_loss = total_loss.item()
        mel_loss = mel_loss.item()
        mel_postnet_loss = mel_postnet_loss.item()
        d_loss = d_loss.item()
        f_loss = f_loss.item()
        e_loss = e_loss.item()

        utils.simple_table(
            [
                ("Mode", mode),
                ("Time", datetime.today().strftime("%Y-%m-%d-%H:%M")),
                ("Step", f"{self.train_step}"),
                ("Total Loss", f"{total_loss:.4f}"),
                ("Mel Loss", f"{mel_loss:.4f}"),
                ("Mel PostNet Loss", f"{mel_postnet_loss:.4f}"),
                ("Duration Loss", f"{d_loss:.4f}"),
                ("F0 Loss", f"{f_loss:.4f}"),
                ("Energy Loss", f"{e_loss:.4f}"),
            ]
        )

        if mode == "train":
            logger = self.train_logger
        elif mode == "eval":
            logger = self.eval_logger

        logger.add_scalar("Loss/total_loss", total_loss, self.train_step)
        logger.add_scalar("Loss/mel_loss", mel_loss, self.train_step)
        logger.add_scalar("Loss/mel_postnet_loss", mel_postnet_loss, self.train_step)
        logger.add_scalar("Loss/d_loss", d_loss, self.train_step)
        logger.add_scalar("Loss/f_loss", f_loss, self.train_step)
        logger.add_scalar("Loss/e_loss", e_loss, self.train_step)

    def __synth(
        self, mel: torch.Tensor, mel_lens: torch.Tensor, data_ids, save_dir: Path
    ):
        # mel.shape : (batch, time, mel_dim)
        save_dir.mkdir(parents=True, exist_ok=True)
        wav_lens = [m * hp.hop_length for m in mel_lens]
        wavs = self.vocoder.infer(mel.transpose(1, 2), wav_lens)
        utils.save_audios(wavs, wav_lens, data_ids, save_dir)

    def __eval_and_synth(self):
        self.model.eval()
        print(f"[Evaluation] Evaluating at step {str(self.train_step)}")
        batch_num = 0
        gt_synth_path = self.paths["synth_path"] / "gt"
        pred_synth_path = self.paths["synth_path"] / str(self.train_step)

        synth_gt = True if not gt_synth_path.exists() else False
        # total_loss, mel_loss, mel_postnet_loss, d_loss, f_loss, e_loss
        L = (0, 0, 0, 0, 0, 0)
        for i, batches in tqdm(enumerate(self.valid_loader), total=len(self.valid_loader)):
            for batch in batches:
                model_batch, gt_batch = utils.data_to_device(batch, self.device)
                with torch.no_grad():
                    # forward
                    (model_pred, text_mask, mel_mask, mel_lens) = self.model(
                        *model_batch
                    )
                    # cal loss
                    mel_loss, mel_postnet_loss, d_loss, f_loss, e_loss = self.loss(
                        *model_pred, *gt_batch, ~text_mask, ~mel_mask,
                    )

                    total_loss = mel_loss + mel_postnet_loss + d_loss + f_loss + e_loss

                    l = (total_loss, mel_loss, mel_postnet_loss, d_loss, f_loss, e_loss)

                    # accumulate loss
                    L = tuple(a + b for a, b in zip(L, l))

                if i < 1:  # only synth first batch
                    # synthesize ground truth mel spectrogram
                    if synth_gt:
                        self.__synth(
                            mel=gt_batch[-1],
                            mel_lens=mel_lens,
                            data_ids=batch["data_id"],
                            save_dir=gt_synth_path,
                        )

                    # synthesize predicted mel spectrogram
                    self.__synth(
                        mel=model_pred[1],
                        mel_lens=mel_lens,
                        data_ids=batch["data_id"],
                        save_dir=pred_synth_path,
                    )

                    # plot ground truth and predicted mel spectrogram
                    utils.plot_mel(
                        mel_gt=gt_batch[-1],
                        mel_pred=model_pred[1],
                        data_ids=batch["data_id"],
                        save_dir=pred_synth_path,
                    )

                batch_num += 1
        # log
        L = (l / batch_num for l in L)
        self.__log("eval", *L)
        self.model.train()

        # synth info
        if synth_gt:
            print(
                f"[Evaluation] {len(os.listdir(gt_synth_path))} files were saved in {gt_synth_path}"
            )
        print(
            f"[Evaluation] {len(os.listdir(pred_synth_path))} files were saved in {pred_synth_path}"
        )

    def __load_model(self, checkpoint_path, restore_step):
        checkpoint = torch.load(checkpoint_path / f"checkpoint_{restore_step}.pth.tar")
        self.model.load_state_dict(checkpoint["model"])
        self.optimizer.load_state_dict(checkpoint["optimizer"])
        print("\n---Model Restored at Step {}---\n".format(restore_step))

    def __save_model_and_optimizer(self):
        save_path = (
            self.paths["checkpoint_path"] / f"checkpoint_{self.train_step}.pth.tar"
        )
        if not save_path.exists():
            torch.save(
                {
                    "model": self.model.state_dict(),
                    "optimizer": self.optimizer.state_dict(),
                },
                save_path,
            )
            print(f"[Training] Model saved at step {self.train_step}")


def main(args):
    # date_time = datetime.today().strftime("%Y-%m-%d-%H:%M")
    record_idx = f"{args.exp_name}"
    paths = {}
    paths["data_dir"] = args.data_dir
    record_root = Path(args.record_dir).resolve()
    paths["checkpoint_path"] = Path(record_root / record_idx / "ckpt").resolve()
    paths["synth_path"] = Path(record_root / record_idx / "synth").resolve()
    paths["log_path"] = Path(record_root / record_idx / "log").resolve()
    record_file_path = Path(record_root / "comments.txt").resolve()

    utils.make_paths(list(paths.values())[1:])
    utils.add_comment(record_file_path, record_idx, args.comment)

    # === training === #
    torch.manual_seed(0)
    trainer = Trainer(paths, restore_step=args.restore_step)
    trainer.train()


if __name__ == "__main__":
    """
    e.g.
    # LJSpeech #
        * python train.py ./processed/LJSpeech --comment "Hello LJSpeech" 
    
    # LibriTTS #
        * python train.py ./processed/LibriTTS --comment "Hello LibriTTS" 
    
    # VCTK #
        * python train.py ./processed/VCTK --comment "Hello VCTK" 
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, nargs='+')
    parser.add_argument("--exp_name", type=str)
    parser.add_argument("--record_dir", type=str, default="./records")
    parser.add_argument("--comment", type=str, default="None")
    parser.add_argument("--restore_step", type=int, default=0)
    args = parser.parse_args()
    main(args)
