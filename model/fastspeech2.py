import torch
import torch.nn as nn
from config import hparams as hp
from transformer.Layers import PostNet
from transformer.Models import Decoder, Encoder
from utils.mask import get_mask_from_lengths

from model.modules import Embedding, SpeakerIntegrator, VarianceAdaptor

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class FastSpeech2(nn.Module):
    """ FastSpeech2 """

    def __init__(self, n_spkers=1, spker_embed_dim=256, spker_embed_std=0.01):
        super(FastSpeech2, self).__init__()
        self.n_spkers = n_spkers
        self.spker_embed_dim = spker_embed_dim
        self.spker_embed_std = spker_embed_std

        ### Encoder, Speaker Integrator, Variance Adaptor, Deocder, Postnet ###
        self.spker_embeds = Embedding(
            n_spkers, spker_embed_dim, padding_idx=None, std=spker_embed_std
        )
        self.encoder = Encoder()
        self.speaker_integrator = SpeakerIntegrator()
        self.variance_adaptor = VarianceAdaptor()
        self.decoder = Decoder()
        self.to_mel = nn.Linear(hp.decoder_hidden, hp.n_mels)
        self.postnet = PostNet()

    def forward(
        self,
        spker_ids,
        text_seq,
        text_len,
        d_gt=None,
        p_gt=None,
        e_gt=None,
        mel_len=None,
        max_text_len=None,
        max_mel_len=None,
    ):
        text_mask = get_mask_from_lengths(text_len, max_text_len)
        mel_mask = (
            get_mask_from_lengths(mel_len, max_mel_len) if mel_len is not None else None
        )

        # === Encoder === #
        spker_embed = self.spker_embeds(spker_ids)

        encoder_output = self.encoder(text_seq, text_mask)
        encoder_output = self.speaker_integrator(encoder_output, spker_embed)

        # === Variance Adaptor === #
        if d_gt is not None:
            (
                variance_adaptor_output,
                d_pred,
                p_pred,
                e_pred,
                _,
                _,
            ) = self.variance_adaptor(
                encoder_output, text_mask, mel_mask, d_gt, p_gt, e_gt, max_mel_len,
            )
        else:
            (
                variance_adaptor_output,
                d_pred,
                p_pred,
                e_pred,
                mel_len,
                mel_mask,
            ) = self.variance_adaptor(
                encoder_output, text_mask, mel_mask, d_gt, p_gt, e_gt, max_mel_len,
            )

        variance_adaptor_output = self.speaker_integrator(
            variance_adaptor_output, spker_embed
        )

        # === Decoder === #
        decoder_output = self.decoder(variance_adaptor_output, mel_mask)
        mel = self.to_mel(decoder_output)
        mel_postnet = self.postnet(mel) + mel

        # === Masking === #
        if mel_mask is not None:
            mel = mel.masked_fill(mel_mask.unsqueeze(-1), 0)
            mel_postnet = mel_postnet.masked_fill(mel_mask.unsqueeze(-1), 0)

        # === Output === #
        pred = (mel, mel_postnet, d_pred, p_pred, e_pred)
        return (pred, text_mask, mel_mask, mel_len)


if __name__ == "__main__":
    """
    write some tests here
    """
    model = FastSpeech2()
    print(model)
    print(sum(param.numel() for param in model.parameters()))
