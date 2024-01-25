import torch
import torch.nn as nn

from .vqvae import VQVAE
from .discriminator import Discriminator

class VQGAN(nn.Module):
    def __init__(self, 
                 vae_in_channels: int,
                 vae_out_channels: int,
                 vae_embedding_dim: int,
                 vae_num_embeddings: int,
                 vae_hidden_dims: list = None,
                 vae_beta: float = 0.25,
                 img_size: int = 32,
                 input_channel=3,
                 ndf=64, 
                 n_layers=3):
        super(VQGAN, self).__init__()
        self.vqvae = VQVAE(vae_in_channels,
                           vae_out_channels,
                           vae_embedding_dim,
                           vae_num_embeddings,
                           vae_hidden_dims,
                           vae_beta,
                           img_size)
        self.discriminator = Discriminator(input_channel, ndf, n_layers)

    def forward(self, x):
        vq_loss, quantized, x_recon = self.vqvae(x)
        disc = self.discriminator(x_recon)
        return vq_loss, quantized, x_recon, disc