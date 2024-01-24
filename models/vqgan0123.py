import torch
import torch.nn as nn

from .vqvae0123 import VQVAE
from .discriminator import Discriminator

# https://github.com/dome272/MaskGIT-pytorch

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)

class VQGAN:
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
        # super(VQGAN, self).__init__()
        self.vqvae = VQVAE(vae_in_channels,
                           vae_out_channels,
                           vae_embedding_dim,
                           vae_num_embeddings,
                           vae_hidden_dims,
                           vae_beta,
                           img_size)
        self.discriminator = Discriminator(input_channel, ndf, n_layers)
        self.discriminator.apply(weights_init)

    # def forward(self, x):
    #     vq_loss, quantized, x_recon = self.vqvae(x)
    #     disc = self.discriminator(x_recon)
    #     return vq_loss, quantized, x_recon, disc
        
    @staticmethod
    def adopt_weight(disc_factor, i, threshold, value=0.):
        if i < threshold:
            disc_factor = value
        return disc_factor