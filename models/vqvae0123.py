import torch
import torch.nn as nn

from .vq import VectorQuantizer
from .res import ResidualLayer



# ImageNet Image is 128 * 128 * 3

class VQVAE(nn.Module):
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 embedding_dim: int,
                 num_embeddings: int,
                 hidden_dims: list = None,
                 beta: float = 0.25,
                 img_size: int = 32,
                 **kwargs) -> None:
        super().__init__()


        self.embedding_dim = embedding_dim
        self.num_embeddings = num_embeddings
        self.img_size = img_size
        self.beta = beta

        modules = []

        if hidden_dims is None:
            hidden_dims = [128, 256]

        # Build Encoder
        for h_dim in hidden_dims:
            modules.append(
                nn.Sequential(
                    nn.Conv2d(in_channels, out_channels=h_dim,
                              kernel_size=4, stride=2, padding=1),
                    nn.LeakyReLU()
                )
            )
            in_channels = h_dim

        modules.append(
            nn.Sequential(
                nn.Conv2d(in_channels, embedding_dim,
                          kernel_size=3, stride=1, padding=1),
                nn.LeakyReLU()
            )
        )


        for _ in range(8):
            modules.append(ResidualLayer(embedding_dim, embedding_dim))

        
        self.encoder = nn.Sequential(*modules)


        self.vq_layer = VectorQuantizer(num_embeddings, embedding_dim, beta)

        # Build Decoder

        modules = []

        modules.append(
            nn.Conv2d(embedding_dim,
                        hidden_dims[-1],
                        kernel_size=3,
                        stride=1,
                        padding=1)
        )

        for _ in range(8):
            modules.append(ResidualLayer(hidden_dims[-1], hidden_dims[-1]))


        hidden_dims.reverse()

        for i in range(len(hidden_dims) - 1):
            modules.append(
                nn.ConvTranspose2d(hidden_dims[i],
                                    hidden_dims[i + 1],
                                    kernel_size=4,
                                    stride=2,
                                    padding=1)
            )

        modules.append(nn.LeakyReLU())


        modules.append(
            nn.ConvTranspose2d(hidden_dims[-1],
                                out_channels=out_channels,
                                kernel_size=4,
                                stride=2,
                                padding=1)
        )

        self.decoder = nn.Sequential(*modules)

        



    def vqencoder(self, x):
       
        z = self.encoder(x)
        z, vq_loss, label = self.vq_layer(z)
        return z, vq_loss, label
    
    def vqdecoder(self, q_x):
        x_recon = self.decoder(q_x)
        return x_recon
    
    def forward(self, x):
        z, vq_loss, label = self.vqencoder(x)
        x_recon = self.vqdecoder(z)
        return x_recon, vq_loss, label

    def calculate_lambda(self, nll_loss, g_loss):
        last_layer = self.decoder[-1]
        last_layer_weight = last_layer.weight
        nll_grads = torch.autograd.grad(nll_loss, last_layer_weight, retain_graph=True)[0]
        g_grads = torch.autograd.grad(g_loss, last_layer_weight, retain_graph=True)[0]

        lamda = torch.norm(nll_grads) / (torch.norm(g_grads) + 1e-6)
        lamda = torch.clamp(lamda, 0, 1e4).detach()
        return 0.8 * lamda