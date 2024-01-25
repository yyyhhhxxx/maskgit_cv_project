import torch
import torch.nn as nn

class Discriminator(nn.Module):
    def __init__(self, input_channel=3, ndf=64, n_layers=3):


        super(Discriminator, self).__init__()

        modules = [nn.Conv2d(input_channel, ndf, kernel_size=4, stride=2, padding=1), 
                   nn.LeakyReLU(0.2, True)]
        nf_mult = 1
        nf_mult_prev = 1
        for n in range(1, n_layers):  # gradually increase the number of filters
            nf_mult_prev = nf_mult
            nf_mult = min(2 ** n, 8)
            modules += [
                nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=4, stride=2, padding=1, bias=False),
                nn.BatchNorm2d(ndf * nf_mult),
                nn.LeakyReLU(0.2, True)
            ]

        nf_mult_prev = nf_mult
        nf_mult = min(2 ** n_layers, 8)
        modules += [
            nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=4, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(ndf * nf_mult),
            nn.LeakyReLU(0.2, True)
        ]

        modules += [
            nn.Conv2d(ndf * nf_mult, 1, kernel_size=4, stride=1, padding=1),
            # nn.Sigmoid()
            ]  # output 1 channel prediction map
        
        self.network = nn.Sequential(*modules)

    def forward(self, input):
        """Standard forward."""
        return self.network(input)

