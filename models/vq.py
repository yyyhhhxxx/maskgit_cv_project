import torch
import torch.nn as nn
import torch.nn.functional as F


class VectorQuantizer(nn.Module):
    '''
    Reference:
    Van Den Oord, Aaron, and Oriol Vinyals. "Neural discrete representation learning." Advances in neural information processing systems 30 (2017).
    '''
    def __init__(self,
                 num_embeddings: int,
                 embedding_dim: int,
                 beta: float = 0.25):
        super(VectorQuantizer, self).__init__()
        # num_embeddings: the number of embeddings of codebook
        # embedding_dim: the dimensions of each embedding
        # beta: the weight of 'embedding loss'
        self.codebook = nn.Embedding(num_embeddings, embedding_dim)
        self.codebook.weight.data.uniform_(-1.0 / num_embeddings, 1.0 / num_embeddings)
        self.beta = beta

    def forward(self, latents: torch.Tensor) :

        # latents: features from encoder

        
        # Flatten latents
        latents = latents.permute(0, 2, 3, 1).contiguous()
        latents_shape = latents.shape
        latents = latents.view(-1, latents_shape[-1])
        

        # Compute L2 distance between latents and embedding weights
        L2_distance = torch.cdist(latents, self.codebook.weight)

        # Get the encoding indices that has the min distance

        encoding_indices = torch.argmin(L2_distance, dim=1)

        # Quantize the latents

        quantized_latents = self.codebook(encoding_indices)

        # Compute the VQ Losses
        commitment_loss = F.mse_loss(quantized_latents.detach(), latents)
        embedding_loss = F.mse_loss(quantized_latents, latents.detach())
        vq_loss = commitment_loss * self.beta + embedding_loss

        # Add the residue back to the latents
        quantized_latents = latents + (quantized_latents - latents).detach()

        # Reshape quantized latents
        quantized_latents = quantized_latents.view(latents_shape)
        quantized_latents = quantized_latents.permute(0, 3, 1, 2).contiguous()



        # Return the quantized latents, vq_loss and encoding indices
        return quantized_latents, vq_loss, encoding_indices
    