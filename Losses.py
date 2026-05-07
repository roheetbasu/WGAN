import torch
from torch import Tensor

def discriminator_loss(real_score: Tensor, fake_score:Tensor):
    return fake_score.mean() - real_score.mean()
