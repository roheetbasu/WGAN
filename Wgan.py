import torch
from torch import nn, Tensor
from torch.utils.data import DataLoader

from Losses import discriminator_loss, generator_loss, gradient_penalty
from config import DISCRIMINATOR_EXTRA_STEPS, GP_WEIGHT, NOISE_DIM, DEVICE

