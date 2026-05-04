from torch import nn, Tensor
from config import NOISE_DIM

class Generator(nn.Module):
    
    def __init__(self, noise_dim: int = NOISE_DIM):
        super().__init__()
        self.noise_dim = NOISE_DIM
        
        self.project = nn.Sequential(
            nn.Linear(noise_dim, 4 * 4 * 256, bias = False),
            nn.BatchNorm1d(4 * 4 * 256),
            nn.LeakyReLU(0.2, inplace=True)
        )

        self.up1 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="nearest"),
            nn.Conv2d(256, 128, kernel_size=3, padding=1, bias=False)
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True)
        )
