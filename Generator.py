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

        # 4 x 4 -> 8 x 8
        self.up1 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="nearest"),
            nn.Conv2d(256, 128, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True)
        )
        # 8 x 8 -> 16 x 16
        self.up2 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="nearest"),
            nn.Conv2d(128, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2, inplace=True)
        )
        
        # 16 x 16 -> 32 x 32
        self.up3 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="nearest"),
            nn.Conv2d(64, 1, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(1),
            nn.Tanh()
        )
        
        self.crop = nn.ZeroPad2d(-2)
        
    
    def forward(self, z:Tensor):
        x = self.project(z)
        x = x.view(-1, 256, 4 , 4)
        x = self.up1(x)
        x = self.up2(x)
        x = self.up3(x)
        return self.crop(x)
    
def build_generator(noise_dim: int = NOISE_DIM):
    return Generator(noise_dim=NOISE_DIM)
