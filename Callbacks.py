import os
import torch
from torchvision.utils import save_image
 
from config import NOISE_DIM, DEVICE

class GANMonitor:
    def __init__(
        self,
        num_img: int = 6,
        latent_dim: int = NOISE_DIM,
        save_dir: str = "generated",
        device: torch.device = DEVICE,
    ):
        self.num_img = num_img
        self.save_dir = save_dir
        self.device = device
 
        os.makedirs(save_dir, exist_ok=True)
 
        # Fix latent vectors so progress is visually comparable across epochs
        self.fixed_latent = torch.randn(num_img, latent_dim, device=device)