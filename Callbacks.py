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
 
    def on_epoch_end(self, epoch: int, trainer, logs = None):
        trainer.generator.eval()
        with torch.no_grad():
            fake_images = trainer.generator(self.fixed_latent)  # (N, 1, 28, 28) in [-1, 1]
 
        # Rescale [-1, 1] → [0, 1] for saving
        fake_images = (fake_images * 0.5) + 0.5
 
        path = os.path.join(self.save_dir, f"epoch_{epoch + 1:03d}.png")
        save_image(fake_images, path, nrow=self.num_img)
        print(f"  [GANMonitor] Saved preview → {path}")