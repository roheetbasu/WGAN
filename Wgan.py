import torch
from torch import nn, Tensor
from torch.utils.data import DataLoader

from Losses import discriminator_loss, generator_loss, gradient_penalty
from config import DISCRIMINATOR_EXTRA_STEPS, GP_WEIGHT, NOISE_DIM, DEVICE

class WGANTrainer:
    
    def __init__(self,
                 generator: nn.Module,
                 discriminator: nn.Module,
                 g_optimizer: torch.optim.Optimizer,
                 d_optimizer: torch.optim.Optimizer,
                 latent_dim: int = NOISE_DIM,
                 d_steps: int = DISCRIMINATOR_EXTRA_STEPS,
                 gp_weight: float = GP_WEIGHT,
                 device: torch.device = DEVICE,
                 ):
        self.generator = generator.to(device)
        self.discriminator = discriminator.to(device)
        self.g_optimizer = g_optimizer
        self.d_optimizer = d_optimizer
        self.latent_dim = latent_dim
        self.d_steps = d_steps
        self.gp_weight = gp_weight
        self.device = device
        
        
        def train_crtic(self, real_images: Tensor):
            
            batch_size = real_images.size(0)
            
            z = torch.randn(batch_size, self.latent_dim, device=self.device)
            with torch.no_grad():
                fake_images = self.generator(z)
                
            real_scores = self.discriminator(real_images)
            fake_scores = self.discriminator(fake_images)
            
            d_loss = discriminator_loss(real_scores, fake_scores)
            gp = gradient_penalty(
                self.discriminator, real_images, fake_images.detach(), self.device
            )
            total_d_loss =  d_loss + self.gp_weight * gp
            
            self.d_optimizer.zero_grad()
            total_d_loss.backward()
            self.d_optimizer.step()
            
            return total_d_loss.item()
       