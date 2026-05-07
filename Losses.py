import torch
from torch import Tensor

def discriminator_loss(real_score: Tensor, fake_score:Tensor):
    return fake_score.mean() - real_score.mean()

def generator_loss(fake_score: Tensor):
    return -fake_score.mean()

def gradient_penalty(
    critic: torch.nn.Module,
    real_images: Tensor,
    fake_images: Tensor,
    device: torch.device,    
):
    batch_size = real_images.size(0)
    
    #Random interpolation coefficient
    alpha = torch.rand(batch_size,1,1,1,device=device)
    interpolated = real_images + alpha * (fake_images - real_images)
    interpolated.requires_grad_(True)
    
    critic_scores = critic(interpolated)
    
    gradients = torch.autograd.grad(
        outputs=critic_scores,
        inputs=interpolated,
        grad_outputs=torch.ones_like(critic_scores),
        create_graph=True,
        retain_graph=True,
    )[0]
    
    # Flatten spatial dims and compute per-sample L2 norm
    gradients = gradients.view(batch_size, -1)
    grad_norm = gradients.norm(2, dim=1)
    penalty = ((grad_norm - 1.0) ** 2).mean()
    return penalty