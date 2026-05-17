import torch

import config
from data import get_dataloader
from Generator import build_generator
from Discriminator import build_discriminator
from Wgan import WGANTrainer
from Callbacks import GANMonitor

def main():
    
    # data
    dataloader = get_dataloader(batch_size = config.BATCH_SIZE)
    
    # models
    g_model = build_generator(noise_dim=config.NOISE_DIM)
    d_model = build_discriminator(img_shape=config.IMG_SHAPE)
 
    print(g_model)
    print(d_model)
 
    # ── 3. Optimizers ──────────────────────────────────────────────────
    g_optimizer = torch.optim.Adam(
        g_model.parameters(),
        lr=config.LEARNING_RATE,
        betas=(config.ADAM_BETA_1, config.ADAM_BETA_2),
    )
    d_optimizer = torch.optim.Adam(
        d_model.parameters(),
        lr=config.LEARNING_RATE,
        betas=(config.ADAM_BETA_1, config.ADAM_BETA_2),
    )
 
    # ── 4. Trainer ─────────────────────────────────────────────────────
    trainer = WGANTrainer(
        generator=g_model,
        discriminator=d_model,
        g_optimizer=g_optimizer,
        d_optimizer=d_optimizer,
        latent_dim=config.NOISE_DIM,
        d_steps=config.DISCRIMINATOR_EXTRA_STEPS,
        gp_weight=config.GP_WEIGHT,
        device=config.DEVICE,
    )
 
    # ── 5. Callbacks ───────────────────────────────────────────────────
    monitor = GANMonitor(
        num_img=config.NUM_PREVIEW_IMG,
        latent_dim=config.NOISE_DIM,
        save_dir="generated",
        device=config.DEVICE,
    )
 
    # ── 6. Train ───────────────────────────────────────────────────────
    trainer.fit(
        dataloader=dataloader,
        epochs=config.EPOCHS,
        callbacks=[monitor],
    )
 
 
if __name__ == "__main__":
    main()