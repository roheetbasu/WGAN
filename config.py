#All hyper parameters

IMG_SHAPE = (28 ,28 ,1)
BATCH_SIZE = 64
NOISE_DIM = 128
EPOCHS = 20
NUM_PREVIEW_IMG = 3

# Critic / gradient penalty
DISCRIMINATOR_EXTRA_STEPS = 3
GP_WEIGHT = 10.0
 
# Adam optimizer hyperparameters (from the WGAN-GP paper)
LEARNING_RATE = 0.0002
ADAM_BETA_1 = 0.5
ADAM_BETA_2 = 0.9