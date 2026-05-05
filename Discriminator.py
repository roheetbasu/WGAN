from torch import nn, Tensor
from config import IMG_SHAPE

class Discriminator(nn.Module):
    
    def __init__(self, img_shape = IMG_SHAPE):
        super().__init__()
        
        in_channels = img_shape[0] # 1 for MNIST
        
        self.net = nn.Sequential(
            #28 x 28 -> 32 x 32
            nn.ZeroPad2d(2),
            
            #32 x 32 -> 16x16
            nn.Conv2d(in_channels, 64, kernel_size=5, stride=2, padding=2),
            nn.LeakyReLU(0.2, inplace=True),
            
            #16x16 ->  8x8
            nn.Conv2d(64, 128, kernel_size=5, stride=2, padding=2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout2d(0.3),
            
            #8x8 -> 4x4
            nn.Conv2d(128, 256, kernel_size=5, stride=2, padding=2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout2d(0.3),
            
            #4x4 -> 2x2
            nn.Conv2d(256, 512, kernel_size=5, stride=2, padding=2),
            nn.LeakyReLU(0.2, inplace=True),
            
            #Head
            nn.Flatten(),
            nn.Dropout(0.2),
            nn.Linear(512 * 2 * 2,1) # no sigmoid - raw critic score      
        )
        
        def forward(self, img):
            return self.net(img)
        
def build_discriminator(img_shape: tuple = IMG_SHAPE) -> Discriminator:
    return Discriminator(img_shape=img_shape)