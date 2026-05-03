from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from config import BATCH_SIZE, IMG_SHAPE

def get_dataloader(batch_size: int = BATCH_SIZE, data_root: str = "./data"):
    
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.5,), std=(0.5,)),
    ])
    
    dataset = datasets.MNIST(
        root = data_root,
        train = True,
        download = True,
        transform = transform,
    )