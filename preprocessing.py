import os
import numpy as np
from PIL import Image
import cv2
from torchvision import transforms

# Constants
EMOTION_LABELS = {
    0: "Colère",
    1: "Dégoût",
    2: "Peur",
    3: "Joie",
    4: "Neutre",
    5: "Tristesse",
    6: "Surprise",
}

IMG_SIZE = 48
MEAN = 0.5071
STD = 0.2520

NOISE_LOW = 50
NOISE_HIGH = 150


def _estimate_noise(img_np: np.ndarray) -> float:
    smoothed = cv2.GaussianBlur(img_np, (5, 5), 1.0)
    residual = img_np.astype(np.float32) - smoothed.astype(np.float32)
    return float(np.std(residual))


class AdaptiveDenoiser(object):
    def __init__(self, noise_low: float = NOISE_LOW, noise_high: float = NOISE_HIGH):
        self.noise_low = noise_low
        self.noise_high = noise_high

    def __call__(self, img: Image.Image) -> Image.Image:
        img_np = np.array(img, dtype=np.uint8)
        noise = _estimate_noise(img_np)

        if noise < self.noise_low:
            denoised = img_np
        elif noise < self.noise_high:
            denoised = cv2.bilateralFilter(img_np, d=5, sigmaColor=50, sigmaSpace=50)
        else:
            denoised = cv2.fastNlMeansDenoising(img_np, h=10, templateWindowSize=7, searchWindowSize=21)

        return Image.fromarray(denoised)


class CLAHEEnhancer(object):
    def __init__(self, clip_limit: float = 1.5, grid_size: tuple = (4, 4)):
        self.clip_limit = clip_limit
        self.grid_size = grid_size

    def __call__(self, img: Image.Image) -> Image.Image:
        img_np = np.array(img, dtype=np.uint8)
        clahe = cv2.createCLAHE(clipLimit=self.clip_limit, tileGridSize=self.grid_size)
        img_final = clahe.apply(img_np)
        return Image.fromarray(img_final)


class _ForceNLMDenoiser(object):
    def __call__(self, img: Image.Image) -> Image.Image:
        img_np = np.array(img, dtype=np.uint8)
        denoised = cv2.fastNlMeansDenoising(img_np, h=15, templateWindowSize=7, searchWindowSize=21)
        return Image.fromarray(denoised)


def _build_preprocessing_steps(pipeline: str) -> list:
    if pipeline == "minimal":
        return [CLAHEEnhancer()]
    elif pipeline == "light":
        return [AdaptiveDenoiser(), CLAHEEnhancer()]
    elif pipeline == "aggressive":
        return [_ForceNLMDenoiser(), CLAHEEnhancer()]
    else:
        raise ValueError(f"Pipeline inconnu: '{pipeline}'")


def _make_train_transform(pipeline: str = "light"):
    preprocessing_steps = _build_preprocessing_steps(pipeline)
    return transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        *preprocessing_steps,
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=10),
        transforms.RandomAffine(degrees=0, translate=(0.05, 0.05), scale=(0.95, 1.05)),
        transforms.ColorJitter(brightness=0.3, contrast=0.3),
        transforms.RandomAdjustSharpness(sharpness_factor=2, p=0.3),
        transforms.ToTensor(),
        transforms.Normalize(mean=[MEAN], std=[STD]),
    ])


def _make_test_transform(pipeline: str = "light"):
    preprocessing_steps = _build_preprocessing_steps(pipeline)
    return transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        *preprocessing_steps,
        transforms.ToTensor(),
        transforms.Normalize(mean=[MEAN], std=[STD]),
    ])


from torchvision import datasets
from torch.utils.data import Dataset


class NumpyImageDataset(Dataset):
    def __init__(self, X, y, transform=None):
        self.X = X
        self.y = y
        self.transform = transform

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        img = self.X[idx].squeeze()
        img_pil = Image.fromarray((img * 255).astype(np.uint8), mode='L')
        if self.transform:
            img_pil = self.transform(img_pil)
        label = self.y[idx].argmax()
        return img_pil, label
