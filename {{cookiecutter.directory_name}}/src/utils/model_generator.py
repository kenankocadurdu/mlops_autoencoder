import torch
import torch.nn as nn
import torch.nn.functional as F
from utils import model_arch
import logging
from fastai.vision.all import *

class Generator:
    """
    A class to generate different types of models (e.g., AutoEncoder).
    
    Args:
        name (str): The name of the model.
        num_class (int): The number of output classes (for classification models).
        image_size (int): The input image size (assumed to be square images).
    """
    def __init__(self, name: str, num_class: int, image_size: int):
        self.name = name
        self.num_class = num_class
        self.image_size = image_size

        # Dynamically create the model based on the provided name
        if self.name == "AutoEncoder":
            self.model = AutoEncoder(self.image_size)
        else:
            raise ValueError(f"Model '{self.name}' is not supported.")
        
        # Logging the details of the model
        logging.info("-" * 50)
        logging.info("Model generated successfully.")
        logging.info(f"Model name: {self.name}")
        logging.info(f"Number of classes: {self.num_class}")
        logging.info(f"Image size: {self.image_size}x{self.image_size}")


##################### AutoEncoders ##############################

class AutoEncoder(model_arch.BaseModule):
    """
    A Convolutional AutoEncoder model.
    
    Args:
        image_size (int): The input image size (assumed to be square images, e.g., 96x96).
    """
    def __init__(self, image_size: int):
        super(AutoEncoder, self).__init__()
        self.encoder = self.build_encoder()
        self.decoder = self.build_decoder()
        
        # Verify that the image size is supported (minimum size: 96x96)
        if image_size < 96:
            raise ValueError("Image size must be at least 96x96.")

    def build_encoder(self) -> nn.Sequential:
        """
        Builds the encoder part of the AutoEncoder.
        
        Returns:
            nn.Sequential: Encoder layers.
        """
        return nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, stride=2, padding=1),  # Output: 48x48x16
            nn.ReLU(True),
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),  # Output: 24x24x32
            nn.ReLU(True),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),  # Output: 12x12x64
            nn.ReLU(True),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1), # Output: 6x6x128
            nn.ReLU(True),
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1) # Output: 3x3x256
        )

    def build_decoder(self) -> nn.Sequential:
        """
        Builds the decoder part of the AutoEncoder.
        
        Returns:
            nn.Sequential: Decoder layers.
        """
        return nn.Sequential(
            nn.ConvTranspose2d(256, 128, kernel_size=3, stride=2, padding=1, output_padding=1),  # Output: 6x6x128
            nn.ReLU(True),
            nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1),   # Output: 12x12x64
            nn.ReLU(True),
            nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1),    # Output: 24x24x32
            nn.ReLU(True),
            nn.ConvTranspose2d(32, 16, kernel_size=3, stride=2, padding=1, output_padding=1),    # Output: 48x48x16
            nn.ReLU(True),
            nn.ConvTranspose2d(16, 3, kernel_size=3, stride=2, padding=1, output_padding=1),     # Output: 96x96x3
            nn.Sigmoid()  # Output normalized to [0, 1]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Defines the forward pass through the AutoEncoder.
        
        Args:
            x (torch.Tensor): Input image tensor.
        
        Returns:
            torch.Tensor: Reconstructed image tensor.
        """
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded
