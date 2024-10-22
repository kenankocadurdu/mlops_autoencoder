import os
from PIL import Image
import torch.utils.data as data

class Dataset(data.Dataset):
    def __init__(self, root_dir, transform=None, valid_extensions=('jpg', 'jpeg', 'png', 'bmp')):
        """
        Custom Dataset class for loading images.

        Args:
            root_dir (str): Directory where images are stored.
            transform (callable, optional): Optional transform to be applied on a sample.
            valid_extensions (tuple, optional): Valid image file extensions. Defaults to ('jpg', 'jpeg', 'png', 'bmp').
        """
        if not os.path.isdir(root_dir):
            raise ValueError(f"Provided root_dir '{root_dir}' is not a valid directory.")

        self.root_dir = root_dir
        self.transform = transform
        self.valid_extensions = valid_extensions
        self.image_paths = self._load_image_paths()

    def _load_image_paths(self):
        """Load valid image paths from the root directory."""
        image_paths = []
        for fname in os.listdir(self.root_dir):
            if fname.lower().endswith(self.valid_extensions):
                full_path = os.path.join(self.root_dir, fname)
                image_paths.append(full_path)
        if not image_paths:
            raise FileNotFoundError(f"No valid images found in '{self.root_dir}' with extensions {self.valid_extensions}.")
        return image_paths

    def __len__(self):
        """Return the total number of images."""
        return len(self.image_paths)

    def __getitem__(self, idx):
        """
        Load and return an image from the dataset at the given index.

        Args:
            idx (int): Index of the image to retrieve.

        Returns:
            Image: The loaded image, potentially transformed.
        """
        if idx >= len(self.image_paths) or idx < 0:
            raise IndexError(f"Index {idx} is out of bounds for dataset of size {len(self.image_paths)}.")
        
        img_path = self.image_paths[idx]
        try:
            image = Image.open(img_path).convert("RGB")
        except Exception as e:
            raise IOError(f"Error loading image at '{img_path}': {e}")
        
        if self.transform:
            image = self.transform(image)
        
        return image

