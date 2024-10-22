import os
import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from utils import data_generator
from tqdm.auto import tqdm
import logging

class Executor:
    def __init__(self, path_testset: str, save_path: str, path_pth: str, normalize: bool, path_trainset: str, product: bool) -> None:
        self.path_testset = path_testset
        self.save_path = save_path
        self.path_pth = path_pth
        self.normalize = normalize
        self.path_trainset = path_trainset
        self.product = product

        self._log_initial_params()

    def _log_initial_params(self):
        logging.info("-" * 50)
        logging.info("Executor initialized with parameters:")
        logging.info(f"save_path: {self.save_path}")
        logging.info(f"path_pth: {self.path_pth}")
        logging.info(f"normalize: {self.normalize}")
        logging.info(f"path_trainset: {self.path_trainset}")
        logging.info(f"product: {self.product}")
        logging.info("-" * 50)

    def execute(self, model_generator, batch_size: int) -> None:
        transform_pipeline = self._get_transform_pipeline(model_generator)

        # Prepare datasets and dataloaders
        train_dataset = data_generator.Dataset(self.path_trainset, transform=transform_pipeline)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle = True, num_workers = 4, pin_memory = True, prefetch_factor=2)

        if self.normalize:
            mean, std = self._calculate_normalization(train_loader)
            transform_pipeline = self._get_transform_pipeline(model_generator, mean, std)

        test_dataset = data_generator.Dataset(self.path_testset, transform=transform_pipeline)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers = 4, pin_memory = True, prefetch_factor=2, drop_last=True)

        # Load and run the model
        self._run_inference(model_generator, test_loader, batch_size)

    def _get_transform_pipeline(self, model_generator, mean=None, std=None):
        transformations = [
            transforms.Resize((model_generator.image_size, model_generator.image_size)),
            transforms.ToTensor(),
        ]
        if mean and std:
            transformations.append(transforms.Normalize(mean.tolist(), std.tolist()))

        return transforms.Compose(transformations)

    def _calculate_normalization(self, data_loader: DataLoader):
        logging.info("Normalization started...")
        mean, std = batch_mean_and_sd(data_loader)
        logging.info(f"Normalization finished. Mean: {mean}, Std: {std}")
        return mean, std

    def _run_inference(self, model_generator, data_loader: DataLoader, batch_size: int) -> None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = model_generator.model.to(device)
        model.load_state_dict(torch.load(self.path_pth))
        model.eval()

        os.makedirs(self.save_path, exist_ok=True)

        logging.info("Starting inference...")

        with torch.no_grad():
            for i, images in enumerate(tqdm(data_loader, 0)):

                images = images.to(device)

                # Generate multiple outputs per image
                for a in range(5):
                    outputs = model(images)
                    self._save_output_images(outputs, images, data_loader, i, batch_size, a)

        logging.info("Inference completed.")



    def _save_output_images(self, outputs, images, data_loader, batch_idx, batch_size, iteration):
        for k in range(batch_size):
            output_image = transforms.ToPILImage()(outputs[k].cpu().squeeze(0))
            original_image = transforms.ToPILImage()(images[k].cpu().squeeze(0))
            fname = data_loader.dataset.image_paths[batch_idx * batch_size + k]

            output_dir = os.path.join(self.save_path, fname.split("/")[-2])
            os.makedirs(output_dir, exist_ok=True)

            output_image.save(os.path.join(output_dir, f"{fname.split('/')[-1]}_{iteration}_.png"))
            original_image.save(os.path.join(output_dir, f"{fname.split('/')[-1]}_orig.png"))



def batch_mean_and_sd(data_loader: DataLoader):
    cnt = 0
    fst_moment = torch.zeros(3)
    snd_moment = torch.zeros(3)

    for images, _ in tqdm(data_loader, desc="Calculating mean and std"):
        b, c, h, w = images.shape
        nb_pixels = b * h * w

        sum_ = torch.sum(images, dim=[0, 2, 3])
        sum_of_square = torch.sum(images ** 2, dim=[0, 2, 3])

        fst_moment = (cnt * fst_moment + sum_) / (cnt + nb_pixels)
        snd_moment = (cnt * snd_moment + sum_of_square) / (cnt + nb_pixels)

        cnt += nb_pixels

    mean = fst_moment
    std = torch.sqrt(snd_moment - fst_moment ** 2)

    return mean, std
