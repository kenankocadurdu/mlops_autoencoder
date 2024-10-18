import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import logging
from tqdm.auto import tqdm
from utils import data_generator
from utils import model_arch


class Executor:
    """
    Class to manage the execution of training, validation, and testing of a deep learning model.

    Attributes:
    -----------
    path_dataset_train : str
        Path to the training dataset.
    path_dataset_val : str
        Path to the validation dataset.
    batch_size : int
        Batch size for training and validation data loaders.
    num_threads : int
        Number of CPU threads to use for data loading.
    device_id : int
        GPU device ID to use for training.
    num_epochs : int
        Number of epochs to train the model.
    lr : float
        Learning rate for the optimizer.
    patience : int
        Early stopping patience for model training.
    opt_func : str
        Optimizer function to use for training.
    criterion : str
        Loss function to be used.
    normalize : bool
        Whether to apply normalization to the dataset.
    ml_flow : bool
        Whether to log the training process in MLflow.
    log_desc : str
        Description for logging purposes.
    path_testset : str
        Path to the test dataset.
    save_path : str
        Directory path to save the model.
    path_pth : str
        Path to the saved model weights.
    load_w : bool
        Flag to determine if model weights should be loaded from a pre-trained model.
    """

    def __init__(self, path_dataset_train, path_dataset_val, batch_size: int, num_threads: int, device_id: int, 
                 num_epochs: int, lr: float, patience: int, opt_func: str, criterion: str, normalize: bool, 
                 ml_flow: bool, log_desc: str, path_testset: str, save_path: str, path_pth: str, load_w: bool) -> None:
        
        self.path_dataset_train = path_dataset_train
        self.path_dataset_val = path_dataset_val
        self.batch_size = batch_size
        self.num_threads = num_threads
        self.device_id = device_id
        self.num_epochs = num_epochs
        self.lr = lr
        self.patience = patience
        self.opt_func = opt_func
        self.criterion = criterion
        self.normalize = normalize
        self.ml_flow = ml_flow
        self.log_desc = log_desc
        self.path_testset = path_testset
        self.save_path = save_path
        self.path_pth = path_pth
        self.load_w = load_w

        self._log_initial_parameters()

    def _log_initial_parameters(self):
        """Logs the initial parameters for debugging and traceability."""
        logging.info("Executor Parameters:")
        params = vars(self)  # Fetches all class attributes
        for key, value in params.items():
            logging.info(f"{key}: {value}")

    def execute(self, model_generator):
        """
        Executes the training process with data augmentation, normalization (if applicable), 
        and training over specified epochs.

        Parameters:
        -----------
        model_generator : object
            The model generator object responsible for creating the model.
        """
        tt_transforms_train, tt_transforms = self._get_transforms(model_generator)

        # Prepare data loaders
        ds_train = data_generator.Dataset(self.path_dataset_train, transform=tt_transforms_train)
        ds_val = data_generator.Dataset(self.path_dataset_val, transform=tt_transforms)

        dl_train = DataLoader(ds_train, batch_size=self.batch_size, shuffle=True, 
                              num_workers=self.num_threads, pin_memory=True, prefetch_factor=2)
        dl_val = DataLoader(ds_val, batch_size=self.batch_size, shuffle=False, 
                            num_workers=self.num_threads, pin_memory=True, prefetch_factor=2)

        # Naming convention for saving the model
        pth_name = f"{model_generator.name}_{self.opt_func}_{self.criterion}"

        # Fit the model
        history, model_fit = model_arch.fit(self.num_epochs, self.lr, model_generator, dl_train, dl_val, 
                                            self.opt_func, self.patience, self.criterion, pth_name, self.ml_flow, 
                                            self.log_desc, self.save_path, self.path_pth, 
                                            self.batch_size, self.load_w)


    def _get_transforms(self, model_generator):
        """
        Creates the necessary image transforms including normalization if required.

        Parameters:
        -----------
        model_generator : object
            The model generator object that holds image-related parameters.

        Returns:
        --------
        tt_transforms_train : torchvision.transforms.Compose
            Transformations applied to the training dataset.
        tt_transforms : torchvision.transforms.Compose
            Transformations applied to the validation dataset.
        """
        # Default transformations
        tt_transforms_train = transforms.Compose([
            transforms.Resize((model_generator.image_size, model_generator.image_size)),
            transforms.ToTensor(),
        ])
        
        tt_transforms = transforms.Compose([
            transforms.Resize((model_generator.image_size, model_generator.image_size)),
            transforms.ToTensor(),
        ])

        if self.normalize:
            logging.info("Normalizing dataset...")

            # Pre-fetch data loader to calculate mean and std
            ds_train_1 = data_generator.Dataset(self.path_dataset_train, transform=tt_transforms)
            dl_train_1 = DataLoader(ds_train_1, batch_size=self.batch_size, shuffle=False, 
                                    num_workers=self.num_threads, pin_memory=True, prefetch_factor=2)

            mean, std = self._calculate_mean_and_std(dl_train_1)
            logging.info(f"Calculated Mean: {mean.tolist()}, Std: {std.tolist()}")

            # Update transformations with normalization
            tt_transforms_train = transforms.Compose([
                transforms.Resize((model_generator.image_size, model_generator.image_size)),
                transforms.RandomRotation(degrees=(0, 15)),
                transforms.RandomVerticalFlip(),
                transforms.RandomHorizontalFlip(p=0.2),
                transforms.ToTensor(),
                transforms.Normalize(mean.tolist(), std.tolist())
            ])
            tt_transforms = transforms.Compose([
                transforms.Resize((model_generator.image_size, model_generator.image_size)),
                transforms.ToTensor(),
                transforms.Normalize(mean.tolist(), std.tolist())
            ])

        return tt_transforms_train, tt_transforms

    def _calculate_mean_and_std(self, dl_train):
        """
        Calculates the mean and standard deviation of a dataset for normalization.

        Parameters:
        -----------
        dl_train : DataLoader
            DataLoader object of the training dataset.

        Returns:
        --------
        mean : torch.Tensor
            Calculated mean across channels.
        std : torch.Tensor
            Calculated standard deviation across channels.
        """
        cnt = 0
        fst_moment = torch.zeros(3)
        snd_moment = torch.zeros(3)
        
        logging.info("Starting dataset mean and standard deviation calculation...")
        
        for images, _ in tqdm(dl_train, desc="Calculating mean and std"):
            b, c, h, w = images.shape
            nb_pixels = b * h * w

            sum_ = torch.sum(images, dim=[0, 2, 3])
            sum_of_square = torch.sum(images ** 2, dim=[0, 2, 3])

            fst_moment = (cnt * fst_moment + sum_) / (cnt + nb_pixels)
            snd_moment = (cnt * snd_moment + sum_of_square) / (cnt + nb_pixels)

            cnt += nb_pixels

        mean = fst_moment
        std = torch.sqrt(snd_moment - fst_moment ** 2)

        logging.info(f"Dataset normalization - Mean: {mean}, Std: {std}")
        return mean, std
