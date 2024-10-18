import cv2
import glob
import logging
import os
import hydra
from hydra.utils import instantiate
from omegaconf import DictConfig
from tqdm.auto import tqdm

# Global logging configuration for more consistency and clarity in output
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


@hydra.main(config_path="../config", config_name="preprocessing", version_base=None)
def preprocess_images(config: DictConfig):
    """
    Function to preprocess images based on the provided configuration.
    
    Args:
        config (DictConfig): Hydra configuration dictionary that includes preprocessing, data paths, and file settings.
    """
    logging.info("-" * 50)
    logging.info("Preprocessing started...")
    
    # Instantiate the preprocessing object from the config
    pre_processor = instantiate(config.preprocessing)
    dir_root = os.getcwd()  # Get the current working directory for relative pathing
    total_saved_images = 0
    
    # Iterate over directories and classes defined in the configuration
    for data_dir in config.data["dirs"]:
        logging.info(f"Processing directory: {data_dir}...")
        
        for class_label in config.data["classes"]:
            image_class_path = os.path.join(data_dir, class_label)
            logging.info(f"Processing class: {image_class_path}")
            
            # Fetch all image paths matching the defined file format
            image_paths = glob.glob(os.path.join(dir_root, config.data["raw_path"], image_class_path, f"*.{config.data['file']}"))
            
            # Process each image file
            for idx, image_path in enumerate(tqdm(image_paths, desc=f"Processing images for {class_label}")):
                save_path = os.path.join(dir_root, config.data["processed_path"], image_class_path, os.path.basename(image_path))
                
                # Read and process the image using the defined pre_processor
                image = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
                processed_image = pre_processor.do_process(image)
                
                # Save the processed image to the designated path
                cv2.imwrite(save_path, processed_image)
                total_saved_images += 1
    
    logging.info(f"Total {total_saved_images} images saved.")
    logging.info("Image preprocessing finished.")
    logging.info("-" * 50)


@hydra.main(config_path="../config", config_name="autoencoder", version_base=None)
def train_autoencoder(config: DictConfig):
    """
    Function to train an autoencoder model as per the configuration.
    
    Args:
        config (DictConfig): Hydra configuration dictionary that includes autoencoder parameters and model settings.
    """
    logging.info("-" * 50)
    logging.info("Autoencoder training started...")
    
    # Instantiate the necessary objects from the config
    training_executor = instantiate(config.autoencoder_params)
    autoencoder_model = instantiate(config.autoencoder)
    
    # Log the model name for tracking
    logging.info(f"Training started with model: {autoencoder_model.name}")
    
    # Execute the training process
    training_executor.execute(autoencoder_model)
    
    logging.info(f"Training with {autoencoder_model.name} completed.")
    logging.info("-" * 50)


@hydra.main(config_path="../config", config_name="autoencoder", version_base=None)
def predict_autoencoder(config: DictConfig):
    """
    Function to make predictions using the trained autoencoder model.
    
    Args:
        config (DictConfig): Hydra configuration dictionary that includes autoencoder parameters, model, and prediction settings.
    """
    logging.info("-" * 50)
    logging.info("Autoencoder prediction started...")
    
    # Instantiate necessary components
    training_executor = instantiate(config.autoencoder_params)
    autoencoder_model = instantiate(config.autoencoder)
    prediction_executor = instantiate(config.autoencoder_prediction)
    
    # Execute prediction process
    logging.info(f"Prediction process with model: {autoencoder_model.name} started...")
    prediction_executor.execute(autoencoder_model, training_executor.batch_size)
    
    logging.info(f"Prediction with {autoencoder_model.name} completed.")
    logging.info("-" * 50)


if __name__ == "__main__":
    #preprocess_images()
    train_autoencoder()
    #predict_autoencoder()
