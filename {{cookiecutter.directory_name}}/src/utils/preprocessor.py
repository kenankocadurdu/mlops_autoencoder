import numpy as np
import cv2

class ImageProcessor:
    """
    A class to handle preprocessing of medical images with various filtering and enhancement techniques.
    
    Attributes:
    -----------
    image_size : int
        The size to which images will be resized.
    remove_noise : bool
        Flag to apply noise reduction.
    apply_clahe : bool
        Flag to apply Contrast Limited Adaptive Histogram Equalization (CLAHE).
    apply_gamma_corr : bool
        Flag to apply gamma correction to the image.
    median_filter : bool
        Flag to apply a median filter for noise reduction.
    adaptive_mean_filter : bool
        Flag to apply an adaptive mean filter.
    gaussian_filter : bool
        Flag to apply Gaussian filtering for blurring.
    """
    
    def __init__(self, image_size: int, remove_noise: bool = False, apply_clahe: bool = False, 
                 apply_gamma_corr: bool = False, median_filter: bool = False,
                 adaptive_mean_filter: bool = False, gaussian_filter: bool = False) -> None:
        """
        Initializes the ImageProcessor with the desired image transformations.

        Parameters:
        -----------
        image_size : int
            The target image size after preprocessing.
        remove_noise : bool
            If True, noise reduction will be applied to the image.
        apply_clahe : bool
            If True, CLAHE will be applied to enhance the contrast of the image.
        apply_gamma_corr : bool
            If True, gamma correction will be applied to adjust the brightness.
        median_filter : bool
            If True, a median filter will be applied to reduce noise.
        adaptive_mean_filter : bool
            If True, adaptive mean filtering will be applied for noise reduction.
        gaussian_filter : bool
            If True, Gaussian filtering will be applied to the image.
        """
        self.image_size = image_size
        self.remove_noise = remove_noise
        self.apply_clahe = apply_clahe
        self.apply_gamma_corr = apply_gamma_corr
        self.median_filter = median_filter
        self.adaptive_mean_filter = adaptive_mean_filter
        self.gaussian_filter = gaussian_filter

    def process_image(self, image: np.ndarray) -> np.ndarray:
        """
        Processes the input image according to the initialized flags.

        Parameters:
        -----------
        image : np.ndarray
            The input image to be processed.

        Returns:
        --------
        np.ndarray
            The processed and resized image.
        """
        if image is None:
            raise ValueError("Input image is None")
        
        if self.remove_noise:
            image = self._remove_noise(image)
        if self.apply_clahe:
            image = self._apply_clahe(image)
        if self.apply_gamma_corr:
            image = self._apply_gamma_correction(image)
        if self.median_filter:
            image = self._apply_median_filter(image)
        if self.adaptive_mean_filter:
            image = self._apply_adaptive_mean_filter(image)
        if self.gaussian_filter:
            image = self._apply_gaussian_filter(image)
        
        return self._resize_image(image, self.image_size)

    def _remove_noise(self, image: np.ndarray, ksize: int = 3) -> np.ndarray:
        """
        Applies noise reduction by combining the original image with Gaussian blurring.

        Parameters:
        -----------
        image : np.ndarray
            The input image.
        ksize : int
            The kernel size for Gaussian blurring. Defaults to 3.

        Returns:
        --------
        np.ndarray
            The noise-reduced image.
        """
        weighted_image = cv2.addWeighted(image, 1, image, -0.2, 0)
        return cv2.GaussianBlur(weighted_image, (ksize, ksize), 0)

    def _apply_clahe(self, image: np.ndarray, clip_limit: float = 2.0, tile_grid_size: tuple = (32, 32)) -> np.ndarray:
        """
        Applies CLAHE to enhance image contrast.

        Parameters:
        -----------
        image : np.ndarray
            The input image.
        clip_limit : float
            Threshold for contrast clipping. Defaults to 2.0.
        tile_grid_size : tuple
            Size of the grid for histogram equalization. Defaults to (32, 32).

        Returns:
        --------
        np.ndarray
            The contrast-enhanced image.
        """
        clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
        return clahe.apply(image)

    def _apply_gamma_correction(self, image: np.ndarray, gamma: float = 1.3) -> np.ndarray:
        """
        Adjusts the image brightness using gamma correction.

        Parameters:
        -----------
        image : np.ndarray
            The input image.
        gamma : float
            The gamma value for correction. Defaults to 1.3.

        Returns:
        --------
        np.ndarray
            The gamma-corrected image.
        """
        normalized_image = image / 255.0
        corrected_image = np.power(normalized_image, gamma) * 255.0
        return corrected_image.astype(np.uint8)

    def _apply_median_filter(self, image: np.ndarray, kernel_size: int = 3) -> np.ndarray:
        """
        Applies a median filter to reduce noise.

        Parameters:
        -----------
        image : np.ndarray
            The input image.
        kernel_size : int
            The kernel size for the median filter. Defaults to 3.

        Returns:
        --------
        np.ndarray
            The image after median filtering.
        """
        return cv2.medianBlur(image, kernel_size)

    def _apply_adaptive_mean_filter(self, image: np.ndarray, window_size: int = 3, constant_c: int = 15) -> np.ndarray:
        """
        Applies an adaptive mean filter for noise reduction on grayscale images.

        Parameters:
        -----------
        image : np.ndarray
            The input image (2D grayscale).
        window_size : int
            The size of the sliding window. Defaults to 3.
        constant_c : int
            A constant subtracted from the mean for adaptive behavior. Defaults to 15.

        Returns:
        --------
        np.ndarray
            The filtered image.
        """
        if image.ndim != 2:
            raise ValueError("Adaptive mean filter requires a 2D grayscale image.")
        
        pad = window_size // 2
        padded_image = np.pad(image, pad, mode='constant', constant_values=0)
        output_image = np.zeros_like(image)

        for i in range(pad, padded_image.shape[0] - pad):
            for j in range(pad, padded_image.shape[1] - pad):
                window = padded_image[i-pad:i+pad+1, j-pad:j+pad+1]
                local_mean = np.mean(window)
                output_image[i-pad, j-pad] = np.clip(local_mean - constant_c, 0, 255)

        return output_image

    def _apply_gaussian_filter(self, image: np.ndarray, kernel_size: tuple = (5, 5), sigma_x: int = 0) -> np.ndarray:
        """
        Applies a Gaussian filter to blur the image.

        Parameters:
        -----------
        image : np.ndarray
            The input image.
        kernel_size : tuple
            Size of the Gaussian kernel. Defaults to (5, 5).
        sigma_x : int
            Standard deviation in the x-direction. Defaults to 0.

        Returns:
        --------
        np.ndarray
            The blurred image.
        """
        return cv2.GaussianBlur(image, kernel_size, sigmaX=sigma_x)

    def _resize_image(self, image: np.ndarray, image_size: int) -> np.ndarray:
        """
        Resizes the image to the target size.

        Parameters:
        -----------
        image : np.ndarray
            The input image.
        image_size : int
            The desired output image size.

        Returns:
        --------
        np.ndarray
            The resized image.
        """
        return cv2.resize(image, (image_size, image_size))
