"""
Classical denoising methods implementation
"""
import numpy as np
import cv2
from scipy.signal import wiener
from scipy.ndimage import median_filter
import pywt
import yaml


class ClassicalDenoisers:
    """Implementation of classical denoising methods"""
    
    def __init__(self, config_path='config.yaml'):
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
    
    def median_filter_denoise(self, image, kernel_size=5):
        """
        Median filter denoising
        Effective for salt & pepper noise
        """
        if len(image.shape) == 3:
            # Process each channel separately
            denoised = np.zeros_like(image)
            for i in range(image.shape[2]):
                denoised[:, :, i] = cv2.medianBlur(image[:, :, i], kernel_size)
            return denoised
        else:
            return cv2.medianBlur(image, kernel_size)
    
    def wiener_filter_denoise(self, image, noise_variance=None):
        """
        Wiener filter denoising
        Good for Gaussian noise
        """
        if len(image.shape) == 3:
            # Process each channel separately
            denoised = np.zeros_like(image, dtype=np.float32)
            for i in range(image.shape[2]):
                channel = image[:, :, i].astype(np.float32)
                
                # Estimate noise variance if not provided
                if noise_variance is None:
                    # Use robust median absolute deviation method
                    sigma = np.median(np.abs(channel - np.median(channel))) / 0.6745
                    noise_var = sigma ** 2
                else:
                    noise_var = noise_variance
                
                # Apply Wiener filter
                denoised_channel = wiener(channel, mysize=(5, 5), noise=noise_var)
                denoised[:, :, i] = denoised_channel
            
            return np.clip(denoised, 0, 255).astype(np.uint8)
        else:
            channel = image.astype(np.float32)
            if noise_variance is None:
                sigma = np.median(np.abs(channel - np.median(channel))) / 0.6745
                noise_var = sigma ** 2
            else:
                noise_var = noise_variance
            
            denoised = wiener(channel, mysize=(5, 5), noise=noise_var)
            return np.clip(denoised, 0, 255).astype(np.uint8)
    
    def wavelet_denoise(self, image, wavelet='db1', threshold_mode='soft'):
        """
        Wavelet transform denoising
        Effective for various noise types
        """
        if len(image.shape) == 3:
            # Process each channel separately
            denoised = np.zeros_like(image)
            for i in range(image.shape[2]):
                denoised[:, :, i] = self._wavelet_denoise_single_channel(
                    image[:, :, i], wavelet, threshold_mode
                )
            return denoised
        else:
            return self._wavelet_denoise_single_channel(image, wavelet, threshold_mode)
    
    def _wavelet_denoise_single_channel(self, channel, wavelet, threshold_mode):
        """Apply wavelet denoising to a single channel"""
        # Convert to float
        channel_float = channel.astype(np.float32)
        
        # Perform wavelet decomposition
        coeffs = pywt.wavedec2(channel_float, wavelet, level=3)
        
        # Calculate threshold using Universal Threshold method
        sigma = self._estimate_sigma(coeffs[-1])
        threshold = sigma * np.sqrt(2 * np.log(channel.size))
        
        # Apply thresholding to detail coefficients
        coeffs_thresh = list(coeffs)
        for i in range(1, len(coeffs_thresh)):
            if isinstance(coeffs_thresh[i], tuple):
                coeffs_thresh[i] = tuple(
                    pywt.threshold(c, threshold, mode=threshold_mode) 
                    for c in coeffs_thresh[i]
                )
            else:
                coeffs_thresh[i] = pywt.threshold(coeffs_thresh[i], threshold, mode=threshold_mode)
        
        # Reconstruct the image
        reconstructed = pywt.waverec2(coeffs_thresh, wavelet)
        
        # Ensure same size as input
        reconstructed = reconstructed[:channel.shape[0], :channel.shape[1]]
        
        return np.clip(reconstructed, 0, 255).astype(np.uint8)
    
    def _estimate_sigma(self, detail_coeffs):
        """Estimate noise standard deviation from detail coefficients"""
        if isinstance(detail_coeffs, tuple):
            # For multi-dimensional detail coefficients
            all_coeffs = np.concatenate([c.flatten() for c in detail_coeffs])
        else:
            all_coeffs = detail_coeffs.flatten()
        
        # Robust median absolute deviation method
        sigma = np.median(np.abs(all_coeffs - np.median(all_coeffs))) / 0.6745
        return sigma
    
    def bilateral_filter_denoise(self, image, d=9, sigma_color=75, sigma_space=75):
        """
        Bilateral filter denoising
        Preserves edges while smoothing
        """
        if len(image.shape) == 3:
            denoised = cv2.bilateralFilter(image, d, sigma_color, sigma_space)
        else:
            denoised = cv2.bilateralFilter(image, d, sigma_color, sigma_space)
        
        return denoised
    
    def nlm_denoise(self, image, h=10, template_window_size=7, search_window_size=21):
        """
        Non-Local Means denoising
        Advanced denoising method
        """
        if len(image.shape) == 3:
            denoised = cv2.fastNlMeansDenoisingColored(
                image, None, h, h, template_window_size, search_window_size
            )
        else:
            denoised = cv2.fastNlMeansDenoising(
                image, None, h, template_window_size, search_window_size
            )
        
        return denoised
    
    def denoise_image(self, image, method='median', **kwargs):
        """
        Unified interface for all denoising methods
        
        Args:
            image: Input noisy image
            method: One of ['median', 'wiener', 'wavelet', 'bilateral', 'nlm']
            **kwargs: Method-specific parameters
        
        Returns:
            Denoised image
        """
        if method == 'median':
            kernel_size = kwargs.get('kernel_size', 5)
            return self.median_filter_denoise(image, kernel_size)
        
        elif method == 'wiener':
            noise_variance = kwargs.get('noise_variance', None)
            return self.wiener_filter_denoise(image, noise_variance)
        
        elif method == 'wavelet':
            wavelet = kwargs.get('wavelet', 'db1')
            threshold_mode = kwargs.get('threshold_mode', 'soft')
            return self.wavelet_denoise(image, wavelet, threshold_mode)
        
        elif method == 'bilateral':
            d = kwargs.get('d', 9)
            sigma_color = kwargs.get('sigma_color', 75)
            sigma_space = kwargs.get('sigma_space', 75)
            return self.bilateral_filter_denoise(image, d, sigma_color, sigma_space)
        
        elif method == 'nlm':
            h = kwargs.get('h', 10)
            template_window_size = kwargs.get('template_window_size', 7)
            search_window_size = kwargs.get('search_window_size', 21)
            return self.nlm_denoise(image, h, template_window_size, search_window_size)
        
        else:
            raise ValueError(f"Unknown denoising method: {method}")


if __name__ == "__main__":
    # Test classical denoisers
    denoiser = ClassicalDenoisers()
    
    # Create test image
    test_image = np.random.randint(0, 256, (256, 256, 3), dtype=np.uint8)
    
    # Test all methods
    methods = ['median', 'wiener', 'wavelet', 'bilateral', 'nlm']
    
    for method in methods:
        print(f"Testing {method} denoising...")
        denoised = denoiser.denoise_image(test_image, method)
        print(f"  Input shape: {test_image.shape}, Output shape: {denoised.shape}")
        print(f"  Output dtype: {denoised.dtype}, Range: [{denoised.min()}, {denoised.max()}]")
