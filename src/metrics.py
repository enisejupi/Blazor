"""
Image quality metrics for denoising evaluation
"""
import numpy as np
import cv2
import yaml

# Custom implementations to avoid scikit-image dependency
def _psnr(original, denoised, data_range=255):
    """Calculate PSNR without scikit-image"""
    mse = np.mean((original.astype(float) - denoised.astype(float)) ** 2)
    if mse == 0:
        return float('inf')
    return 20 * np.log10(data_range / np.sqrt(mse))

def _ssim(original, denoised, data_range=255, multichannel=False, channel_axis=None):
    """Calculate SSIM without scikit-image using OpenCV"""
    # Convert to uint8 if needed
    if original.dtype != np.uint8:
        original = np.clip(original, 0, 255).astype(np.uint8)
    if denoised.dtype != np.uint8:
        denoised = np.clip(denoised, 0, 255).astype(np.uint8)
    
    # Use cv2.quality.QualitySSIM if available, otherwise manual calculation
    try:
        # OpenCV 4.x has quality module
        score = cv2.quality.QualitySSIM_compute(original, denoised)
        return score[0].mean()
    except AttributeError:
        # Fallback to manual SSIM calculation
        C1 = (0.01 * data_range) ** 2
        C2 = (0.03 * data_range) ** 2
        
        original = original.astype(np.float64)
        denoised = denoised.astype(np.float64)
        kernel = cv2.getGaussianKernel(11, 1.5)
        window = np.outer(kernel, kernel.transpose())
        
        mu1 = cv2.filter2D(original, -1, window)[5:-5, 5:-5]
        mu2 = cv2.filter2D(denoised, -1, window)[5:-5, 5:-5]
        mu1_sq = mu1 ** 2
        mu2_sq = mu2 ** 2
        mu1_mu2 = mu1 * mu2
        sigma1_sq = cv2.filter2D(original ** 2, -1, window)[5:-5, 5:-5] - mu1_sq
        sigma2_sq = cv2.filter2D(denoised ** 2, -1, window)[5:-5, 5:-5] - mu2_sq
        sigma12 = cv2.filter2D(original * denoised, -1, window)[5:-5, 5:-5] - mu1_mu2
        
        ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
        return ssim_map.mean()


class MetricsEvaluator:
    """Calculate and compare image quality metrics"""
    
    def __init__(self, config_path='config.yaml'):
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
    
    def calculate_psnr(self, original, denoised):
        """
        Calculate Peak Signal-to-Noise Ratio (PSNR)
        Higher is better (typically > 30 dB is good)
        """
        if original.shape != denoised.shape:
            raise ValueError("Images must have the same dimensions")
        
        # Convert to float for accurate computation
        original = original.astype(np.float64)
        denoised = denoised.astype(np.float64)
        
        mse = np.mean((original - denoised) ** 2)
        
        if mse == 0:
            return float('inf')
        
        max_pixel = 255.0
        psnr_value = 20 * np.log10(max_pixel / np.sqrt(mse))
        
        return psnr_value
    
    def calculate_ssim(self, original, denoised):
        """
        Calculate Structural Similarity Index (SSIM)
        Range: [-1, 1], higher is better (1 means identical)
        """
        if original.shape != denoised.shape:
            raise ValueError("Images must have the same dimensions")
        
        # Handle grayscale and color images
        if len(original.shape) == 3 and original.shape[2] == 3:
            # For color images, calculate SSIM for each channel and average
            ssim_values = []
            for i in range(3):
                ssim_val = _ssim(
                    original[:, :, i], 
                    denoised[:, :, i],
                    data_range=255
                )
                ssim_values.append(ssim_val)
            return np.mean(ssim_values)
        else:
            # For grayscale images
            return _ssim(original, denoised, data_range=255)
    
    def calculate_niqe(self, image):
        """
        Calculate Natural Image Quality Evaluator (NIQE)
        No-reference metric - lower is better
        """
        try:
            import pyiqa
            
            # Initialize NIQE metric
            niqe_metric = pyiqa.create_metric('niqe', device='cpu')
            
            # Convert to tensor format expected by pyiqa
            if len(image.shape) == 2:
                image = np.stack([image, image, image], axis=2)
            
            # Normalize to [0, 1]
            image_normalized = image.astype(np.float32) / 255.0
            
            # Calculate NIQE
            import torch
            image_tensor = torch.from_numpy(image_normalized).permute(2, 0, 1).unsqueeze(0)
            niqe_score = niqe_metric(image_tensor).item()
            
            return niqe_score
            
        except Exception as e:
            print(f"NIQE calculation failed: {e}")
            # Fallback to simple sharpness metric
            return self._calculate_sharpness(image)
    
    def _calculate_sharpness(self, image):
        """
        Fallback sharpness metric using Laplacian variance
        Higher variance indicates sharper image
        """
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            gray = image
        
        laplacian = cv2.Laplacian(gray, cv2.CV_64F)
        sharpness = laplacian.var()
        
        # Return inverse to match NIQE (lower is better)
        return 100.0 / (sharpness + 1.0)
    
    def calculate_mse(self, original, denoised):
        """Calculate Mean Squared Error"""
        if original.shape != denoised.shape:
            raise ValueError("Images must have the same dimensions")
        
        mse = np.mean((original.astype(np.float64) - denoised.astype(np.float64)) ** 2)
        return mse
    
    def calculate_mae(self, original, denoised):
        """Calculate Mean Absolute Error"""
        if original.shape != denoised.shape:
            raise ValueError("Images must have the same dimensions")
        
        mae = np.mean(np.abs(original.astype(np.float64) - denoised.astype(np.float64)))
        return mae
    
    def evaluate_all_metrics(self, original, denoised, include_niqe=True):
        """
        Calculate all metrics for a denoised image
        
        Returns:
            Dictionary with all metric values
        """
        metrics = {
            'psnr': self.calculate_psnr(original, denoised),
            'ssim': self.calculate_ssim(original, denoised),
            'mse': self.calculate_mse(original, denoised),
            'mae': self.calculate_mae(original, denoised)
        }
        
        if include_niqe:
            metrics['niqe'] = self.calculate_niqe(denoised)
        
        return metrics
    
    def compare_methods(self, original, noisy, denoised_dict):
        """
        Compare multiple denoising methods
        
        Args:
            original: Original clean image
            noisy: Noisy image
            denoised_dict: Dictionary of {method_name: denoised_image}
        
        Returns:
            Dictionary of results
        """
        results = {}
        
        # Metrics for noisy image
        results['noisy'] = self.evaluate_all_metrics(original, noisy)
        
        # Metrics for each denoising method
        for method_name, denoised_img in denoised_dict.items():
            results[method_name] = self.evaluate_all_metrics(original, denoised_img)
        
        return results
    
    def get_best_method(self, results, metric='psnr'):
        """
        Determine the best denoising method based on a specific metric
        
        Args:
            results: Results dictionary from compare_methods
            metric: Metric to use for comparison ('psnr', 'ssim', 'niqe')
        
        Returns:
            Name of the best method
        """
        method_scores = {}
        
        for method_name, metrics in results.items():
            if method_name == 'noisy':
                continue
            method_scores[method_name] = metrics[metric]
        
        # For NIQE, lower is better; for PSNR and SSIM, higher is better
        if metric == 'niqe':
            best_method = min(method_scores.items(), key=lambda x: x[1])
        else:
            best_method = max(method_scores.items(), key=lambda x: x[1])
        
        return best_method[0], best_method[1]
    
    def format_results(self, results):
        """
        Format results for display
        
        Returns:
            Formatted string
        """
        output = []
        output.append("\n" + "="*60)
        output.append("IMAGE QUALITY METRICS COMPARISON")
        output.append("="*60)
        
        for method_name, metrics in results.items():
            output.append(f"\n{method_name.upper()}:")
            output.append(f"  PSNR:  {metrics['psnr']:7.2f} dB")
            output.append(f"  SSIM:  {metrics['ssim']:7.4f}")
            if 'niqe' in metrics:
                output.append(f"  NIQE:  {metrics['niqe']:7.2f}")
            output.append(f"  MSE:   {metrics['mse']:7.2f}")
            output.append(f"  MAE:   {metrics['mae']:7.2f}")
        
        output.append("\n" + "="*60)
        
        # Find best methods
        if len(results) > 1:
            output.append("\nBEST METHODS:")
            try:
                best_psnr = self.get_best_method(results, 'psnr')
                output.append(f"  PSNR: {best_psnr[0]} ({best_psnr[1]:.2f} dB)")
            except:
                pass
            
            try:
                best_ssim = self.get_best_method(results, 'ssim')
                output.append(f"  SSIM: {best_ssim[0]} ({best_ssim[1]:.4f})")
            except:
                pass
            
            try:
                best_niqe = self.get_best_method(results, 'niqe')
                output.append(f"  NIQE: {best_niqe[0]} ({best_niqe[1]:.2f})")
            except:
                pass
        
        output.append("="*60)
        
        return "\n".join(output)


if __name__ == "__main__":
    # Test metrics
    evaluator = MetricsEvaluator()
    
    # Create test images
    original = np.random.randint(0, 256, (256, 256, 3), dtype=np.uint8)
    noisy = np.clip(original + np.random.randn(*original.shape) * 25, 0, 255).astype(np.uint8)
    denoised1 = np.clip(original + np.random.randn(*original.shape) * 10, 0, 255).astype(np.uint8)
    denoised2 = np.clip(original + np.random.randn(*original.shape) * 5, 0, 255).astype(np.uint8)
    
    # Test individual metrics
    print("Testing individual metrics:")
    print(f"PSNR: {evaluator.calculate_psnr(original, denoised1):.2f} dB")
    print(f"SSIM: {evaluator.calculate_ssim(original, denoised1):.4f}")
    
    # Test comparison
    denoised_dict = {
        'method1': denoised1,
        'method2': denoised2
    }
    
    results = evaluator.compare_methods(original, noisy, denoised_dict)
    print(evaluator.format_results(results))
