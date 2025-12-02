"""
Advanced denoising methods: Hybrid and Noise Detection
"""
import numpy as np
import cv2
from scipy import stats
import yaml


class NoiseDetector:
    """Automatic noise type detection"""
    
    def __init__(self):
        pass
    
    def detect_noise_type(self, image):
        """
        Detect the predominant noise type in an image
        
        Returns:
            noise_type: 'gaussian', 'salt_pepper', 'speckle', or 'unknown'
            confidence: confidence score [0, 1]
        """
        # Calculate various statistical features
        features = self._extract_noise_features(image)
        
        # Score each noise type
        gaussian_score = self._score_gaussian(features)
        salt_pepper_score = self._score_salt_pepper(features)
        speckle_score = self._score_speckle(features)
        
        scores = {
            'gaussian': gaussian_score,
            'salt_pepper': salt_pepper_score,
            'speckle': speckle_score
        }
        
        # Get the maximum score
        best_noise = max(scores.items(), key=lambda x: x[1])
        
        noise_type = best_noise[0]
        confidence = best_noise[1]
        
        # If confidence is too low, return unknown
        if confidence < 0.3:
            return 'unknown', confidence
        
        return noise_type, confidence
    
    def _extract_noise_features(self, image):
        """Extract statistical features for noise detection"""
        # Convert to grayscale if needed
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            gray = image
        
        gray_float = gray.astype(np.float32)
        
        # Calculate noise estimation using high-pass filter
        kernel = np.array([[-1, -1, -1],
                          [-1,  8, -1],
                          [-1, -1, -1]])
        high_pass = cv2.filter2D(gray_float, -1, kernel)
        
        features = {
            'std': np.std(high_pass),
            'variance': np.var(high_pass),
            'kurtosis': stats.kurtosis(high_pass.flatten()),
            'skewness': stats.skew(high_pass.flatten()),
            'num_outliers': self._count_outliers(gray),
            'edge_strength': self._calculate_edge_strength(gray)
        }
        
        return features
    
    def _count_outliers(self, image):
        """Count the number of outlier pixels (potential salt & pepper)"""
        # Calculate median and MAD
        median = np.median(image)
        mad = np.median(np.abs(image - median))
        
        # Count pixels far from median
        threshold = 3 * mad
        outliers = np.sum(np.abs(image - median) > threshold)
        
        return outliers / image.size
    
    def _calculate_edge_strength(self, image):
        """Calculate edge strength using Sobel"""
        sobelx = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=3)
        sobely = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=3)
        edge_magnitude = np.sqrt(sobelx**2 + sobely**2)
        
        return np.mean(edge_magnitude)
    
    def _score_gaussian(self, features):
        """Score likelihood of Gaussian noise"""
        score = 0.0
        
        # Gaussian noise has moderate kurtosis (around 3)
        kurtosis_diff = abs(features['kurtosis'] - 3.0)
        if kurtosis_diff < 1.0:
            score += 0.4
        elif kurtosis_diff < 2.0:
            score += 0.2
        
        # Gaussian noise has near-zero skewness
        if abs(features['skewness']) < 0.5:
            score += 0.3
        
        # Low outlier count
        if features['num_outliers'] < 0.05:
            score += 0.3
        
        return score
    
    def _score_salt_pepper(self, features):
        """Score likelihood of salt & pepper noise"""
        score = 0.0
        
        # High kurtosis (peaky distribution)
        if features['kurtosis'] > 5.0:
            score += 0.4
        elif features['kurtosis'] > 3.0:
            score += 0.2
        
        # High outlier count
        if features['num_outliers'] > 0.1:
            score += 0.5
        elif features['num_outliers'] > 0.05:
            score += 0.2
        
        # Low edge strength (corrupted edges)
        if features['edge_strength'] < 20:
            score += 0.1
        
        return score
    
    def _score_speckle(self, features):
        """Score likelihood of speckle noise"""
        score = 0.0
        
        # Moderate to high variance
        if features['variance'] > 100:
            score += 0.3
        
        # Positive skewness typical for speckle
        if features['skewness'] > 0.5:
            score += 0.4
        elif features['skewness'] > 0:
            score += 0.2
        
        # Moderate outliers
        if 0.03 < features['num_outliers'] < 0.08:
            score += 0.3
        
        return score


class HybridDenoiser:
    """Hybrid denoising combining classical and deep learning methods"""
    
    def __init__(self, classical_denoiser, dncnn_denoiser, noise_detector=None):
        self.classical = classical_denoiser
        self.dncnn = dncnn_denoiser
        self.dncnn_available = dncnn_denoiser is not None
        self.noise_detector = noise_detector or NoiseDetector()
    
    def denoise_adaptive(self, noisy_image, detect_noise=True):
        """
        Adaptive denoising based on noise type detection
        
        Args:
            noisy_image: Noisy input image
            detect_noise: Whether to detect noise type automatically
        
        Returns:
            denoised_image: Denoised result
            method_used: Name of the method used
        """
        if detect_noise:
            noise_type, confidence = self.noise_detector.detect_noise_type(noisy_image)
            print(f"Detected noise type: {noise_type} (confidence: {confidence:.2f})")
        else:
            noise_type = 'unknown'
        
        # Choose best method based on noise type
        if noise_type == 'salt_pepper':
            denoised = self.classical.denoise_image(noisy_image, method='median', kernel_size=5)
            method_used = 'median_filter'
        elif noise_type == 'gaussian':
            denoised = self.classical.denoise_image(noisy_image, method='wavelet')
            method_used = 'wavelet'
        elif noise_type == 'speckle':
            denoised = self.classical.denoise_image(noisy_image, method='wiener')
            method_used = 'wiener'
        else:
            # Use DnCNN for unknown noise if available, otherwise wavelet
            if self.dncnn_available:
                denoised = self.dncnn.denoise(noisy_image)
                method_used = 'dncnn'
            else:
                denoised = self.classical.denoise_image(noisy_image, method='wavelet')
                method_used = 'wavelet_fallback'
        
        return denoised, method_used
    
    def denoise_hybrid_cascade(self, noisy_image):
        """
        Cascade hybrid: Classical preprocessing + DnCNN refinement
        Falls back to classical-only if DnCNN is not available
        """
        # First stage: Classical denoising (wavelet)
        stage1 = self.classical.denoise_image(noisy_image, method='wavelet')
        
        # Second stage: DnCNN refinement if available
        if self.dncnn_available:
            stage2 = self.dncnn.denoise(stage1)
            return stage2
        else:
            # Fallback: apply median filter as second stage
            stage2 = self.classical.denoise_image(stage1, method='median', kernel_size=3)
            return stage2
    
    def denoise_hybrid_average(self, noisy_image, weights=None):
        """
        Weighted average of multiple denoising methods
        
        Args:
            noisy_image: Noisy input
            weights: Dictionary of {method: weight}, defaults to equal weights
        """
        if weights is None:
            if self.dncnn_available:
                weights = {
                    'median': 0.2,
                    'wavelet': 0.3,
                    'wiener': 0.2,
                    'dncnn': 0.3
                }
            else:
                # Equal weights for classical methods only
                weights = {
                    'median': 0.33,
                    'wavelet': 0.34,
                    'wiener': 0.33
                }
        
        # Denoise with each method
        results = {}
        results['median'] = self.classical.denoise_image(noisy_image, method='median').astype(np.float32)
        results['wavelet'] = self.classical.denoise_image(noisy_image, method='wavelet').astype(np.float32)
        results['wiener'] = self.classical.denoise_image(noisy_image, method='wiener').astype(np.float32)
        
        if self.dncnn_available:
            results['dncnn'] = self.dncnn.denoise(noisy_image).astype(np.float32)
        
        # Weighted average
        hybrid = np.zeros_like(results['median'])
        total_weight = 0
        
        for method, weight in weights.items():
            if method in results:
                hybrid += results[method] * weight
                total_weight += weight
        
        hybrid /= total_weight
        
        return np.clip(hybrid, 0, 255).astype(np.uint8)
    
    def denoise_hybrid_selective(self, noisy_image, block_size=32):
        """
        Block-wise selective denoising
        Choose best method for each image block
        """
        h, w = noisy_image.shape[:2]
        denoised = np.zeros_like(noisy_image)
        
        # Process image in blocks
        for i in range(0, h, block_size):
            for j in range(0, w, block_size):
                # Extract block
                block = noisy_image[i:i+block_size, j:j+block_size]
                
                # Detect noise in block
                noise_type, _ = self.noise_detector.detect_noise_type(block)
                
                # Denoise block with appropriate method
                if noise_type == 'salt_pepper':
                    denoised_block = self.classical.denoise_image(block, method='median')
                elif noise_type == 'gaussian':
                    denoised_block = self.classical.denoise_image(block, method='wavelet')
                else:
                    # Use DnCNN if available, otherwise wavelet
                    if self.dncnn_available:
                        denoised_block = self.dncnn.denoise(block)
                    else:
                        denoised_block = self.classical.denoise_image(block, method='wavelet')
                
                # Place denoised block back
                denoised[i:i+block_size, j:j+block_size] = denoised_block
        
        return denoised


class MethodRecommender:
    """Recommend best denoising method based on image and noise characteristics"""
    
    def __init__(self):
        self.noise_detector = NoiseDetector()
    
    def recommend_method(self, noisy_image):
        """
        Recommend the best denoising method
        
        Returns:
            recommended_method: Method name
            reason: Explanation
        """
        # Detect noise type
        noise_type, confidence = self.noise_detector.detect_noise_type(noisy_image)
        
        recommendations = {
            'gaussian': {
                'method': 'wavelet',
                'reason': 'Wavelet transform effectively handles Gaussian noise while preserving edges'
            },
            'salt_pepper': {
                'method': 'median',
                'reason': 'Median filter is optimal for salt & pepper noise removal'
            },
            'speckle': {
                'method': 'wiener',
                'reason': 'Wiener filter effectively reduces speckle noise'
            },
            'unknown': {
                'method': 'dncnn',
                'reason': 'DnCNN deep learning model handles multiple noise types effectively'
            }
        }
        
        recommendation = recommendations.get(noise_type, recommendations['unknown'])
        
        return recommendation['method'], recommendation['reason'], noise_type, confidence


if __name__ == "__main__":
    # Test noise detection
    print("Testing Noise Detection...")
    
    # Create test images with different noise types
    clean = np.random.randint(100, 156, (256, 256, 3), dtype=np.uint8)
    
    # Gaussian noise
    gaussian_noisy = np.clip(clean + np.random.randn(*clean.shape) * 25, 0, 255).astype(np.uint8)
    
    detector = NoiseDetector()
    noise_type, confidence = detector.detect_noise_type(gaussian_noisy)
    print(f"Gaussian noisy image detected as: {noise_type} (confidence: {confidence:.2f})")
    
    # Test recommender
    recommender = MethodRecommender()
    method, reason, detected_noise, conf = recommender.recommend_method(gaussian_noisy)
    print(f"\nRecommended method: {method}")
    print(f"Reason: {reason}")
    print(f"Detected noise: {detected_noise} (confidence: {conf:.2f})")
