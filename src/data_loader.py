"""
Image downloader and dataset preparation module
"""
import os
import requests
import numpy as np
import cv2
from PIL import Image
from io import BytesIO
from tqdm import tqdm
import yaml


class ImageDownloader:
    """Download and prepare images for denoising experiments"""
    
    def __init__(self, config_path='config.yaml'):
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.download_dir = self.config['dataset']['download_dir']
        os.makedirs(self.download_dir, exist_ok=True)
        
        # Backup URLs for diverse test images
        self.image_urls = [
            "https://images.pexels.com/photos/417074/pexels-photo-417074.jpeg?auto=compress&cs=tinysrgb&w=600",  # Mountain landscape
            "https://images.pexels.com/photos/414612/pexels-photo-414612.jpeg?auto=compress&cs=tinysrgb&w=600",  # Nature
            "https://images.pexels.com/photos/415829/pexels-photo-415829.jpeg?auto=compress&cs=tinysrgb&w=600",  # Desert
            "https://images.pexels.com/photos/1166209/pexels-photo-1166209.jpeg?auto=compress&cs=tinysrgb&w=600",  # Portrait
            "https://images.pexels.com/photos/1108099/pexels-photo-1108099.jpeg?auto=compress&cs=tinysrgb&w=600",  # Cat
            "https://images.pexels.com/photos/1851164/pexels-photo-1851164.jpeg?auto=compress&cs=tinysrgb&w=600",  # Dog
            "https://images.pexels.com/photos/157811/pexels-photo-157811.jpeg?auto=compress&cs=tinysrgb&w=600",  # House
            "https://images.pexels.com/photos/325185/pexels-photo-325185.jpeg?auto=compress&cs=tinysrgb&w=600",  # Beach
            "https://images.pexels.com/photos/1149137/pexels-photo-1149137.jpeg?auto=compress&cs=tinysrgb&w=600",  # Car
            "https://images.pexels.com/photos/1640777/pexels-photo-1640777.jpeg?auto=compress&cs=tinysrgb&w=600",  # Coffee
            "https://images.pexels.com/photos/1477166/pexels-photo-1477166.jpeg?auto=compress&cs=tinysrgb&w=600",  # Flower
            "https://images.pexels.com/photos/1323550/pexels-photo-1323550.jpeg?auto=compress&cs=tinysrgb&w=600",  # Food
            "https://images.pexels.com/photos/1779487/pexels-photo-1779487.jpeg?auto=compress&cs=tinysrgb&w=600",  # Building
            "https://images.pexels.com/photos/3374210/pexels-photo-3374210.jpeg?auto=compress&cs=tinysrgb&w=600",  # Technology
            "https://images.pexels.com/photos/235994/pexels-photo-235994.jpeg?auto=compress&cs=tinysrgb&w=600",  # Abstract
        ]
        
        self.image_size = tuple(self.config['dataset']['image_size'])
    
    def download_images(self, num_images=None):
        """Download sample images from the internet"""
        if num_images is None:
            num_images = self.config['dataset']['num_images']
        
        print(f"Downloading {num_images} sample images...")
        
        downloaded_count = 0
        for idx, url in enumerate(tqdm(self.image_urls[:num_images])):
            try:
                # Download image
                response = requests.get(url, timeout=10, headers={
                    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
                })
                response.raise_for_status()
                
                # Open and process image
                img = Image.open(BytesIO(response.content))
                img = img.convert('RGB')
                
                # Resize to consistent size
                img = img.resize(self.image_size, Image.Resampling.LANCZOS)
                
                # Save image
                output_path = os.path.join(self.download_dir, f"image_{idx:03d}.png")
                img.save(output_path)
                
                downloaded_count += 1
                
            except Exception as e:
                print(f"Failed to download image {idx}: {e}")
                # Create a synthetic image as fallback
                self._create_synthetic_image(idx)
                downloaded_count += 1
        
        print(f"Successfully downloaded/created {downloaded_count} images")
        return downloaded_count
    
    def _create_synthetic_image(self, idx):
        """Create a synthetic test image as fallback"""
        # Create diverse synthetic images
        img = np.zeros((self.image_size[1], self.image_size[0], 3), dtype=np.uint8)
        
        # Different patterns for different indices
        pattern = idx % 5
        
        if pattern == 0:
            # Gradient
            for i in range(self.image_size[1]):
                img[i, :] = [i * 255 // self.image_size[1], 128, 255 - i * 255 // self.image_size[1]]
        elif pattern == 1:
            # Checkerboard
            square_size = 32
            for i in range(0, self.image_size[1], square_size):
                for j in range(0, self.image_size[0], square_size):
                    if ((i // square_size) + (j // square_size)) % 2 == 0:
                        img[i:i+square_size, j:j+square_size] = [200, 200, 200]
        elif pattern == 2:
            # Circles
            center = (self.image_size[0] // 2, self.image_size[1] // 2)
            for radius in range(20, min(self.image_size) // 2, 30):
                cv2.circle(img, center, radius, (255, 100, 50), 2)
        elif pattern == 3:
            # Random noise pattern
            img = np.random.randint(0, 256, (self.image_size[1], self.image_size[0], 3), dtype=np.uint8)
        else:
            # Solid colors with shapes
            img[:] = [100, 150, 200]
            cv2.rectangle(img, (50, 50), (200, 200), (255, 255, 0), -1)
        
        output_path = os.path.join(self.download_dir, f"image_{idx:03d}.png")
        cv2.imwrite(output_path, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
    
    def load_images(self):
        """Load all images from download directory"""
        image_files = sorted([f for f in os.listdir(self.download_dir) if f.endswith(('.png', '.jpg', '.jpeg'))])
        
        images = []
        for img_file in image_files:
            img_path = os.path.join(self.download_dir, img_file)
            img = cv2.imread(img_path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            images.append(img)
        
        print(f"Loaded {len(images)} images")
        return images, image_files


class NoiseGenerator:
    """Generate different types of noise for testing"""
    
    def __init__(self, config_path='config.yaml'):
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
    
    def add_gaussian_noise(self, image, sigma=25):
        """Add Gaussian noise to image"""
        noise = np.random.normal(0, sigma, image.shape)
        noisy_image = image.astype(np.float32) + noise
        noisy_image = np.clip(noisy_image, 0, 255).astype(np.uint8)
        return noisy_image
    
    def add_salt_pepper_noise(self, image, amount=0.05):
        """Add salt and pepper noise to image"""
        noisy_image = image.copy()
        
        # Salt noise (white pixels)
        num_salt = int(amount * image.size * 0.5)
        coords = [np.random.randint(0, i - 1, num_salt) for i in image.shape[:2]]
        noisy_image[coords[0], coords[1], :] = 255
        
        # Pepper noise (black pixels)
        num_pepper = int(amount * image.size * 0.5)
        coords = [np.random.randint(0, i - 1, num_pepper) for i in image.shape[:2]]
        noisy_image[coords[0], coords[1], :] = 0
        
        return noisy_image
    
    def add_speckle_noise(self, image, variance=0.1):
        """Add speckle (multiplicative) noise to image"""
        noise = np.random.randn(*image.shape) * variance
        noisy_image = image.astype(np.float32) * (1 + noise)
        noisy_image = np.clip(noisy_image, 0, 255).astype(np.uint8)
        return noisy_image
    
    def generate_noisy_dataset(self, images):
        """Generate noisy versions of all images with different noise types"""
        noisy_dataset = {}
        
        for noise_type in self.config['noise']['types']:
            noisy_dataset[noise_type] = []
            
            if noise_type == 'gaussian':
                for img in images:
                    sigma = self.config['noise']['gaussian']['sigma'][1]  # Use middle sigma value
                    noisy_img = self.add_gaussian_noise(img, sigma)
                    noisy_dataset[noise_type].append(noisy_img)
            
            elif noise_type == 'salt_pepper':
                for img in images:
                    amount = self.config['noise']['salt_pepper']['amount'][1]
                    noisy_img = self.add_salt_pepper_noise(img, amount)
                    noisy_dataset[noise_type].append(noisy_img)
            
            elif noise_type == 'speckle':
                for img in images:
                    variance = self.config['noise']['speckle']['variance'][1]
                    noisy_img = self.add_speckle_noise(img, variance)
                    noisy_dataset[noise_type].append(noisy_img)
        
        return noisy_dataset


if __name__ == "__main__":
    # Test the downloader
    downloader = ImageDownloader()
    downloader.download_images()
    
    images, filenames = downloader.load_images()
    print(f"Loaded {len(images)} images: {filenames}")
    
    # Test noise generation
    noise_gen = NoiseGenerator()
    noisy_dataset = noise_gen.generate_noisy_dataset(images[:3])
    
    for noise_type, noisy_imgs in noisy_dataset.items():
        print(f"{noise_type}: {len(noisy_imgs)} noisy images generated")
