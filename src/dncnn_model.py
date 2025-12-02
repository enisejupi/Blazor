"""
DnCNN (Denoising Convolutional Neural Network) implementation
"""
import numpy as np
import os
from tqdm import tqdm
import yaml

# Try to import PyTorch - it's optional
try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import Dataset, DataLoader
    PYTORCH_AVAILABLE = True
except ImportError:
    PYTORCH_AVAILABLE = False
    print("⚠️  Warning: PyTorch not installed. DNN-based denoising will be unavailable.")
    print("   To enable DNN features, install PyTorch or use Python 3.11/3.12.")
    # Create stub classes to avoid errors
    class nn:
        class Module:
            def __init__(self):
                pass
        class Conv2d:
            def __init__(self, *args, **kwargs):
                pass
        class ReLU:
            def __init__(self, *args, **kwargs):
                pass
        class BatchNorm2d:
            def __init__(self, *args, **kwargs):
                pass
        class Sequential:
            def __init__(self, *args, **kwargs):
                pass
        class MSELoss:
            def __init__(self, *args, **kwargs):
                pass
    class optim:
        class Adam:
            def __init__(self, *args, **kwargs):
                pass
    class Dataset:
        def __init__(self):
            pass
    class DataLoader:
        def __init__(self, *args, **kwargs):
            pass
    class torch:
        @staticmethod
        def device(x):
            return None
        @staticmethod
        def cuda(*args, **kwargs):
            class Cuda:
                @staticmethod
                def is_available():
                    return False
            return Cuda()
        @staticmethod
        def from_numpy(*args, **kwargs):
            raise ImportError("PyTorch is not installed")
        @staticmethod
        def load(*args, **kwargs):
            raise ImportError("PyTorch is not installed")
        @staticmethod
        def save(*args, **kwargs):
            raise ImportError("PyTorch is not installed")


class DnCNN(nn.Module):
    """
    DnCNN architecture for image denoising
    Based on: "Beyond a Gaussian Denoiser: Residual Learning of Deep CNN for Image Denoising"
    """
    
    def __init__(self, depth=17, channels=64, kernel_size=3, num_channels=3):
        if not PYTORCH_AVAILABLE:
            raise ImportError("PyTorch is required for DnCNN but is not installed. "
                            "Install PyTorch or use Python 3.11/3.12 for pre-built wheels.")
        super(DnCNN, self).__init__()
        
        layers = []
        
        # First layer
        layers.append(nn.Conv2d(num_channels, channels, kernel_size, padding=kernel_size//2, bias=False))
        layers.append(nn.ReLU(inplace=True))
        
        # Middle layers
        for _ in range(depth - 2):
            layers.append(nn.Conv2d(channels, channels, kernel_size, padding=kernel_size//2, bias=False))
            layers.append(nn.BatchNorm2d(channels))
            layers.append(nn.ReLU(inplace=True))
        
        # Last layer
        layers.append(nn.Conv2d(channels, num_channels, kernel_size, padding=kernel_size//2, bias=False))
        
        self.dncnn = nn.Sequential(*layers)
    
    def forward(self, x):
        # Predict the noise (residual learning)
        noise = self.dncnn(x)
        # Return denoised image
        return x - noise


class DenoisingDataset(Dataset):
    """Dataset for training DnCNN"""
    
    def __init__(self, clean_images, noisy_images):
        if not PYTORCH_AVAILABLE:
            raise ImportError("PyTorch is required for DenoisingDataset but is not installed.")
        super().__init__()
        self.clean_images = clean_images
        self.noisy_images = noisy_images
    
    def __len__(self):
        return len(self.clean_images)
    
    def __getitem__(self, idx):
        clean = torch.from_numpy(self.clean_images[idx]).permute(2, 0, 1).float() / 255.0
        noisy = torch.from_numpy(self.noisy_images[idx]).permute(2, 0, 1).float() / 255.0
        return noisy, clean


class DnCNNTrainer:
    """Training pipeline for DnCNN"""
    
    def __init__(self, config_path='config.yaml'):
        if not PYTORCH_AVAILABLE:
            print("⚠️  DnCNNTrainer initialized but PyTorch is not available.")
            self.model = None
            self.optimizer = None
            self.criterion = None
            self.device = None
            return
            
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")
        
        # Model parameters
        self.depth = self.config['dncnn']['depth']
        self.channels = self.config['dncnn']['channels']
        self.kernel_size = self.config['dncnn']['kernel_size']
        
        # Training parameters
        self.batch_size = self.config['dncnn']['batch_size']
        self.epochs = self.config['dncnn']['epochs']
        self.learning_rate = self.config['dncnn']['learning_rate']
        
        # Initialize model
        self.model = None
        self.optimizer = None
        self.criterion = nn.MSELoss()
    
    def create_model(self, num_channels=3):
        """Create DnCNN model"""
        if not PYTORCH_AVAILABLE:
            print("❌ Cannot create model: PyTorch is not installed.")
            return None
            
        self.model = DnCNN(
            depth=self.depth,
            channels=self.channels,
            kernel_size=self.kernel_size,
            num_channels=num_channels
        ).to(self.device)
        
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        
        return self.model
    
    def train(self, train_loader, val_loader=None, save_path='models/dncnn.pth'):
        """Train the DnCNN model"""
        if not PYTORCH_AVAILABLE:
            print("❌ Cannot train: PyTorch is not installed.")
            return [], []
            
        if self.model is None:
            raise ValueError("Model not created. Call create_model() first.")
        
        best_loss = float('inf')
        training_losses = []
        validation_losses = []
        
        print(f"\nTraining DnCNN for {self.epochs} epochs...")
        
        for epoch in range(self.epochs):
            # Training phase
            self.model.train()
            train_loss = 0.0
            
            for noisy, clean in tqdm(train_loader, desc=f"Epoch {epoch+1}/{self.epochs}"):
                noisy = noisy.to(self.device)
                clean = clean.to(self.device)
                
                # Forward pass
                self.optimizer.zero_grad()
                denoised = self.model(noisy)
                loss = self.criterion(denoised, clean)
                
                # Backward pass
                loss.backward()
                self.optimizer.step()
                
                train_loss += loss.item()
            
            avg_train_loss = train_loss / len(train_loader)
            training_losses.append(avg_train_loss)
            
            # Validation phase
            if val_loader is not None:
                self.model.eval()
                val_loss = 0.0
                
                with torch.no_grad():
                    for noisy, clean in val_loader:
                        noisy = noisy.to(self.device)
                        clean = clean.to(self.device)
                        
                        denoised = self.model(noisy)
                        loss = self.criterion(denoised, clean)
                        val_loss += loss.item()
                
                avg_val_loss = val_loss / len(val_loader)
                validation_losses.append(avg_val_loss)
                
                print(f"Epoch {epoch+1}: Train Loss = {avg_train_loss:.6f}, Val Loss = {avg_val_loss:.6f}")
                
                # Save best model
                if avg_val_loss < best_loss:
                    best_loss = avg_val_loss
                    os.makedirs(os.path.dirname(save_path), exist_ok=True)
                    torch.save({
                        'epoch': epoch,
                        'model_state_dict': self.model.state_dict(),
                        'optimizer_state_dict': self.optimizer.state_dict(),
                        'loss': best_loss,
                    }, save_path)
                    print(f"  -> Model saved with validation loss: {best_loss:.6f}")
            else:
                print(f"Epoch {epoch+1}: Train Loss = {avg_train_loss:.6f}")
                
                # Save model every 10 epochs
                if (epoch + 1) % 10 == 0:
                    os.makedirs(os.path.dirname(save_path), exist_ok=True)
                    torch.save({
                        'epoch': epoch,
                        'model_state_dict': self.model.state_dict(),
                        'optimizer_state_dict': self.optimizer.state_dict(),
                        'loss': avg_train_loss,
                    }, save_path)
                    print(f"  -> Model saved at epoch {epoch+1}")
        
        return training_losses, validation_losses
    
    def load_model(self, model_path, num_channels=3):
        """Load a trained model"""
        if not PYTORCH_AVAILABLE:
            print("❌ Cannot load model: PyTorch is not installed.")
            return False
            
        self.create_model(num_channels)
        
        if os.path.exists(model_path):
            checkpoint = torch.load(model_path, map_location=self.device)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            print(f"Model loaded from {model_path}")
            return True
        else:
            print(f"Model file not found: {model_path}")
            return False
    
    def denoise(self, noisy_image):
        """Denoise a single image"""
        if not PYTORCH_AVAILABLE:
            raise ImportError("Cannot denoise: PyTorch is not installed. Use classical methods instead.")
            
        if self.model is None:
            raise ValueError("Model not loaded. Call load_model() or train a model first.")
        
        self.model.eval()
        
        # Prepare input
        if len(noisy_image.shape) == 2:
            noisy_image = np.expand_dims(noisy_image, axis=2)
        
        input_tensor = torch.from_numpy(noisy_image).permute(2, 0, 1).unsqueeze(0).float() / 255.0
        input_tensor = input_tensor.to(self.device)
        
        # Denoise
        with torch.no_grad():
            denoised_tensor = self.model(input_tensor)
        
        # Convert back to numpy
        denoised = denoised_tensor.squeeze(0).permute(1, 2, 0).cpu().numpy()
        denoised = (denoised * 255.0).clip(0, 255).astype(np.uint8)
        
        if denoised.shape[2] == 1:
            denoised = denoised.squeeze(2)
        
        return denoised


def create_training_data(clean_images, noise_generator, num_samples=400):
    """Create training dataset with augmentation"""
    if not PYTORCH_AVAILABLE:
        print("❌ Cannot create training data: PyTorch is not installed.")
        return None, None
        
    from data_loader import NoiseGenerator
    
    training_clean = []
    training_noisy = []
    
    print(f"Creating {num_samples} training samples...")
    
    noise_gen = NoiseGenerator()
    
    for _ in range(num_samples // len(clean_images) + 1):
        for img in clean_images:
            if len(training_clean) >= num_samples:
                break
            
            # Random crop if image is large
            h, w = img.shape[:2]
            if h > 128 or w > 128:
                crop_size = 128
                top = np.random.randint(0, max(1, h - crop_size))
                left = np.random.randint(0, max(1, w - crop_size))
                img_crop = img[top:top+crop_size, left:left+crop_size]
            else:
                img_crop = img
            
            # Add random noise type
            noise_type = np.random.choice(['gaussian', 'salt_pepper', 'speckle'])
            
            if noise_type == 'gaussian':
                sigma = np.random.uniform(15, 35)
                noisy = noise_gen.add_gaussian_noise(img_crop, sigma)
            elif noise_type == 'salt_pepper':
                amount = np.random.uniform(0.02, 0.1)
                noisy = noise_gen.add_salt_pepper_noise(img_crop, amount)
            else:
                variance = np.random.uniform(0.05, 0.15)
                noisy = noise_gen.add_speckle_noise(img_crop, variance)
            
            training_clean.append(img_crop)
            training_noisy.append(noisy)
        
        if len(training_clean) >= num_samples:
            break
    
    return np.array(training_clean[:num_samples]), np.array(training_noisy[:num_samples])


if __name__ == "__main__":
    # Test DnCNN
    print("Testing DnCNN implementation...")
    
    if not PYTORCH_AVAILABLE:
        print("❌ Cannot test: PyTorch is not installed.")
        exit(1)
    
    # Create dummy data
    dummy_clean = np.random.randint(0, 256, (10, 128, 128, 3), dtype=np.uint8)
    dummy_noisy = np.clip(dummy_clean + np.random.randn(10, 128, 128, 3) * 25, 0, 255).astype(np.uint8)
    
    # Create dataset and dataloader
    dataset = DenoisingDataset(dummy_clean, dummy_noisy)
    dataloader = DataLoader(dataset, batch_size=2, shuffle=True)
    
    # Create and train model for 2 epochs (just for testing)
    trainer = DnCNNTrainer()
    trainer.create_model()
    
    print("Model created successfully!")
    print(f"Model parameters: {sum(p.numel() for p in trainer.model.parameters()):,}")
