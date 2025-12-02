"""
Visualization utilities for denoising results
"""
import numpy as np
import matplotlib.pyplot as plt
import cv2
import os
from matplotlib.gridspec import GridSpec
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots


class Visualizer:
    """Create visualizations for denoising results"""
    
    def __init__(self, output_dir='visualizations'):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        # Set matplotlib style
        try:
            plt.style.use('seaborn-v0_8-darkgrid')
        except OSError:
            # Fallback to default style if seaborn style not available
            plt.style.use('default')
            plt.rcParams['axes.grid'] = True
            plt.rcParams['grid.alpha'] = 0.3
    
    def plot_comparison(self, original, noisy, denoised_dict, save_path=None):
        """
        Plot side-by-side comparison of original, noisy, and denoised images
        
        Args:
            original: Original clean image
            noisy: Noisy image
            denoised_dict: Dictionary of {method_name: denoised_image}
            save_path: Path to save the figure
        """
        num_methods = len(denoised_dict)
        num_cols = min(4, num_methods + 2)
        num_rows = (num_methods + 2 + num_cols - 1) // num_cols
        
        fig, axes = plt.subplots(num_rows, num_cols, figsize=(4*num_cols, 4*num_rows))
        axes = axes.flatten() if num_rows > 1 or num_cols > 1 else [axes]
        
        # Original
        axes[0].imshow(original)
        axes[0].set_title('Original', fontsize=14, fontweight='bold')
        axes[0].axis('off')
        
        # Noisy
        axes[1].imshow(noisy)
        axes[1].set_title('Noisy', fontsize=14, fontweight='bold')
        axes[1].axis('off')
        
        # Denoised versions
        for idx, (method_name, denoised_img) in enumerate(denoised_dict.items(), start=2):
            axes[idx].imshow(denoised_img)
            axes[idx].set_title(method_name.replace('_', ' ').title(), fontsize=14, fontweight='bold')
            axes[idx].axis('off')
        
        # Hide unused subplots
        for idx in range(len(denoised_dict) + 2, len(axes)):
            axes[idx].axis('off')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Comparison saved to {save_path}")
        
        plt.close()
    
    def plot_metrics_comparison(self, results, save_path=None):
        """
        Plot bar chart comparing metrics across methods
        
        Args:
            results: Results dictionary from metrics.compare_methods
            save_path: Path to save the figure
        """
        methods = [m for m in results.keys() if m != 'noisy']
        metrics = ['psnr', 'ssim']
        
        fig, axes = plt.subplots(1, len(metrics), figsize=(6*len(metrics), 5))
        if len(metrics) == 1:
            axes = [axes]
        
        for idx, metric in enumerate(metrics):
            values = [results[method][metric] for method in methods]
            
            # Use matplotlib's tab10 colormap for distinct colors
            colors = plt.cm.tab10(np.linspace(0, 1, len(methods)))
            axes[idx].bar(range(len(methods)), values, color=colors)
            axes[idx].set_xticks(range(len(methods)))
            axes[idx].set_xticklabels([m.replace('_', '\n') for m in methods], rotation=0)
            axes[idx].set_ylabel(metric.upper(), fontsize=12, fontweight='bold')
            axes[idx].set_title(f'{metric.upper()} Comparison', fontsize=14, fontweight='bold')
            axes[idx].grid(True, alpha=0.3)
            
            # Add value labels on bars
            for i, v in enumerate(values):
                axes[idx].text(i, v, f'{v:.2f}', ha='center', va='bottom', fontweight='bold')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Metrics comparison saved to {save_path}")
        
        plt.close()
    
    def plot_noise_comparison(self, original, noise_types_dict, save_path=None):
        """
        Compare different noise types
        
        Args:
            original: Original image
            noise_types_dict: Dictionary of {noise_type: noisy_image}
            save_path: Path to save the figure
        """
        num_images = len(noise_types_dict) + 1
        num_cols = min(4, num_images)
        num_rows = (num_images + num_cols - 1) // num_cols
        
        fig, axes = plt.subplots(num_rows, num_cols, figsize=(4*num_cols, 4*num_rows))
        axes = axes.flatten() if isinstance(axes, np.ndarray) else [axes]
        
        # Original
        axes[0].imshow(original)
        axes[0].set_title('Original', fontsize=14, fontweight='bold')
        axes[0].axis('off')
        
        # Different noise types
        for idx, (noise_type, noisy_img) in enumerate(noise_types_dict.items(), start=1):
            axes[idx].imshow(noisy_img)
            axes[idx].set_title(f'{noise_type.replace("_", " ").title()} Noise', 
                              fontsize=14, fontweight='bold')
            axes[idx].axis('off')
        
        # Hide unused subplots
        for idx in range(num_images, len(axes)):
            axes[idx].axis('off')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Noise comparison saved to {save_path}")
        
        plt.close()
    
    def create_interactive_comparison(self, original, noisy, denoised_dict, metrics_dict, save_path=None):
        """
        Create interactive Plotly visualization
        
        Args:
            original: Original image
            noisy: Noisy image
            denoised_dict: Dictionary of denoised images
            metrics_dict: Dictionary of metrics
            save_path: Path to save HTML file
        """
        num_methods = len(denoised_dict)
        
        # Create subplots
        fig = make_subplots(
            rows=2, cols=num_methods + 2,
            subplot_titles=['Original', 'Noisy'] + [m.replace('_', ' ').title() for m in denoised_dict.keys()],
            specs=[[{'type': 'image'}] * (num_methods + 2),
                   [{'type': 'bar', 'colspan': num_methods + 2}, None] + [None] * num_methods],
            row_heights=[0.7, 0.3],
            vertical_spacing=0.1
        )
        
        # Add images
        fig.add_trace(go.Image(z=original), row=1, col=1)
        fig.add_trace(go.Image(z=noisy), row=1, col=2)
        
        for idx, (method_name, denoised_img) in enumerate(denoised_dict.items(), start=3):
            fig.add_trace(go.Image(z=denoised_img), row=1, col=idx)
        
        # Add metrics bar chart
        methods = list(denoised_dict.keys())
        psnr_values = [metrics_dict[m]['psnr'] for m in methods]
        ssim_values = [metrics_dict[m]['ssim'] for m in methods]
        
        fig.add_trace(
            go.Bar(name='PSNR', x=methods, y=psnr_values, yaxis='y'),
            row=2, col=1
        )
        fig.add_trace(
            go.Bar(name='SSIM', x=methods, y=ssim_values, yaxis='y2'),
            row=2, col=1
        )
        
        # Update layout
        fig.update_layout(
            title_text="Image Denoising Comparison",
            showlegend=True,
            height=800,
            barmode='group'
        )
        
        if save_path:
            fig.write_html(save_path)
            print(f"Interactive visualization saved to {save_path}")
        
        return fig
    
    def plot_metrics_heatmap(self, all_results, save_path=None):
        """
        Create heatmap of metrics across multiple images and methods
        
        Args:
            all_results: List of result dictionaries
            save_path: Path to save the figure
        """
        methods = [m for m in all_results[0].keys() if m != 'noisy']
        metrics = ['psnr', 'ssim']
        
        fig, axes = plt.subplots(1, len(metrics), figsize=(8*len(metrics), len(all_results)))
        if len(metrics) == 1:
            axes = [axes]
        
        for idx, metric in enumerate(metrics):
            # Create matrix
            data = []
            for result in all_results:
                row = [result[method][metric] for method in methods]
                data.append(row)
            
            data = np.array(data)
            
            # Plot heatmap
            im = axes[idx].imshow(data, cmap='RdYlGn', aspect='auto')
            
            # Set ticks and labels
            axes[idx].set_xticks(range(len(methods)))
            axes[idx].set_xticklabels([m.replace('_', '\n') for m in methods], rotation=45, ha='right')
            axes[idx].set_yticks(range(len(all_results)))
            axes[idx].set_yticklabels([f'Image {i+1}' for i in range(len(all_results))])
            axes[idx].set_title(f'{metric.upper()} Heatmap', fontsize=14, fontweight='bold')
            
            # Add colorbar
            plt.colorbar(im, ax=axes[idx])
            
            # Add value annotations
            for i in range(len(all_results)):
                for j in range(len(methods)):
                    text = axes[idx].text(j, i, f'{data[i, j]:.2f}',
                                        ha="center", va="center", color="black", fontsize=8)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Metrics heatmap saved to {save_path}")
        
        plt.close()
    
    def plot_detail_comparison(self, original, noisy, denoised_dict, crop_region=None, save_path=None):
        """
        Plot zoomed-in detail comparison
        
        Args:
            original: Original image
            noisy: Noisy image
            denoised_dict: Dictionary of denoised images
            crop_region: (x, y, width, height) for crop region
            save_path: Path to save the figure
        """
        if crop_region is None:
            # Default: center crop
            h, w = original.shape[:2]
            size = min(h, w) // 4
            x, y = w // 2 - size // 2, h // 2 - size // 2
            crop_region = (x, y, size, size)
        
        x, y, width, height = crop_region
        
        # Crop images
        original_crop = original[y:y+height, x:x+width]
        noisy_crop = noisy[y:y+height, x:x+width]
        
        num_methods = len(denoised_dict)
        fig, axes = plt.subplots(2, num_methods + 2, figsize=(4*(num_methods + 2), 8))
        
        # Full images (top row)
        axes[0, 0].imshow(original)
        axes[0, 0].set_title('Original (Full)', fontsize=12, fontweight='bold')
        axes[0, 0].add_patch(plt.Rectangle((x, y), width, height, 
                                          fill=False, edgecolor='red', linewidth=2))
        axes[0, 0].axis('off')
        
        axes[0, 1].imshow(noisy)
        axes[0, 1].set_title('Noisy (Full)', fontsize=12, fontweight='bold')
        axes[0, 1].add_patch(plt.Rectangle((x, y), width, height, 
                                          fill=False, edgecolor='red', linewidth=2))
        axes[0, 1].axis('off')
        
        for idx, (method_name, denoised_img) in enumerate(denoised_dict.items(), start=2):
            axes[0, idx].imshow(denoised_img)
            axes[0, idx].set_title(f'{method_name.replace("_", " ").title()} (Full)', 
                                  fontsize=12, fontweight='bold')
            axes[0, idx].add_patch(plt.Rectangle((x, y), width, height, 
                                                fill=False, edgecolor='red', linewidth=2))
            axes[0, idx].axis('off')
        
        # Cropped images (bottom row)
        axes[1, 0].imshow(original_crop)
        axes[1, 0].set_title('Original (Detail)', fontsize=12, fontweight='bold')
        axes[1, 0].axis('off')
        
        axes[1, 1].imshow(noisy_crop)
        axes[1, 1].set_title('Noisy (Detail)', fontsize=12, fontweight='bold')
        axes[1, 1].axis('off')
        
        for idx, (method_name, denoised_img) in enumerate(denoised_dict.items(), start=2):
            denoised_crop = denoised_img[y:y+height, x:x+width]
            axes[1, idx].imshow(denoised_crop)
            axes[1, idx].set_title(f'{method_name.replace("_", " ").title()} (Detail)', 
                                  fontsize=12, fontweight='bold')
            axes[1, idx].axis('off')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Detail comparison saved to {save_path}")
        
        plt.close()


if __name__ == "__main__":
    # Test visualizer
    viz = Visualizer()
    
    # Create test data
    original = np.random.randint(0, 256, (256, 256, 3), dtype=np.uint8)
    noisy = np.clip(original + np.random.randn(*original.shape) * 25, 0, 255).astype(np.uint8)
    
    denoised_dict = {
        'median': np.clip(original + np.random.randn(*original.shape) * 10, 0, 255).astype(np.uint8),
        'wavelet': np.clip(original + np.random.randn(*original.shape) * 8, 0, 255).astype(np.uint8),
        'dncnn': np.clip(original + np.random.randn(*original.shape) * 5, 0, 255).astype(np.uint8)
    }
    
    print("Testing visualization functions...")
    viz.plot_comparison(original, noisy, denoised_dict, 
                       save_path='visualizations/test_comparison.png')
    print("Visualizer test completed!")
