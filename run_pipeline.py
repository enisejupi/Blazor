"""
Main Automation Script - Run Complete Pipeline
Skripti Kryesor i Automatizimit - Ekzekuto të gjithë procesin
"""
import os
import sys
import yaml
import numpy as np
import cv2
from tqdm import tqdm
import json
from datetime import datetime

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.data_loader import ImageDownloader, NoiseGenerator
from src.classical_methods import ClassicalDenoisers
from src.metrics import MetricsEvaluator
from src.hybrid_methods import HybridDenoiser, MethodRecommender
from src.visualization import Visualizer
from src.report_generator import PDFReportGenerator

# Try to import DnCNN components - they're optional
try:
    from src.dncnn_model import DnCNNTrainer, create_training_data, DenoisingDataset, PYTORCH_AVAILABLE
    from torch.utils.data import DataLoader
    if not PYTORCH_AVAILABLE:
        print("⚠️  Warning: PyTorch not available. DNN-based denoising will be skipped.")
        print("   Classical and hybrid methods will still work.")
except ImportError:
    PYTORCH_AVAILABLE = False
    DnCNNTrainer = None
    create_training_data = None
    DenoisingDataset = None
    DataLoader = None
    print("⚠️  Warning: PyTorch not installed. DNN-based denoising will be skipped.")
    print("   To enable DNN features, install PyTorch or use Python 3.11/3.12.")


class AutomatedPipeline:
    """Automated pipeline for complete denoising comparison project"""
    
    def __init__(self, config_path='config.yaml'):
        print("="*80)
        print("🚀 FILLIMI I PROJEKTIT TË HEQJES SË ZHURMËS NGA IMAZHET")
        print("   STARTING IMAGE DENOISING COMPARISON PROJECT")
        print("="*80)
        
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        # Initialize components
        self.downloader = ImageDownloader(config_path)
        self.noise_gen = NoiseGenerator(config_path)
        self.classical = ClassicalDenoisers(config_path)
        self.metrics_eval = MetricsEvaluator(config_path)
        self.visualizer = Visualizer()
        self.report_gen = PDFReportGenerator()
        
        # Initialize DnCNN only if PyTorch is available
        if PYTORCH_AVAILABLE and DnCNNTrainer is not None:
            self.dncnn_trainer = DnCNNTrainer(config_path)
            self.pytorch_enabled = True
        else:
            self.dncnn_trainer = None
            self.pytorch_enabled = False
            print("ℹ️  DNN methods disabled (PyTorch not available)")
        
        # Results storage
        self.results = {}
    
    def step1_download_images(self):
        """Step 1: Download and prepare sample images"""
        print("\n" + "="*80)
        print("📥 HAPI 1: Shkarkimi i imazheve / STEP 1: Downloading Images")
        print("="*80)
        
        num_downloaded = self.downloader.download_images()
        self.images, self.image_files = self.downloader.load_images()
        
        print(f"✅ U shkarkuan/krijuan {num_downloaded} imazhe")
        print(f"✅ Downloaded/created {num_downloaded} images")
    
    def step2_train_dncnn(self, quick_train=False):
        """Step 2: Train DnCNN model"""
        print("\n" + "="*80)
        print("🧠 HAPI 2: Trajnimi i modelit DnCNN / STEP 2: Training DnCNN Model")
        print("="*80)
        
        if not self.pytorch_enabled:
            print("⚠️  Duke anashkaluar trajnimin DnCNN (PyTorch nuk është i disponueshëm)")
            print("⚠️  Skipping DnCNN training (PyTorch not available)")
            return
        
        model_path = 'models/dncnn.pth'
        
        if os.path.exists(model_path):
            print("ℹ️  Modeli ekzistues u gjet, po ngarkohet...")
            print("ℹ️  Existing model found, loading...")
            self.dncnn_trainer.load_model(model_path)
            return
        
        print("🔨 Duke krijuar të dhëna trajnimi...")
        print("🔨 Creating training data...")
        
        # Create training data
        if quick_train:
            train_size = 100
            val_size = 20
            epochs = 10
        else:
            train_size = self.config['dncnn']['train_size']
            val_size = self.config['dncnn']['val_size']
            epochs = self.config['dncnn']['epochs']
        
        train_clean, train_noisy = create_training_data(self.images, self.noise_gen, train_size)
        val_clean, val_noisy = create_training_data(self.images, self.noise_gen, val_size)
        
        # Create dataloaders
        train_dataset = DenoisingDataset(train_clean, train_noisy)
        val_dataset = DenoisingDataset(val_clean, val_noisy)
        
        train_loader = DataLoader(train_dataset, batch_size=self.config['dncnn']['batch_size'], 
                                 shuffle=True, num_workers=0)
        val_loader = DataLoader(val_dataset, batch_size=self.config['dncnn']['batch_size'], 
                               shuffle=False, num_workers=0)
        
        print(f"📊 Të dhënat e trajnimit: {len(train_dataset)} shembuj")
        print(f"📊 Training data: {len(train_dataset)} samples")
        print(f"📊 Të dhënat e validimit: {len(val_dataset)} shembuj")
        print(f"📊 Validation data: {len(val_dataset)} samples")
        
        # Create and train model
        self.dncnn_trainer.create_model()
        
        # Override epochs for quick training
        if quick_train:
            self.dncnn_trainer.epochs = epochs
        
        print(f"\n🎓 Duke filluar trajnimin për {self.dncnn_trainer.epochs} epoka...")
        print(f"🎓 Starting training for {self.dncnn_trainer.epochs} epochs...")
        
        self.dncnn_trainer.train(train_loader, val_loader, model_path)
        
        print("✅ Trajnimi i DnCNN përfundoi me sukses!")
        print("✅ DnCNN training completed successfully!")
    
    def step3_generate_noisy_images(self):
        """Step 3: Generate noisy versions of all images"""
        print("\n" + "="*80)
        print("🔊 HAPI 3: Gjenerimi i imazheve me zhurmë / STEP 3: Generating Noisy Images")
        print("="*80)
        
        self.noisy_dataset = self.noise_gen.generate_noisy_dataset(self.images)
        
        for noise_type, noisy_imgs in self.noisy_dataset.items():
            print(f"  ✓ {noise_type}: {len(noisy_imgs)} imazhe / images")
    
    def step4_denoise_all_images(self):
        """Step 4: Apply all denoising methods to all images"""
        print("\n" + "="*80)
        print("🎨 HAPI 4: Aplikimi i metodave të heqjes së zhurmës / STEP 4: Applying Denoising Methods")
        print("="*80)
        
        # Initialize hybrid denoiser if PyTorch is available
        if self.pytorch_enabled:
            self.hybrid = HybridDenoiser(self.classical, self.dncnn_trainer)
            methods = ['median', 'wiener', 'wavelet', 'dncnn', 'hybrid']
        else:
            self.hybrid = None
            methods = ['median', 'wiener', 'wavelet']
            print("ℹ️  Using classical methods only (DnCNN and hybrid skipped)")
        
        # Process first 5 images for each noise type
        num_test_images = min(5, len(self.images))
        
        for noise_type, noisy_images in self.noisy_dataset.items():
            print(f"\n📌 Duke përpunuar zhurmën {noise_type}...")
            print(f"📌 Processing {noise_type} noise...")
            
            self.results[noise_type] = []
            
            for idx in tqdm(range(num_test_images), desc=f"  {noise_type}"):
                original = self.images[idx]
                noisy = noisy_images[idx]
                
                denoised_dict = {}
                
                # Apply each method
                denoised_dict['median'] = self.classical.denoise_image(noisy, method='median', kernel_size=5)
                denoised_dict['wiener'] = self.classical.denoise_image(noisy, method='wiener')
                denoised_dict['wavelet'] = self.classical.denoise_image(noisy, method='wavelet')
                
                # Apply DNN methods only if available
                if self.pytorch_enabled:
                    denoised_dict['dncnn'] = self.dncnn_trainer.denoise(noisy)
                    denoised_dict['hybrid'] = self.hybrid.denoise_hybrid_cascade(noisy)
                
                # Calculate metrics
                metrics_dict = {}
                for method, denoised_img in denoised_dict.items():
                    metrics_dict[method] = self.metrics_eval.evaluate_all_metrics(
                        original, denoised_img, include_niqe=False
                    )
                
                self.results[noise_type].append({
                    'original': original,
                    'noisy': noisy,
                    'denoised': denoised_dict,
                    'metrics': metrics_dict,
                    'image_file': self.image_files[idx]
                })
        
        print("\n✅ Të gjitha metodat u aplikuan me sukses!")
        print("✅ All methods applied successfully!")
    
    def step5_generate_visualizations(self):
        """Step 5: Generate all visualizations"""
        print("\n" + "="*80)
        print("📊 HAPI 5: Gjenerimi i vizualizimeve / STEP 5: Generating Visualizations")
        print("="*80)
        
        os.makedirs('visualizations', exist_ok=True)
        
        for noise_type, results_list in self.results.items():
            print(f"\n  📈 Vizualizime për {noise_type}...")
            
            for idx, result in enumerate(results_list[:3]):  # First 3 images
                # Comparison plot
                save_path = f"visualizations/{noise_type}_comparison_{idx+1}.png"
                self.visualizer.plot_comparison(
                    result['original'],
                    result['noisy'],
                    result['denoised'],
                    save_path
                )
                
                # Metrics comparison
                save_path = f"visualizations/{noise_type}_metrics_{idx+1}.png"
                self.visualizer.plot_metrics_comparison(
                    result['metrics'],
                    save_path
                )
                
                # Detail comparison
                save_path = f"visualizations/{noise_type}_detail_{idx+1}.png"
                self.visualizer.plot_detail_comparison(
                    result['original'],
                    result['noisy'],
                    result['denoised'],
                    save_path=save_path
                )
        
        # Create noise type comparison
        if len(self.images) > 0:
            sample_img = self.images[0]
            noise_comparison = {}
            for noise_type, noisy_imgs in self.noisy_dataset.items():
                noise_comparison[noise_type] = noisy_imgs[0]
            
            self.visualizer.plot_noise_comparison(
                sample_img,
                noise_comparison,
                save_path='visualizations/noise_types_comparison.png'
            )
        
        print("✅ Të gjitha vizualizimet u krijuan!")
        print("✅ All visualizations created!")
    
    def step6_generate_reports(self):
        """Step 6: Generate PDF reports"""
        print("\n" + "="*80)
        print("📄 HAPI 6: Gjenerimi i raporteve PDF / STEP 6: Generating PDF Reports")
        print("="*80)
        
        for noise_type, results_list in self.results.items():
            print(f"\n  📝 Raporti për {noise_type}...")
            
            # Generate report for first result
            result = results_list[0]
            
            # Convert method names to proper format for report
            denoised_display = {
                'Median': result['denoised']['median'],
                'Wiener': result['denoised']['wiener'],
                'Wavelet': result['denoised']['wavelet']
            }
            
            metrics_display = {
                'Median': result['metrics']['median'],
                'Wiener': result['metrics']['wiener'],
                'Wavelet': result['metrics']['wavelet']
            }
            
            # Add DNN methods if available
            if self.pytorch_enabled and 'dncnn' in result['denoised']:
                denoised_display['DnCNN'] = result['denoised']['dncnn']
                denoised_display['Hybrid'] = result['denoised']['hybrid']
                metrics_display['DnCNN'] = result['metrics']['dncnn']
                metrics_display['Hybrid'] = result['metrics']['hybrid']
            
            output_filename = f"report_{noise_type}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf"
            
            self.report_gen.generate_comparison_report(
                result['original'],
                result['noisy'],
                denoised_display,
                metrics_display,
                noise_type,
                output_filename=output_filename,
                language='sq'
            )
        
        print("\n✅ Të gjitha raportet u krijuan!")
        print("✅ All reports generated!")
    
    def step7_save_summary(self):
        """Step 7: Save summary statistics"""
        print("\n" + "="*80)
        print("💾 HAPI 7: Ruajtja e statistikave / STEP 7: Saving Statistics")
        print("="*80)
        
        summary = {
            'timestamp': datetime.now().isoformat(),
            'num_images': len(self.images),
            'noise_types': list(self.results.keys()),
            'methods': ['median', 'wiener', 'wavelet'] + (['dncnn', 'hybrid'] if self.pytorch_enabled else []),
            'pytorch_enabled': self.pytorch_enabled,
            'results_by_noise_type': {}
        }
        
        for noise_type, results_list in self.results.items():
            # Calculate average metrics
            avg_metrics = {}
            methods = list(results_list[0]['metrics'].keys())
            
            for method in methods:
                psnr_values = [r['metrics'][method]['psnr'] for r in results_list]
                ssim_values = [r['metrics'][method]['ssim'] for r in results_list]
                
                avg_metrics[method] = {
                    'avg_psnr': float(np.mean(psnr_values)),
                    'std_psnr': float(np.std(psnr_values)),
                    'avg_ssim': float(np.mean(ssim_values)),
                    'std_ssim': float(np.std(ssim_values))
                }
            
            summary['results_by_noise_type'][noise_type] = avg_metrics
        
        # Save summary
        summary_path = 'results/summary_statistics.json'
        os.makedirs('results', exist_ok=True)
        
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)
        
        print(f"✅ Statistikat u ruajtën në / Statistics saved to: {summary_path}")
        
        # Print summary
        print("\n" + "="*80)
        print("📊 PËRMBLEDHJE E REZULTATEVE / RESULTS SUMMARY")
        print("="*80)
        
        for noise_type, metrics in summary['results_by_noise_type'].items():
            print(f"\n🔊 {noise_type.upper()}:")
            for method, values in metrics.items():
                print(f"  {method:8s}: PSNR = {values['avg_psnr']:6.2f} ± {values['std_psnr']:4.2f} dB, "
                      f"SSIM = {values['avg_ssim']:.4f} ± {values['std_ssim']:.4f}")
    
    def run_complete_pipeline(self, quick_train=False):
        """Run the complete automated pipeline"""
        start_time = datetime.now()
        
        try:
            self.step1_download_images()
            self.step2_train_dncnn(quick_train=quick_train)
            self.step3_generate_noisy_images()
            self.step4_denoise_all_images()
            self.step5_generate_visualizations()
            self.step6_generate_reports()
            self.step7_save_summary()
            
            end_time = datetime.now()
            duration = end_time - start_time
            
            print("\n" + "="*80)
            print("🎉 PROJEKTI PËRFUNDOI ME SUKSES! / PROJECT COMPLETED SUCCESSFULLY!")
            print("="*80)
            print(f"⏱️  Kohëzgjatja totale / Total duration: {duration}")
            print(f"📁 Rezultatet u ruajtën në / Results saved in:")
            print(f"   • Vizualizime / Visualizations: ./visualizations/")
            print(f"   • Raporte / Reports: ./reports/")
            print(f"   • Statistika / Statistics: ./results/")
            print(f"   • Modeli / Model: ./models/")
            print("="*80)
            
            return True
            
        except Exception as e:
            print(f"\n❌ GABIM / ERROR: {str(e)}")
            import traceback
            traceback.print_exc()
            return False


def main():
    """Main entry point"""
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Automated Image Denoising Comparison Pipeline / '
                    'Tubacioni i Automatizuar i Krahasimit të Heqjes së Zhurmës'
    )
    parser.add_argument('--quick', action='store_true',
                       help='Quick training mode (fewer epochs) / Mënyra e trajnimit të shpejtë')
    parser.add_argument('--skip-training', action='store_true',
                       help='Skip DnCNN training / Anashkalo trajnimin e DnCNN')
    
    args = parser.parse_args()
    
    pipeline = AutomatedPipeline()
    
    if args.skip_training:
        print("ℹ️  Trajnimi i DnCNN do të anashkalohet / Skipping DnCNN training")
        # Create untrained model only if PyTorch is available
        if pipeline.pytorch_enabled:
            pipeline.dncnn_trainer.create_model()
        else:
            print("ℹ️  PyTorch nuk është i disponueshëm / PyTorch not available")
    
    success = pipeline.run_complete_pipeline(quick_train=args.quick)
    
    if success:
        print("\n🚀 Për të hapur dashboard-in / To open the dashboard:")
        print("   streamlit run dashboard_app.py")
        if not pipeline.pytorch_enabled:
            print("   (Vetëm metodat klasike do të jenë të disponueshme / Only classical methods will be available)")
    
    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())
