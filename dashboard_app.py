"""
Interactive Streamlit Dashboard for Image Denoising
Dashbord Interaktiv për Heqjen e Zhurmës nga Imazhet
Enhanced Multi-Page Version with Advanced Features
"""
import streamlit as st
import numpy as np
import cv2
from PIL import Image
import sys
import os
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import io
import zipfile
import json
from datetime import datetime
import base64
import pandas as pd
import time

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.classical_methods import ClassicalDenoisers
from src.data_loader import NoiseGenerator
from src.metrics import MetricsEvaluator
from src.hybrid_methods import HybridDenoiser, MethodRecommender
from src.visualization import Visualizer

# Try to import DnCNN components - they're optional
try:
    from src.dncnn_model import DnCNNTrainer, PYTORCH_AVAILABLE
    if not PYTORCH_AVAILABLE:
        DnCNNTrainer = None
except ImportError:
    PYTORCH_AVAILABLE = False
    DnCNNTrainer = None


# Configure page
st.set_page_config(
    page_title="Image Denoising Dashboard - Enhanced",
    page_icon="🖼️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 1rem;
        color: white;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        margin-bottom: 1rem;
    }
    .metric-value {
        font-size: 2.5rem;
        font-weight: bold;
    }
    .metric-label {
        font-size: 1rem;
        opacity: 0.9;
    }
    .stButton>button {
        width: 100%;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        padding: 0.75rem;
        border-radius: 0.5rem;
        font-weight: bold;
    }
    .stButton>button:hover {
        background: linear-gradient(135deg, #764ba2 0%, #667eea 100%);
    }
    .success-box {
        background-color: #d4edda;
        border-left: 4px solid #28a745;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 1rem 0;
    }
    .info-box {
        background-color: #d1ecf1;
        border-left: 4px solid #17a2b8;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 1rem 0;
    }
    .warning-box {
        background-color: #fff3cd;
        border-left: 4px solid #ffc107;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)


# Initialize session state
def init_session_state():
    """Initialize session state variables"""
    if 'page' not in st.session_state:
        st.session_state.page = 'Home'
    if 'processing_history' not in st.session_state:
        st.session_state.processing_history = []
    if 'current_results' not in st.session_state:
        st.session_state.current_results = None
    if 'language' not in st.session_state:
        st.session_state.language = 'English'
    if 'batch_results' not in st.session_state:
        st.session_state.batch_results = []
    if 'settings' not in st.session_state:
        st.session_state.settings = {
            'default_noise': 'Gaussian',
            'default_sigma': 25,
            'auto_save': True,
            'theme': 'Light'
        }


@st.cache_resource
def load_models():
    """Load all models and components"""
    classical = ClassicalDenoisers()
    noise_gen = NoiseGenerator()
    metrics_eval = MetricsEvaluator()
    visualizer = Visualizer()
    
    dncnn_trainer = None
    hybrid = None
    
    if PYTORCH_AVAILABLE and DnCNNTrainer is not None:
        dncnn_trainer = DnCNNTrainer()
        
        # Try to load DnCNN model
        model_path = 'models/dncnn.pth'
        if os.path.exists(model_path):
            dncnn_trainer.load_model(model_path)
        else:
            dncnn_trainer.create_model()
        
        hybrid = HybridDenoiser(classical, dncnn_trainer)
    
    recommender = MethodRecommender()
    
    return classical, dncnn_trainer, noise_gen, metrics_eval, hybrid, recommender, visualizer


def add_noise_to_image(image, noise_type, noise_gen, **params):
    """Add noise to image"""
    if noise_type == "Gaussian":
        sigma = params.get('sigma', 25)
        return noise_gen.add_gaussian_noise(image, sigma)
    elif noise_type == "Salt & Pepper":
        amount = params.get('amount', 0.05)
        return noise_gen.add_salt_pepper_noise(image, amount)
    elif noise_type == "Speckle":
        variance = params.get('variance', 0.1)
        return noise_gen.add_speckle_noise(image, variance)
    return image


def denoise_with_method(noisy_image, method, classical, dncnn, hybrid):
    """Denoise image with specific method"""
    method_lower = method.lower()
    
    if method_lower == 'median':
        return classical.denoise_image(noisy_image, method='median', kernel_size=5)
    elif method_lower == 'wiener':
        return classical.denoise_image(noisy_image, method='wiener')
    elif method_lower == 'wavelet':
        return classical.denoise_image(noisy_image, method='wavelet')
    elif method_lower == 'dncnn' and dncnn is not None:
        return dncnn.denoise(noisy_image)
    elif method_lower == 'hybrid' and hybrid is not None:
        return hybrid.denoise_hybrid_cascade(noisy_image)
    else:
        return noisy_image


def estimate_processing_time(image_shape, methods_count):
    """Estimate processing time based on image size and methods"""
    pixels = image_shape[0] * image_shape[1]
    base_time = (pixels / 1000000) * 2  # 2 seconds per megapixel
    total_time = base_time * methods_count
    return total_time


def create_histogram_comparison(original, noisy, denoised):
    """Create histogram comparison plot"""
    fig = make_subplots(
        rows=1, cols=3,
        subplot_titles=('Original', 'Noisy', 'Denoised')
    )
    
    for idx, (img, title) in enumerate([(original, 'Original'), (noisy, 'Noisy'), (denoised, 'Denoised')]):
        # Convert to grayscale for histogram
        if len(img.shape) == 3:
            gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        else:
            gray = img
        
        hist, bins = np.histogram(gray.flatten(), bins=256, range=[0, 256])
        
        fig.add_trace(
            go.Scatter(x=bins[:-1], y=hist, mode='lines', name=title, line=dict(color=['blue', 'red', 'green'][idx])),
            row=1, col=idx+1
        )
    
    fig.update_layout(height=300, showlegend=False)
    fig.update_xaxes(title_text="Pixel Intensity")
    fig.update_yaxes(title_text="Frequency")
    
    return fig


def create_metric_card(label, value, icon="📊"):
    """Create a styled metric card"""
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-label">{icon} {label}</div>
        <div class="metric-value">{value}</div>
    </div>
    """, unsafe_allow_html=True)


def save_to_history(original, noisy, denoised, method, metrics, noise_type):
    """Save results to processing history"""
    history_entry = {
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'method': method,
        'noise_type': noise_type,
        'metrics': metrics,
        'images': {
            'original': original,
            'noisy': noisy,
            'denoised': denoised
        }
    }
    st.session_state.processing_history.append(history_entry)


# ============================================================================
# PAGE: HOME / OVERVIEW
# ============================================================================
def page_home(classical, dncnn, noise_gen, metrics_eval, hybrid, recommender, visualizer):
    """Main home page with enhanced features"""
    
    st.markdown('<div class="main-header">🖼️ Image Denoising Dashboard</div>', unsafe_allow_html=True)
    
    # Quick stats at the top
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("📁 Total Processed", len(st.session_state.processing_history))
    with col2:
        available_methods = ['Median', 'Wiener', 'Wavelet']
        if dncnn: available_methods.extend(['DnCNN', 'Hybrid'])
        st.metric("🔧 Available Methods", len(available_methods))
    with col3:
        st.metric("📊 Batch Results", len(st.session_state.batch_results))
    with col4:
        st.metric("🌍 Language", st.session_state.language)
    
    st.markdown("---")
    
    # Image source selection
    st.subheader("📷 Image Source")
    col_source1, col_source2 = st.columns(2)
    
    with col_source1:
        image_source = st.radio(
            "Select Source",
            ["Upload Image", "Use Sample Image"],
            horizontal=True
        )
    
    original_image = None
    
    if image_source == "Upload Image":
        with col_source2:
            uploaded_file = st.file_uploader(
                "Upload Image (PNG, JPG, JPEG)",
                type=['png', 'jpg', 'jpeg']
            )
        
        if uploaded_file:
            # Show upload progress
            progress_bar = st.progress(0)
            for i in range(100):
                time.sleep(0.01)
                progress_bar.progress(i + 1)
            
            image = Image.open(uploaded_file)
            original_image = np.array(image)
            if len(original_image.shape) == 2:
                original_image = cv2.cvtColor(original_image, cv2.COLOR_GRAY2RGB)
            elif original_image.shape[2] == 4:
                original_image = cv2.cvtColor(original_image, cv2.COLOR_RGBA2RGB)
            
            st.success(f"✅ Image loaded: {original_image.shape[1]}x{original_image.shape[0]} pixels")
    else:
        sample_images = []
        if os.path.exists('data/images'):
            sample_images = [f for f in os.listdir('data/images') if f.endswith(('.png', '.jpg', '.jpeg'))]
        
        if sample_images:
            with col_source2:
                selected_sample = st.selectbox("Select Sample Image", sample_images)
            img_path = os.path.join('data/images', selected_sample)
            original_image = cv2.imread(img_path)
            original_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)
            st.success(f"✅ Sample loaded: {original_image.shape[1]}x{original_image.shape[0]} pixels")
        else:
            st.warning("⚠️ No sample images available in data/images/")
    
    if original_image is not None:
        # Noise configuration
        st.subheader("🔊 Noise Configuration")
        
        col_noise1, col_noise2, col_noise3 = st.columns([2, 2, 1])
        
        with col_noise1:
            noise_type = st.selectbox(
                "Noise Type",
                ["Gaussian", "Salt & Pepper", "Speckle"],
                help="Select the type of noise to add to the image"
            )
        
        noise_params = {}
        with col_noise2:
            if noise_type == "Gaussian":
                sigma = st.slider("Sigma (Intensity)", 5, 50, st.session_state.settings['default_sigma'])
                noise_params['sigma'] = sigma
                st.caption(f"Higher sigma = more noise. Recommended: 15-30")
            elif noise_type == "Salt & Pepper":
                amount = st.slider("Amount", 0.01, 0.20, 0.05, 0.01)
                noise_params['amount'] = amount
                st.caption(f"Percentage of pixels affected: {amount*100:.1f}%")
            else:  # Speckle
                variance = st.slider("Variance", 0.01, 0.30, 0.10, 0.01)
                noise_params['variance'] = variance
                st.caption(f"Multiplicative noise variance")
        
        with col_noise3:
            if st.button("🔄 Randomize", help="Generate random noise parameters"):
                if noise_type == "Gaussian":
                    noise_params['sigma'] = np.random.randint(15, 35)
                elif noise_type == "Salt & Pepper":
                    noise_params['amount'] = np.random.uniform(0.02, 0.10)
                else:
                    noise_params['variance'] = np.random.uniform(0.05, 0.15)
                st.rerun()
        
        # Add noise
        noisy_image = add_noise_to_image(original_image, noise_type, noise_gen, **noise_params)
        
        # Method selection
        st.subheader("🎯 Denoising Method Selection")
        
        available_methods = ['Median', 'Wiener', 'Wavelet']
        if dncnn: available_methods.extend(['DnCNN', 'Hybrid'])
        
        method_descriptions = {
            'Median': '📐 Median Filter - Best for salt & pepper noise',
            'Wiener': '🔬 Wiener Filter - Adaptive frequency domain filtering',
            'Wavelet': '🌊 Wavelet Transform - Multi-scale decomposition',
            'DnCNN': '🤖 Deep Neural Network - CNN-based denoising',
            'Hybrid': '⚡ Hybrid Method - Combines classical + DNN'
        }
        
        col_method1, col_method2 = st.columns([3, 1])
        
        with col_method1:
            selected_methods = st.multiselect(
                "Select Methods to Compare",
                available_methods,
                default=['Median', 'Wavelet'],
                help="Choose one or more denoising methods"
            )
            
            # Show descriptions
            for method in selected_methods:
                st.caption(method_descriptions[method])
        
        with col_method2:
            use_auto_recommend = st.checkbox("🎯 Auto Recommend", value=False)
            show_histograms = st.checkbox("📊 Show Histograms", value=True)
            show_metrics_detail = st.checkbox("📈 Detailed Metrics", value=True)
        
        # Preview images
        st.subheader("👀 Image Preview")
        col_prev1, col_prev2 = st.columns(2)
        
        with col_prev1:
            st.image(original_image, caption="Original Image", use_container_width=True)
        with col_prev2:
            st.image(noisy_image, caption=f"Noisy Image ({noise_type})", use_container_width=True)
        
        # Processing time estimate
        if selected_methods:
            est_time = estimate_processing_time(original_image.shape, len(selected_methods))
            st.info(f"⏱️ Estimated processing time: {est_time:.1f} seconds")
        
        # Process button
        if st.button("🚀 Start Processing", type="primary", use_container_width=True):
            if not selected_methods and not use_auto_recommend:
                st.error("❌ Please select at least one method or enable Auto Recommend")
            else:
                # Auto recommend if enabled
                if use_auto_recommend:
                    method, reason, detected_noise, confidence = recommender.recommend_method(noisy_image)
                    st.info(f"""
                    **🎯 Auto Recommendation:**
                    - Detected Noise: {detected_noise.title()} ({confidence:.1%} confidence)
                    - Recommended Method: {method.title()}
                    - Reason: {reason}
                    """)
                    if method.title() not in selected_methods:
                        selected_methods.append(method.title())
                
                # Process with selected methods
                st.subheader("📊 Processing Results")
                
                denoised_results = {}
                processing_times = {}
                
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                for idx, method in enumerate(selected_methods):
                    status_text.text(f"Processing with {method}...")
                    start_time = time.time()
                    
                    denoised_results[method] = denoise_with_method(
                        noisy_image, method, classical, dncnn, hybrid
                    )
                    
                    processing_times[method] = time.time() - start_time
                    progress_bar.progress((idx + 1) / len(selected_methods))
                
                status_text.text("✅ Processing complete!")
                
                # Display results
                cols = st.columns(len(denoised_results))
                for idx, (method_name, denoised_img) in enumerate(denoised_results.items()):
                    with cols[idx]:
                        st.image(denoised_img, caption=f"{method_name}\n({processing_times[method_name]:.2f}s)", 
                                use_container_width=True)
                
                # Show histograms
                if show_histograms:
                    st.subheader("📊 Pixel Distribution Histograms")
                    for method_name, denoised_img in denoised_results.items():
                        with st.expander(f"📈 Histogram - {method_name}"):
                            fig = create_histogram_comparison(original_image, noisy_image, denoised_img)
                            st.plotly_chart(fig, use_container_width=True)
                
                # Calculate and display metrics
                st.subheader("📈 Quality Metrics")
                
                metrics_results = {}
                for method_name, denoised_img in denoised_results.items():
                    metrics_results[method_name] = metrics_eval.evaluate_all_metrics(
                        original_image, denoised_img, include_niqe=False
                    )
                    metrics_results[method_name]['processing_time'] = processing_times[method_name]
                
                # Metrics table
                metrics_df = pd.DataFrame(metrics_results).T
                metrics_df = metrics_df.round(3)
                st.dataframe(metrics_df, use_container_width=True)
                
                # Detailed metric cards
                if show_metrics_detail:
                    st.subheader("🎯 Detailed Metrics by Method")
                    for method_name, metrics in metrics_results.items():
                        with st.expander(f"📊 {method_name} - Detailed View"):
                            col_m1, col_m2, col_m3, col_m4 = st.columns(4)
                            with col_m1:
                                create_metric_card("PSNR", f"{metrics['psnr']:.2f} dB", "📶")
                            with col_m2:
                                create_metric_card("SSIM", f"{metrics['ssim']:.4f}", "🎯")
                            with col_m3:
                                create_metric_card("MSE", f"{metrics['mse']:.2f}", "📉")
                            with col_m4:
                                create_metric_card("Time", f"{metrics['processing_time']:.2f}s", "⏱️")
                
                # Comparison chart
                st.subheader("📊 Methods Comparison")
                
                fig = make_subplots(
                    rows=1, cols=2,
                    subplot_titles=('PSNR Comparison (Higher is Better)', 'Processing Time (Lower is Better)')
                )
                
                methods = list(metrics_results.keys())
                psnr_values = [metrics_results[m]['psnr'] for m in methods]
                time_values = [metrics_results[m]['processing_time'] for m in methods]
                
                fig.add_trace(
                    go.Bar(name='PSNR', x=methods, y=psnr_values, marker_color='lightblue'),
                    row=1, col=1
                )
                
                fig.add_trace(
                    go.Bar(name='Time', x=methods, y=time_values, marker_color='lightcoral'),
                    row=1, col=2
                )
                
                fig.update_layout(height=400, showlegend=False)
                st.plotly_chart(fig, use_container_width=True)
                
                # Best method
                best_method, best_score = metrics_eval.get_best_method(metrics_results, 'psnr')
                st.success(f"✨ Best Method: **{best_method}** with PSNR: {best_score:.2f} dB")
                
                # Save to history
                if st.session_state.settings['auto_save']:
                    for method_name, denoised_img in denoised_results.items():
                        save_to_history(
                            original_image, noisy_image, denoised_img, 
                            method_name, metrics_results[method_name], noise_type
                        )
                    st.info("💾 Results saved to processing history")
                
                # Export options
                st.subheader("💾 Export Results")
                
                col_exp1, col_exp2, col_exp3 = st.columns(3)
                
                with col_exp1:
                    # Download all images as ZIP
                    zip_buffer = io.BytesIO()
                    with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zip_file:
                        # Add original
                        orig_bytes = cv2.imencode('.png', cv2.cvtColor(original_image, cv2.COLOR_RGB2BGR))[1].tobytes()
                        zip_file.writestr('original.png', orig_bytes)
                        
                        # Add noisy
                        noisy_bytes = cv2.imencode('.png', cv2.cvtColor(noisy_image, cv2.COLOR_RGB2BGR))[1].tobytes()
                        zip_file.writestr(f'noisy_{noise_type.lower()}.png', noisy_bytes)
                        
                        # Add denoised
                        for method_name, denoised_img in denoised_results.items():
                            img_bytes = cv2.imencode('.png', cv2.cvtColor(denoised_img, cv2.COLOR_RGB2BGR))[1].tobytes()
                            zip_file.writestr(f'{method_name.lower()}_denoised.png', img_bytes)
                        
                        # Add metrics as JSON
                        metrics_json = json.dumps(metrics_results, indent=2, default=str)
                        zip_file.writestr('metrics.json', metrics_json)
                    
                    st.download_button(
                        label="📦 Download All (ZIP)",
                        data=zip_buffer.getvalue(),
                        file_name=f"denoising_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.zip",
                        mime="application/zip",
                        use_container_width=True
                    )
                
                with col_exp2:
                    # Download metrics as CSV
                    csv_buffer = io.StringIO()
                    metrics_df.to_csv(csv_buffer)
                    
                    st.download_button(
                        label="📊 Download Metrics (CSV)",
                        data=csv_buffer.getvalue(),
                        file_name=f"metrics_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                        mime="text/csv",
                        use_container_width=True
                    )
                
                with col_exp3:
                    # Download best result
                    best_img_bytes = cv2.imencode('.png', cv2.cvtColor(denoised_results[best_method], cv2.COLOR_RGB2BGR))[1].tobytes()
                    
                    st.download_button(
                        label=f"⭐ Download Best ({best_method})",
                        data=best_img_bytes,
                        file_name=f"best_{best_method.lower()}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png",
                        mime="image/png",
                        use_container_width=True
                    )
    
    else:
        st.info("👈 Please upload an image or select a sample to get started")


# ============================================================================
# PAGE: BATCH PROCESSING
# ============================================================================
def page_batch_processing(classical, dncnn, noise_gen, metrics_eval, hybrid, recommender, visualizer):
    """Batch processing page for multiple images"""
    
    st.markdown('<div class="main-header">📦 Batch Processing</div>', unsafe_allow_html=True)
    
    st.info("""
    **Batch Processing allows you to:**
    - Process multiple images at once
    - Apply the same denoising method to all images
    - Compare different methods across multiple images
    - Export batch results efficiently
    """)
    
    # Upload multiple files
    st.subheader("📁 Upload Images")
    uploaded_files = st.file_uploader(
        "Upload multiple images",
        type=['png', 'jpg', 'jpeg'],
        accept_multiple_files=True
    )
    
    if uploaded_files:
        st.success(f"✅ {len(uploaded_files)} images uploaded")
        
        # Batch configuration
        col_config1, col_config2 = st.columns(2)
        
        with col_config1:
            noise_type = st.selectbox("Noise Type", ["Gaussian", "Salt & Pepper", "Speckle"])
            
            noise_params = {}
            if noise_type == "Gaussian":
                sigma = st.slider("Sigma", 5, 50, 25, key="batch_sigma")
                noise_params['sigma'] = sigma
            elif noise_type == "Salt & Pepper":
                amount = st.slider("Amount", 0.01, 0.20, 0.05, 0.01, key="batch_amount")
                noise_params['amount'] = amount
            else:
                variance = st.slider("Variance", 0.01, 0.30, 0.10, 0.01, key="batch_variance")
                noise_params['variance'] = variance
        
        with col_config2:
            available_methods = ['Median', 'Wiener', 'Wavelet']
            if dncnn: available_methods.extend(['DnCNN', 'Hybrid'])
            
            selected_methods = st.multiselect(
                "Select Methods",
                available_methods,
                default=['Median']
            )
        
        # Process batch button
        if st.button("🚀 Process Batch", type="primary", use_container_width=True):
            if not selected_methods:
                st.error("❌ Please select at least one method")
            else:
                batch_results = []
                
                # Overall progress
                overall_progress = st.progress(0)
                status_text = st.empty()
                
                for file_idx, uploaded_file in enumerate(uploaded_files):
                    status_text.text(f"Processing image {file_idx + 1}/{len(uploaded_files)}: {uploaded_file.name}")
                    
                    # Load image
                    image = Image.open(uploaded_file)
                    original_image = np.array(image)
                    if len(original_image.shape) == 2:
                        original_image = cv2.cvtColor(original_image, cv2.COLOR_GRAY2RGB)
                    elif original_image.shape[2] == 4:
                        original_image = cv2.cvtColor(original_image, cv2.COLOR_RGBA2RGB)
                    
                    # Add noise
                    noisy_image = add_noise_to_image(original_image, noise_type, noise_gen, **noise_params)
                    
                    # Process with each method
                    image_results = {
                        'filename': uploaded_file.name,
                        'original': original_image,
                        'noisy': noisy_image,
                        'denoised': {},
                        'metrics': {}
                    }
                    
                    for method in selected_methods:
                        denoised = denoise_with_method(noisy_image, method, classical, dncnn, hybrid)
                        image_results['denoised'][method] = denoised
                        
                        metrics = metrics_eval.evaluate_all_metrics(original_image, denoised, include_niqe=False)
                        image_results['metrics'][method] = metrics
                    
                    batch_results.append(image_results)
                    overall_progress.progress((file_idx + 1) / len(uploaded_files))
                
                status_text.text("✅ Batch processing complete!")
                st.session_state.batch_results = batch_results
                
                # Display results
                st.subheader("📊 Batch Results")
                
                # Summary statistics
                st.subheader("📈 Summary Statistics")
                
                summary_data = {}
                for method in selected_methods:
                    avg_psnr = np.mean([r['metrics'][method]['psnr'] for r in batch_results])
                    avg_ssim = np.mean([r['metrics'][method]['ssim'] for r in batch_results])
                    avg_mse = np.mean([r['metrics'][method]['mse'] for r in batch_results])
                    
                    summary_data[method] = {
                        'Avg PSNR': f"{avg_psnr:.2f}",
                        'Avg SSIM': f"{avg_ssim:.4f}",
                        'Avg MSE': f"{avg_mse:.2f}"
                    }
                
                summary_df = pd.DataFrame(summary_data).T
                st.dataframe(summary_df, use_container_width=True)
                
                # Individual results
                st.subheader("🖼️ Individual Results")
                
                for idx, result in enumerate(batch_results):
                    with st.expander(f"Image {idx + 1}: {result['filename']}"):
                        cols = st.columns(2 + len(selected_methods))
                        
                        cols[0].image(result['original'], caption="Original", use_container_width=True)
                        cols[1].image(result['noisy'], caption="Noisy", use_container_width=True)
                        
                        for method_idx, method in enumerate(selected_methods):
                            cols[2 + method_idx].image(
                                result['denoised'][method],
                                caption=f"{method}\nPSNR: {result['metrics'][method]['psnr']:.2f}",
                                use_container_width=True
                            )
                
                # Export batch results
                st.subheader("💾 Export Batch Results")
                
                zip_buffer = io.BytesIO()
                with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zip_file:
                    for idx, result in enumerate(batch_results):
                        # Create folder for each image
                        folder_name = f"image_{idx + 1}"
                        
                        # Save original
                        orig_bytes = cv2.imencode('.png', cv2.cvtColor(result['original'], cv2.COLOR_RGB2BGR))[1].tobytes()
                        zip_file.writestr(f"{folder_name}/original.png", orig_bytes)
                        
                        # Save noisy
                        noisy_bytes = cv2.imencode('.png', cv2.cvtColor(result['noisy'], cv2.COLOR_RGB2BGR))[1].tobytes()
                        zip_file.writestr(f"{folder_name}/noisy.png", noisy_bytes)
                        
                        # Save denoised
                        for method in selected_methods:
                            denoised_bytes = cv2.imencode('.png', cv2.cvtColor(result['denoised'][method], cv2.COLOR_RGB2BGR))[1].tobytes()
                            zip_file.writestr(f"{folder_name}/{method.lower()}_denoised.png", denoised_bytes)
                        
                        # Save metrics
                        metrics_json = json.dumps(result['metrics'], indent=2, default=str)
                        zip_file.writestr(f"{folder_name}/metrics.json", metrics_json)
                    
                    # Save summary
                    summary_json = json.dumps(summary_data, indent=2)
                    zip_file.writestr("summary.json", summary_json)
                
                st.download_button(
                    label="📦 Download All Batch Results",
                    data=zip_buffer.getvalue(),
                    file_name=f"batch_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.zip",
                    mime="application/zip",
                    use_container_width=True
                )
    else:
        st.info("📁 Please upload multiple images to start batch processing")


# ============================================================================
# PAGE: COMPARISON GALLERY
# ============================================================================
def page_comparison_gallery():
    """Gallery page to view all previous results"""
    
    st.markdown('<div class="main-header">🖼️ Comparison Gallery</div>', unsafe_allow_html=True)
    
    if not st.session_state.processing_history:
        st.info("📷 No processing history yet. Process some images to see them here!")
        return
    
    st.subheader(f"📊 Total Results: {len(st.session_state.processing_history)}")
    
    # Filter options
    col_filter1, col_filter2, col_filter3 = st.columns(3)
    
    with col_filter1:
        filter_method = st.multiselect(
            "Filter by Method",
            list(set([h['method'] for h in st.session_state.processing_history])),
            default=[]
        )
    
    with col_filter2:
        filter_noise = st.multiselect(
            "Filter by Noise Type",
            list(set([h['noise_type'] for h in st.session_state.processing_history])),
            default=[]
        )
    
    with col_filter3:
        sort_by = st.selectbox(
            "Sort by",
            ["Timestamp (Newest)", "Timestamp (Oldest)", "PSNR (Highest)", "PSNR (Lowest)"]
        )
    
    # Apply filters
    filtered_history = st.session_state.processing_history
    
    if filter_method:
        filtered_history = [h for h in filtered_history if h['method'] in filter_method]
    
    if filter_noise:
        filtered_history = [h for h in filtered_history if h['noise_type'] in filter_noise]
    
    # Sort
    if "PSNR" in sort_by:
        filtered_history = sorted(filtered_history, key=lambda x: x['metrics']['psnr'], 
                                 reverse="Highest" in sort_by)
    elif "Oldest" in sort_by:
        filtered_history = filtered_history[::-1]
    
    # Display gallery
    st.markdown("---")
    
    for idx, entry in enumerate(filtered_history):
        with st.expander(f"🖼️ Result #{len(filtered_history) - idx} - {entry['method']} on {entry['noise_type']} noise - {entry['timestamp']}"):
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.image(entry['images']['original'], caption="Original", use_container_width=True)
            with col2:
                st.image(entry['images']['noisy'], caption=f"Noisy ({entry['noise_type']})", use_container_width=True)
            with col3:
                st.image(entry['images']['denoised'], caption=f"Denoised ({entry['method']})", use_container_width=True)
            
            # Metrics
            col_m1, col_m2, col_m3 = st.columns(3)
            col_m1.metric("PSNR", f"{entry['metrics']['psnr']:.2f} dB")
            col_m2.metric("SSIM", f"{entry['metrics']['ssim']:.4f}")
            col_m3.metric("MSE", f"{entry['metrics']['mse']:.2f}")
            
            # Download individual result
            img_bytes = cv2.imencode('.png', cv2.cvtColor(entry['images']['denoised'], cv2.COLOR_RGB2BGR))[1].tobytes()
            st.download_button(
                label="📥 Download Result",
                data=img_bytes,
                file_name=f"result_{idx}.png",
                mime="image/png",
                key=f"download_{idx}"
            )
    
    # Clear history button
    st.markdown("---")
    if st.button("🗑️ Clear All History", type="secondary"):
        st.session_state.processing_history = []
        st.rerun()


# ============================================================================
# PAGE: MODEL PERFORMANCE ANALYTICS
# ============================================================================
def page_analytics():
    """Analytics page with detailed metrics visualization"""
    
    st.markdown('<div class="main-header">📊 Model Performance Analytics</div>', unsafe_allow_html=True)
    
    if not st.session_state.processing_history:
        st.info("📈 No data available yet. Process some images to see analytics!")
        return
    
    # Aggregate data
    methods_data = {}
    for entry in st.session_state.processing_history:
        method = entry['method']
        if method not in methods_data:
            methods_data[method] = {
                'psnr': [],
                'ssim': [],
                'mse': [],
                'timestamps': []
            }
        
        methods_data[method]['psnr'].append(entry['metrics']['psnr'])
        methods_data[method]['ssim'].append(entry['metrics']['ssim'])
        methods_data[method]['mse'].append(entry['metrics']['mse'])
        methods_data[method]['timestamps'].append(entry['timestamp'])
    
    # Summary statistics
    st.subheader("📈 Overall Performance Summary")
    
    summary_cols = st.columns(len(methods_data))
    for idx, (method, data) in enumerate(methods_data.items()):
        with summary_cols[idx]:
            st.markdown(f"### {method}")
            st.metric("Avg PSNR", f"{np.mean(data['psnr']):.2f} dB")
            st.metric("Avg SSIM", f"{np.mean(data['ssim']):.4f}")
            st.metric("Count", len(data['psnr']))
    
    # Performance trends
    st.subheader("📊 Performance Trends Over Time")
    
    fig = go.Figure()
    
    for method, data in methods_data.items():
        fig.add_trace(go.Scatter(
            x=list(range(len(data['psnr']))),
            y=data['psnr'],
            mode='lines+markers',
            name=method,
            line=dict(width=2),
            marker=dict(size=8)
        ))
    
    fig.update_layout(
        title="PSNR Trend Across Processing Sessions",
        xaxis_title="Processing Session",
        yaxis_title="PSNR (dB)",
        height=400,
        hovermode='x unified'
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Distribution plots
    st.subheader("📊 Metric Distributions")
    
    fig = make_subplots(
        rows=1, cols=3,
        subplot_titles=('PSNR Distribution', 'SSIM Distribution', 'MSE Distribution')
    )
    
    for method, data in methods_data.items():
        fig.add_trace(
            go.Box(y=data['psnr'], name=method, showlegend=True),
            row=1, col=1
        )
        fig.add_trace(
            go.Box(y=data['ssim'], name=method, showlegend=False),
            row=1, col=2
        )
        fig.add_trace(
            go.Box(y=data['mse'], name=method, showlegend=False),
            row=1, col=3
        )
    
    fig.update_layout(height=400)
    st.plotly_chart(fig, use_container_width=True)
    
    # Method comparison
    st.subheader("🏆 Method Ranking")
    
    ranking_data = []
    for method, data in methods_data.items():
        ranking_data.append({
            'Method': method,
            'Avg PSNR': np.mean(data['psnr']),
            'Avg SSIM': np.mean(data['ssim']),
            'Avg MSE': np.mean(data['mse']),
            'Std PSNR': np.std(data['psnr']),
            'Count': len(data['psnr'])
        })
    
    ranking_df = pd.DataFrame(ranking_data)
    ranking_df = ranking_df.sort_values('Avg PSNR', ascending=False)
    ranking_df = ranking_df.round(3)
    
    st.dataframe(ranking_df, use_container_width=True)
    
    # Best performer
    best_method = ranking_df.iloc[0]['Method']
    best_psnr = ranking_df.iloc[0]['Avg PSNR']
    
    st.success(f"🏆 Best Performing Method: **{best_method}** with average PSNR of {best_psnr:.2f} dB")


# ============================================================================
# PAGE: SETTINGS
# ============================================================================
def page_settings():
    """Settings and configuration page"""
    
    st.markdown('<div class="main-header">⚙️ Settings & Configuration</div>', unsafe_allow_html=True)
    
    st.subheader("🌍 General Settings")
    
    col_set1, col_set2 = st.columns(2)
    
    with col_set1:
        language = st.selectbox(
            "Language",
            ["English", "Shqip (Albanian)"],
            index=0 if st.session_state.language == "English" else 1
        )
        st.session_state.language = language
        
        theme = st.selectbox(
            "Theme",
            ["Light", "Dark"],
            index=0 if st.session_state.settings['theme'] == "Light" else 1
        )
        st.session_state.settings['theme'] = theme
    
    with col_set2:
        auto_save = st.checkbox(
            "Auto-save results to history",
            value=st.session_state.settings['auto_save']
        )
        st.session_state.settings['auto_save'] = auto_save
    
    st.markdown("---")
    
    # Default noise settings
    st.subheader("🔊 Default Noise Settings")
    
    col_noise1, col_noise2 = st.columns(2)
    
    with col_noise1:
        default_noise = st.selectbox(
            "Default Noise Type",
            ["Gaussian", "Salt & Pepper", "Speckle"],
            index=["Gaussian", "Salt & Pepper", "Speckle"].index(st.session_state.settings['default_noise'])
        )
        st.session_state.settings['default_noise'] = default_noise
    
    with col_noise2:
        if default_noise == "Gaussian":
            default_sigma = st.slider(
                "Default Sigma",
                5, 50,
                st.session_state.settings['default_sigma']
            )
            st.session_state.settings['default_sigma'] = default_sigma
    
    st.markdown("---")
    
    # Export/Import settings
    st.subheader("💾 Data Management")
    
    col_data1, col_data2, col_data3 = st.columns(3)
    
    with col_data1:
        # Export history
        if st.button("📤 Export History", use_container_width=True):
            history_json = json.dumps(
                [{**h, 'images': None} for h in st.session_state.processing_history],
                indent=2,
                default=str
            )
            st.download_button(
                label="Download History JSON",
                data=history_json,
                file_name=f"history_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                mime="application/json"
            )
    
    with col_data2:
        # Clear cache
        if st.button("🗑️ Clear Cache", use_container_width=True):
            st.cache_resource.clear()
            st.success("✅ Cache cleared!")
    
    with col_data3:
        # Reset settings
        if st.button("🔄 Reset Settings", use_container_width=True):
            st.session_state.settings = {
                'default_noise': 'Gaussian',
                'default_sigma': 25,
                'auto_save': True,
                'theme': 'Light'
            }
            st.success("✅ Settings reset to defaults!")
    
    st.markdown("---")
    
    # System info
    st.subheader("ℹ️ System Information")
    
    col_info1, col_info2 = st.columns(2)
    
    with col_info1:
        st.info(f"""
        **Processing History:**
        - Total entries: {len(st.session_state.processing_history)}
        - Batch results: {len(st.session_state.batch_results)}
        """)
    
    with col_info2:
        st.info(f"""
        **Available Features:**
        - PyTorch: {'✅ Available' if PYTORCH_AVAILABLE else '❌ Not available'}
        - DnCNN: {'✅ Available' if DnCNNTrainer is not None else '❌ Not available'}
        - Classical Methods: ✅ Available
        """)


# ============================================================================
# MAIN APP
# ============================================================================
def main():
    """Main application"""
    
    # Initialize session state
    init_session_state()
    
    # Load models
    classical, dncnn, noise_gen, metrics_eval, hybrid, recommender, visualizer = load_models()
    
    # Sidebar navigation
    st.sidebar.title("🧭 Navigation")
    
    pages = {
        "🏠 Home": "Home",
        "📦 Batch Processing": "Batch",
        "🖼️ Comparison Gallery": "Gallery",
        "📊 Analytics": "Analytics",
        "⚙️ Settings": "Settings"
    }
    
    page = st.sidebar.radio("Go to", list(pages.keys()))
    st.session_state.page = pages[page]
    
    # Quick stats in sidebar
    st.sidebar.markdown("---")
    st.sidebar.subheader("📊 Quick Stats")
    st.sidebar.metric("Processed", len(st.session_state.processing_history))
    st.sidebar.metric("Batch Results", len(st.session_state.batch_results))
    
    # Model status
    st.sidebar.markdown("---")
    st.sidebar.subheader("🔧 Model Status")
    st.sidebar.write("✅ Classical Methods")
    if dncnn:
        st.sidebar.write("✅ DnCNN")
        st.sidebar.write("✅ Hybrid")
    else:
        st.sidebar.write("⚠️ DnCNN (Unavailable)")
        st.sidebar.write("⚠️ Hybrid (Unavailable)")
    
    # Render selected page
    st.sidebar.markdown("---")
    
    if st.session_state.page == "Home":
        page_home(classical, dncnn, noise_gen, metrics_eval, hybrid, recommender, visualizer)
    elif st.session_state.page == "Batch":
        page_batch_processing(classical, dncnn, noise_gen, metrics_eval, hybrid, recommender, visualizer)
    elif st.session_state.page == "Gallery":
        page_comparison_gallery()
    elif st.session_state.page == "Analytics":
        page_analytics()
    elif st.session_state.page == "Settings":
        page_settings()
    
    # Footer
    st.sidebar.markdown("---")
    st.sidebar.markdown("""
    <div style='text-align: center; font-size: 0.8rem; color: gray;'>
        🎓 Image Denoising Dashboard<br>
        Enhanced Multi-Page Version<br>
        v2.0
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
