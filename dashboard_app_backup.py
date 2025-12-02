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
        st.error("⚠️ Warning: PyTorch not available. DNN-based denoising will be disabled.")
        DnCNNTrainer = None
except ImportError:
    PYTORCH_AVAILABLE = False
    DnCNNTrainer = None
    st.warning("⚠️ PyTorch not installed. DNN features disabled. Classical methods still available.")


# Configure page
st.set_page_config(
    page_title="Krahasimi i Metodave të Heqjes së Zhurmës - Image Denoising Comparison",
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
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
    }
    .stButton>button {
        width: 100%;
        background-color: #1f77b4;
        color: white;
    }
</style>
""", unsafe_allow_html=True)


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
            st.sidebar.success("✅ Modeli DnCNN u ngarkua me sukses / DnCNN model loaded")
        else:
            dncnn_trainer.create_model()
            st.sidebar.warning("⚠️ DnCNN po përdor peshë të patreniara / Using untrained weights")
        
        hybrid = HybridDenoiser(classical, dncnn_trainer)
    else:
        st.sidebar.info("ℹ️ DnCNN dhe Hybrid jo të disponueshme / DnCNN and Hybrid not available")
    
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


def denoise_with_all_methods(noisy_image, classical, dncnn, hybrid):
    """Denoise image with all available methods"""
    results = {}
    
    with st.spinner('🔄 Duke përpunuar me Median Filter...'):
        results['Median'] = classical.denoise_image(noisy_image, method='median', kernel_size=5)
    
    with st.spinner('🔄 Duke përpunuar me Wiener Filter...'):
        results['Wiener'] = classical.denoise_image(noisy_image, method='wiener')
    
    with st.spinner('🔄 Duke përpunuar me Wavelet...'):
        results['Wavelet'] = classical.denoise_image(noisy_image, method='wavelet')
    
    # Only apply DNN methods if available
    if dncnn is not None and PYTORCH_AVAILABLE:
        with st.spinner('🔄 Duke përpunuar me DnCNN...'):
            results['DnCNN'] = dncnn.denoise(noisy_image)
        
        if hybrid is not None:
            with st.spinner('🔄 Duke përpunuar me metodë Hibride...'):
                results['Hybrid'] = hybrid.denoise_hybrid_cascade(noisy_image)
    
    return results


def main():
    # Title
    st.markdown('<div class="main-header">🖼️ Krahasimi i Metodave të Heqjes së Zhurmës<br>Image Denoising Methods Comparison</div>', 
                unsafe_allow_html=True)
    
    # Load models
    classical, dncnn, noise_gen, metrics_eval, hybrid, recommender, visualizer = load_models()
    
    # Sidebar
    st.sidebar.title("⚙️ Parametrat / Settings")
    
    # Language selector
    language = st.sidebar.selectbox(
        "Gjuha / Language",
        ["Shqip (Albanian)", "English"]
    )
    is_albanian = language.startswith("Shqip")
    
    # Image source
    image_source = st.sidebar.radio(
        "Burimi i imazhit / Image Source" if is_albanian else "Image Source",
        ["Ngarko imazh / Upload", "Përdor shembull / Use Sample"]
    )
    
    original_image = None
    
    if "Upload" in image_source:
        uploaded_file = st.sidebar.file_uploader(
            "Ngarko imazhin / Upload Image",
            type=['png', 'jpg', 'jpeg']
        )
        
        if uploaded_file:
            image = Image.open(uploaded_file)
            original_image = np.array(image)
            if len(original_image.shape) == 2:
                original_image = cv2.cvtColor(original_image, cv2.COLOR_GRAY2RGB)
            elif original_image.shape[2] == 4:
                original_image = cv2.cvtColor(original_image, cv2.COLOR_RGBA2RGB)
    else:
        # Use sample image
        sample_images = []
        if os.path.exists('data/images'):
            sample_images = [f for f in os.listdir('data/images') if f.endswith(('.png', '.jpg', '.jpeg'))]
        
        if sample_images:
            selected_sample = st.sidebar.selectbox(
                "Zgjidhni imazhin / Select Image",
                sample_images
            )
            img_path = os.path.join('data/images', selected_sample)
            original_image = cv2.imread(img_path)
            original_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)
        else:
            st.sidebar.warning("⚠️ Nuk ka imazhe / No sample images available")
    
    if original_image is not None:
        # Noise settings
        st.sidebar.subheader("🔊 Parametrat e zhurmës / Noise Parameters")
        
        noise_type = st.sidebar.selectbox(
            "Lloji i zhurmës / Noise Type",
            ["Gaussian", "Salt & Pepper", "Speckle"]
        )
        
        noise_params = {}
        if noise_type == "Gaussian":
            sigma = st.sidebar.slider("Sigma (intensiteti)", 5, 50, 25)
            noise_params['sigma'] = sigma
        elif noise_type == "Salt & Pepper":
            amount = st.sidebar.slider("Amount (sasia)", 0.01, 0.20, 0.05, 0.01)
            noise_params['amount'] = amount
        else:  # Speckle
            variance = st.sidebar.slider("Variance (varianca)", 0.01, 0.30, 0.10, 0.01)
            noise_params['variance'] = variance
        
        # Add noise
        noisy_image = add_noise_to_image(original_image, noise_type, noise_gen, **noise_params)
        
        # Method selection
        st.sidebar.subheader("🎯 Zgjidhni metodën / Select Method")
        
        method_choice = st.sidebar.radio(
            "Mënyra e krahasimit / Comparison Mode",
            ["Të gjitha metodat / All Methods", "Metodë individuale / Single Method", "Rekomandim automatik / Auto Recommend"]
        )
        
        # Main content
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("📷 Imazhi origjinal / Original Image")
            st.image(original_image, use_container_width=True)
        
        with col2:
            st.subheader(f"🔊 Imazhi me zhurmë / Noisy Image ({noise_type})")
            st.image(noisy_image, use_container_width=True)
        
        # Process button
        if st.button("🚀 Filloni përpunimin / Start Processing", type="primary"):
            if "All Methods" in method_choice:
                # Process with all methods
                st.subheader("📊 Rezultatet e të gjitha metodave / All Methods Results")
                
                denoised_results = denoise_with_all_methods(noisy_image, classical, dncnn, hybrid)
                
                # Display results
                cols = st.columns(len(denoised_results))
                for idx, (method_name, denoised_img) in enumerate(denoised_results.items()):
                    with cols[idx]:
                        st.image(denoised_img, caption=method_name, use_container_width=True)
                
                # Calculate metrics
                st.subheader("📈 Metrikat cilësore / Quality Metrics")
                
                metrics_results = {}
                for method_name, denoised_img in denoised_results.items():
                    metrics_results[method_name] = metrics_eval.evaluate_all_metrics(
                        original_image, denoised_img, include_niqe=False
                    )
                
                # Display metrics in table
                import pandas as pd
                
                metrics_df = pd.DataFrame(metrics_results).T
                metrics_df = metrics_df.round(3)
                st.dataframe(metrics_df, use_container_width=True)
                
                # Plot metrics comparison
                fig = go.Figure()
                
                methods = list(metrics_results.keys())
                psnr_values = [metrics_results[m]['psnr'] for m in methods]
                ssim_values = [metrics_results[m]['ssim'] for m in methods]
                
                fig.add_trace(go.Bar(name='PSNR (dB)', x=methods, y=psnr_values, marker_color='lightblue'))
                
                fig.update_layout(
                    title="Krahasimi i PSNR / PSNR Comparison",
                    xaxis_title="Metoda / Method",
                    yaxis_title="PSNR (dB)",
                    height=400
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
                # Best method
                best_method, best_score = metrics_eval.get_best_method(metrics_results, 'psnr')
                st.success(f"✨ Metoda më e mirë / Best Method: **{best_method}** (PSNR: {best_score:.2f} dB)")
                
                # Export options
                st.subheader("💾 Eksportoni rezultatet / Export Results")
                
                col_exp1, col_exp2 = st.columns(2)
                
                with col_exp1:
                    if st.button("📥 Shkarko të gjitha rezultatet / Download All Results"):
                        # Create ZIP file
                        zip_buffer = io.BytesIO()
                        with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zip_file:
                            # Add original
                            orig_bytes = cv2.imencode('.png', cv2.cvtColor(original_image, cv2.COLOR_RGB2BGR))[1].tobytes()
                            zip_file.writestr('original.png', orig_bytes)
                            
                            # Add noisy
                            noisy_bytes = cv2.imencode('.png', cv2.cvtColor(noisy_image, cv2.COLOR_RGB2BGR))[1].tobytes()
                            zip_file.writestr('noisy.png', noisy_bytes)
                            
                            # Add denoised
                            for method_name, denoised_img in denoised_results.items():
                                img_bytes = cv2.imencode('.png', cv2.cvtColor(denoised_img, cv2.COLOR_RGB2BGR))[1].tobytes()
                                zip_file.writestr(f'{method_name.lower()}.png', img_bytes)
                        
                        st.download_button(
                            label="📦 Shkarko ZIP",
                            data=zip_buffer.getvalue(),
                            file_name="denoising_results.zip",
                            mime="application/zip"
                        )
            
            elif "Single Method" in method_choice:
                # Single method processing
                method = st.sidebar.selectbox(
                    "Zgjidhni metodën / Select Method",
                    ["Median", "Wiener", "Wavelet", "DnCNN", "Hybrid"]
                )
                
                with st.spinner(f'🔄 Duke përpunuar me {method}...'):
                    if method == "Median":
                        denoised = classical.denoise_image(noisy_image, method='median', kernel_size=5)
                    elif method == "Wiener":
                        denoised = classical.denoise_image(noisy_image, method='wiener')
                    elif method == "Wavelet":
                        denoised = classical.denoise_image(noisy_image, method='wavelet')
                    elif method == "DnCNN":
                        denoised = dncnn.denoise(noisy_image)
                    else:  # Hybrid
                        denoised = hybrid.denoise_hybrid_cascade(noisy_image)
                
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.image(original_image, caption="Origjinal / Original", use_container_width=True)
                with col2:
                    st.image(noisy_image, caption="Me zhurmë / Noisy", use_container_width=True)
                with col3:
                    st.image(denoised, caption=f"Përpunuar / Denoised ({method})", use_container_width=True)
                
                # Metrics
                metrics = metrics_eval.evaluate_all_metrics(original_image, denoised, include_niqe=False)
                
                col_m1, col_m2, col_m3 = st.columns(3)
                col_m1.metric("PSNR", f"{metrics['psnr']:.2f} dB")
                col_m2.metric("SSIM", f"{metrics['ssim']:.4f}")
                col_m3.metric("MSE", f"{metrics['mse']:.2f}")
            
            else:  # Auto Recommend
                # Get recommendation
                method, reason, detected_noise, confidence = recommender.recommend_method(noisy_image)
                
                st.info(f"""
                **🎯 Rekomandimi automatik / Automatic Recommendation:**
                
                - **Zhurma e detektuar / Detected Noise:** {detected_noise.title()} (besueshmëri / confidence: {confidence:.2%})
                - **Metoda e rekomanduar / Recommended Method:** {method.title()}
                - **Arsyeja / Reason:** {reason}
                """)
                
                # Apply recommended method
                with st.spinner(f'🔄 Duke aplikuar {method}...'):
                    if method == 'median':
                        denoised = classical.denoise_image(noisy_image, method='median', kernel_size=5)
                    elif method == 'wiener':
                        denoised = classical.denoise_image(noisy_image, method='wiener')
                    elif method == 'wavelet':
                        denoised = classical.denoise_image(noisy_image, method='wavelet')
                    else:  # dncnn
                        denoised = dncnn.denoise(noisy_image)
                
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.image(original_image, caption="Origjinal / Original", use_container_width=True)
                with col2:
                    st.image(noisy_image, caption="Me zhurmë / Noisy", use_container_width=True)
                with col3:
                    st.image(denoised, caption=f"Përpunuar / Denoised ({method.title()})", use_container_width=True)
                
                # Metrics
                metrics = metrics_eval.evaluate_all_metrics(original_image, denoised, include_niqe=False)
                
                col_m1, col_m2, col_m3 = st.columns(3)
                col_m1.metric("PSNR", f"{metrics['psnr']:.2f} dB")
                col_m2.metric("SSIM", f"{metrics['ssim']:.4f}")
                col_m3.metric("MSE", f"{metrics['mse']:.2f}")
    
    else:
        st.info("👈 Ju lutemi ngarkoni një imazh ose zgjidhni një shembull / Please upload an image or select a sample")
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: gray;'>
        🎓 Projekti i Krahasimit të Metodave të Heqjes së Zhurmës nga Imazhet<br>
        Image Denoising Methods Comparison Project<br>
        📧 Kontakti / Contact: info@denoising-project.com
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
