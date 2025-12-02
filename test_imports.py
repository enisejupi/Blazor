"""
Test script to verify all available imports work correctly
"""
import sys
import os

print("Testing imports...")
print("=" * 60)

# Test basic scientific libraries
try:
    import numpy as np
    print(f"✓ NumPy {np.__version__}")
except Exception as e:
    print(f"✗ NumPy failed: {e}")

try:
    import cv2
    print(f"✓ OpenCV {cv2.__version__}")
except Exception as e:
    print(f"✗ OpenCV failed: {e}")

try:
    import scipy
    print(f"✓ SciPy {scipy.__version__}")
except Exception as e:
    print(f"✗ SciPy failed: {e}")

try:
    from PIL import Image
    import PIL
    print(f"✓ Pillow {PIL.__version__}")
except Exception as e:
    print(f"✗ Pillow failed: {e}")

try:
    import pywt
    print(f"✓ PyWavelets {pywt.__version__}")
except Exception as e:
    print(f"✗ PyWavelets failed: {e}")

try:
    import yaml
    print(f"✓ PyYAML")
except Exception as e:
    print(f"✗ PyYAML failed: {e}")

try:
    import matplotlib
    print(f"✓ Matplotlib {matplotlib.__version__}")
except Exception as e:
    print(f"✗ Matplotlib failed: {e}")

try:
    import plotly
    print(f"✓ Plotly {plotly.__version__}")
except Exception as e:
    print(f"✗ Plotly failed: {e}")

try:
    import pandas as pd
    print(f"✓ Pandas {pd.__version__}")
except Exception as e:
    print(f"✗ Pandas failed: {e}")

try:
    import tqdm
    print(f"✓ tqdm {tqdm.__version__}")
except Exception as e:
    print(f"✗ tqdm failed: {e}")

try:
    import reportlab
    print(f"✓ ReportLab")
except Exception as e:
    print(f"✗ ReportLab failed: {e}")

try:
    import pypdf
    print(f"✓ PyPDF {pypdf.__version__}")
except Exception as e:
    print(f"✗ PyPDF failed: {e}")

print("=" * 60)

# Test packages that need compilation (expected to fail on Python 3.14)
print("\nPackages requiring compilation (may not be available):")
print("=" * 60)

try:
    import torch
    print(f"✓ PyTorch {torch.__version__}")
except Exception as e:
    print(f"✗ PyTorch not available: {e}")

try:
    import streamlit
    print(f"✓ Streamlit {streamlit.__version__}")
except Exception as e:
    print(f"✗ Streamlit not available: {e}")

try:
    import skimage
    print(f"✓ scikit-image {skimage.__version__}")
except Exception as e:
    print(f"✗ scikit-image not available: {e}")

print("=" * 60)

# Test project modules
print("\nTesting project modules:")
print("=" * 60)

sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

try:
    from src.classical_methods import ClassicalDenoisers
    print("✓ src.classical_methods")
except Exception as e:
    print(f"✗ src.classical_methods failed: {e}")

try:
    from src.data_loader import NoiseGenerator
    print("✓ src.data_loader")
except Exception as e:
    print(f"✗ src.data_loader failed: {e}")

try:
    from src.metrics import MetricsEvaluator
    print("✓ src.metrics")
except Exception as e:
    print(f"✗ src.metrics failed: {e}")

try:
    from src.visualization import Visualizer
    print("✓ src.visualization")
except Exception as e:
    print(f"✗ src.visualization failed: {e}")

try:
    from src.hybrid_methods import HybridDenoiser, MethodRecommender
    print("✓ src.hybrid_methods")
except Exception as e:
    print(f"✗ src.hybrid_methods failed: {e}")

try:
    from src.report_generator import PDFReportGenerator
    print("✓ src.report_generator")
except Exception as e:
    print(f"✗ src.report_generator failed: {e}")

try:
    from src.dncnn_model import DnCNNTrainer
    print("✓ src.dncnn_model (requires PyTorch)")
except Exception as e:
    print(f"✗ src.dncnn_model not available (requires PyTorch): {type(e).__name__}")

print("=" * 60)
print("\nImport test complete!")
