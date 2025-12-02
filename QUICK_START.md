# 🚀 Quick Start Guide - Enhanced Dashboard

## Launch the Dashboard

### Option 1: Using start.py (Recommended)
```bash
python start.py
```
Then select option **3** (Open dashboard)

### Option 2: Direct Launch
```bash
python -m streamlit run dashboard_app.py
```

The dashboard will open in your browser at: `http://localhost:8501`

---

## 🎯 Quick Feature Guide

### 🏠 Home Page - Single Image Processing
1. **Upload or Select Image**
2. **Configure Noise** (Gaussian/Salt & Pepper/Speckle)
3. **Select Methods** (Median, Wiener, Wavelet, etc.)
4. **Click "Start Processing"**
5. **Download Results**

**Key Features:**
- ⏱️ Processing time estimate
- 📊 Histogram comparisons
- 📈 Detailed metric cards
- 💾 Download as ZIP/CSV

---

### 📦 Batch Processing - Multiple Images
1. **Upload Multiple Images**
2. **Configure Noise & Methods**
3. **Process All at Once**
4. **View Summary Statistics**
5. **Download Batch Results**

**Key Features:**
- 📊 Average metrics across all images
- 🖼️ Individual image expandable views
- 📦 Organized ZIP export

---

### 🖼️ Comparison Gallery - View History
1. **Browse All Processed Images**
2. **Filter by Method or Noise Type**
3. **Sort by PSNR or Timestamp**
4. **Download Individual Results**

**Key Features:**
- 📁 Complete processing history
- 🔍 Advanced filtering
- 📥 Individual downloads
- 🗑️ Clear history option

---

### 📊 Analytics - Performance Insights
1. **View Overall Performance Summary**
2. **Analyze Trends Over Time**
3. **Compare Method Rankings**
4. **Explore Metric Distributions**

**Key Features:**
- 📈 Interactive Plotly charts
- 📊 Box plots for distributions
- 🏆 Method ranking table
- 📉 Trend analysis

---

### ⚙️ Settings - Customize Your Experience
1. **Set Language (English/Albanian)**
2. **Configure Default Noise Settings**
3. **Enable/Disable Auto-Save**
4. **Export/Import History**

**Key Features:**
- 🌍 Language selection
- 🎨 Theme options
- 💾 Data management
- ℹ️ System information

---

## 📊 Understanding Metrics

| Metric | Range | Better | Good Value |
|--------|-------|--------|------------|
| PSNR   | 0-∞ dB | Higher | > 30 dB |
| SSIM   | 0-1   | Higher | > 0.9 |
| MSE    | 0-∞   | Lower  | < 100 |

---

## 🎨 Available Methods

### Classical (Always Available)
- **Median** - Best for salt & pepper noise
- **Wiener** - Adaptive filtering
- **Wavelet** - Multi-scale decomposition

### Deep Learning (Requires PyTorch)
- **DnCNN** - CNN-based denoising
- **Hybrid** - Classical + DNN

---

## 💾 Download Options

### Home Page:
- 📦 **All Results (ZIP)** - Images + metrics.json
- 📊 **Metrics (CSV)** - Comparison table
- ⭐ **Best Method** - Single best result

### Batch Processing:
- 📦 **Batch ZIP** - Organized folders per image
- 📄 **Summary JSON** - Aggregate statistics

### Gallery:
- 📥 **Individual Results** - Per-result downloads

---

## 🔧 Troubleshooting

### Dashboard Won't Start
```bash
# Install streamlit
pip install streamlit

# Try direct launch
python -m streamlit run dashboard_app.py
```

### Missing PyTorch Features
- DnCNN and Hybrid methods require PyTorch
- Classical methods work without PyTorch
- Consider using Python 3.11 or 3.12 for PyTorch support

### Performance Issues
- Close other applications
- Process fewer images in batch mode
- Reduce image resolution if needed

---

## 🎯 Common Workflows

### Workflow 1: Compare All Methods on One Image
1. Go to **Home** page
2. Upload image
3. Select **all available methods**
4. Enable "Show Histograms" and "Detailed Metrics"
5. Process and compare results

### Workflow 2: Batch Process Test Set
1. Go to **Batch Processing** page
2. Upload all test images
3. Select 2-3 methods for comparison
4. Process batch
5. Download organized results

### Workflow 3: Analyze Best Method
1. Process several images (Home or Batch)
2. Go to **Analytics** page
3. View method rankings
4. Analyze performance trends
5. Choose best method for your use case

### Workflow 4: Review Past Results
1. Go to **Comparison Gallery**
2. Filter by method or noise type
3. Sort by PSNR to find best results
4. Download specific results you need

---

## ⚡ Pro Tips

1. **Use Auto-Recommend** for automatic method selection
2. **Enable Auto-Save** to build analytics history
3. **Process multiple methods** to find the best for your data
4. **Export history** regularly for backup
5. **Use batch mode** for consistent comparison across datasets
6. **Check Analytics** to understand method consistency

---

## 📞 Need Help?

Check the comprehensive documentation: `DASHBOARD_ENHANCEMENTS.md`

---

**Enjoy your enhanced dashboard! 🎉**
