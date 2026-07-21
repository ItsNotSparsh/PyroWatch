# PyroWatch: Fire Detection System
## Corrected Presentation Content

---

## SLIDE 1: Title Slide
**PyroWatch: Fire Detection System**

SPARSH KUMAR - S24CSEU0428  
SHIVRAJ SINGH BHATI - S24CSEU0511  
SHIKHAR SHUKHA - S24CSEU412  
SAMAGRA GULATI - S24CSEU0451

**AI-POWERED WILDFIRE DETECTION, RISK ASSESSMENT, AND SPREAD PREDICTION — INTEGRATING DEEP LEARNING, SATELLITE DATA, AND WEATHER INTELLIGENCE.**

---

## SLIDE 2: The Problem

### Devastating Wildfire Impact
Wildfires burn 4 million km² annually, causing billions in damages and massive carbon emissions. Early detection is critical for effective response.

### Limited Detection Capabilities
Traditional systems rely on smoke detectors or manual surveillance — slow, localized, and unable to predict risk or spread patterns.

### Fragmented Data Sources
Satellite data, weather information, and image analysis exist in silos — no unified platform for comprehensive wildfire intelligence.

---

## SLIDE 3: Objectives

### **Detect Fire with AI Precision**
Use state-of-the-art Vision Transformer (ViT) deep learning to classify fire vs no-fire from satellite and aerial imagery with 96.3% accuracy.

### **Assess Ignition Risk in Real-Time**
Predict fire ignition probability using weather conditions (temperature, humidity, wind) and geospatial data with Random Forest ML.

### **Visualize Fire Spread & Hotspots**
Integrate NASA FIRMS satellite hotspot data and physics-based spread modeling to show where fires are active and where they'll spread.

---

## SLIDE 4: System Overview

```
┌─────────────┐      ┌──────────────┐      ┌─────────────┐
│   INPUT     │  →   │  PROCESSING  │  →   │   OUTPUT    │
├─────────────┤      ├──────────────┤      ├─────────────┤
│ • Images    │      │ • ViT Model  │      │ • Fire/     │
│ • Satellite │      │ • Random     │      │   No Fire   │
│   Hotspots  │      │   Forest     │      │ • Risk      │
│ • Weather   │      │ • Spread     │      │   Level     │
│   Data      │      │   Model      │      │ • Heatmaps  │
│ • Location  │      │ • Fusion     │      │ • Maps      │
└─────────────┘      └──────────────┘      └─────────────┘
```

PyroWatch is a **multimodal system** that fuses image classification, satellite monitoring, weather-based risk prediction, and fire spread visualization into a unified dashboard.

---

## SLIDE 5: Technology Stack

### **Deep Learning Framework**
- **PyTorch 2.0** + **Transformers 4.35** - Vision Transformer (ViT) for image classification
- **96.3% accuracy** on Wildfire Dataset (1,887 images)

### **Machine Learning**
- **Scikit-learn** - Random Forest (100 trees) for risk prediction
- **~95% accuracy** on weather-based fire risk assessment

### **Data Integration**
- **NASA FIRMS API** - Real-time MODIS/VIIRS satellite hotspot data
- **OpenWeatherMap API** - Live weather data (temp, humidity, wind)
- **Pandas + NumPy** - Data processing and analysis

### **Visualization & UI**
- **Streamlit** - Interactive web dashboard
- **Folium** - Interactive maps with hotspots and spread zones
- **Plotly** - Time series charts and analytics

---

## SLIDE 6: Working Flow

### **1. Image Upload & Analysis**
User uploads satellite/aerial image → ViT model processes → Fire/No Fire classification + confidence score + attention heatmap

### **2. Satellite Hotspot Monitoring**
Fetch NASA FIRMS MODIS data → Filter confidence ≥70% → Display active fire locations on interactive map

### **3. Weather-Based Risk Assessment**
Get location coordinates → Fetch live weather → Random Forest predicts High/Low ignition risk

### **4. Fire Spread Prediction**
Use wind speed & direction → Calculate geometric spread cone → Visualize danger zone on map

### **5. Unified Dashboard**
All components integrated in Streamlit interface with tabs: Command Center (map), AI Vision Lab (image analysis), Analytics (trends)

---

## SLIDE 7: Detection Logic - Vision Transformer

### **Architecture: ViT-Base-Patch16-224**
- Image divided into 16×16 patches (196 patches total)
- 12 transformer encoder layers with multi-head self-attention
- Fine-tuned on Wildfire Dataset (Kaggle)

### **Training Results**
- **96.3% test accuracy** after 2 epochs
- Training loss: 0.76 → 0.03
- Dataset: 1,509 training images, 378 test images

### **Attention Heatmap Explainability**
- Extracts attention weights from final layer
- Visualizes which image regions the model focuses on
- Generates heatmap in 0.12 seconds
- Helps users understand AI decisions

---

## SLIDE 8: Integration Pipeline

```
┌──────────────────────────────────────────────────────┐
│                    STREAMLIT UI                      │
│  ┌────────────┐  ┌────────────┐  ┌────────────┐    │
│  │  Command   │  │  AI Vision │  │ Analytics  │    │
│  │  Center    │  │    Lab     │  │    Tab     │    │
│  └────────────┘  └────────────┘  └────────────┘    │
└──────────────────────────────────────────────────────┘
                         ↕
┌──────────────────────────────────────────────────────┐
│              BACKEND PROCESSING LAYER                │
│  ┌────────────┐  ┌────────────┐  ┌────────────┐    │
│  │ ViT Image  │  │  Random    │  │  Spread    │    │
│  │ Classifier │  │  Forest    │  │  Model     │    │
│  │ (PyTorch)  │  │  (Sklearn) │  │ (Geometry) │    │
│  └────────────┘  └────────────┘  └────────────┘    │
└──────────────────────────────────────────────────────┘
                         ↕
┌──────────────────────────────────────────────────────┐
│                  DATA SOURCES                        │
│  ┌────────────┐  ┌────────────┐  ┌────────────┐    │
│  │   NASA     │  │ OpenWeather│  │   User     │    │
│  │   FIRMS    │  │    Map     │  │  Uploads   │    │
│  │  (MODIS)   │  │    API     │  │  (Images)  │    │
│  └────────────┘  └────────────┘  └────────────┘    │
└──────────────────────────────────────────────────────┘
```

**Late Fusion Architecture** - Each modality operates independently, results combined at decision level for comprehensive wildfire intelligence.

---

## SLIDE 9: Testing & Results

### **Vision Transformer Performance**
- **96.3% accuracy** on test set (378 images)
- Correctly classifies fire and no-fire images
- Attention heatmaps show model focuses on flames, smoke, and orange/red regions

### **Random Forest Risk Model**
- **~95% accuracy** on synthetic weather data
- Successfully correlates high temp + low humidity + high wind = high risk
- Real-time prediction in <1 second

### **NASA FIRMS Integration**
- Successfully fetches 1-3 days of global hotspot data
- Filters to confidence ≥70% to reduce false positives
- Displays 12,000+ active fire locations on interactive map

### **End-to-End Performance**
- Complete analysis (image + risk + map) in <10 seconds
- Attention heatmap generation: 0.12 seconds
- Real-time weather and satellite data integration

---

## SLIDE 10: Key Features Implemented

### **✅ AI Vision Lab**
- Upload images for fire detection
- 96.3% accurate ViT classification
- Attention heatmap visualization
- Confidence scores for predictions
- Alert history log

### **✅ Command Center**
- Interactive map with fire hotspots
- Weather-based risk prediction
- Fire spread cone visualization
- Live weather data integration
- Place name search with geocoding

### **✅ Analytics Dashboard**
- Historical hotspot trends (time series)
- Confidence distribution charts
- Summary statistics (total hotspots, peak days)
- Multi-day data fetching (1-3 days)

### **✅ Advanced Features**
- Heatmap download as PNG
- PDF report generation
- Fire spread timeline animation
- Coordinate search by place name

---

## SLIDE 11: Conclusion & Future Scope

### **What We Built**
✅ **Multimodal wildfire intelligence system** integrating 4 data sources  
✅ **96.3% accurate** Vision Transformer for fire image classification  
✅ **Real-time satellite integration** with NASA FIRMS MODIS data  
✅ **Weather-based risk prediction** using Random Forest ML  
✅ **Interactive dashboard** with maps, charts, and visualizations  
✅ **Attention heatmap explainability** for AI transparency  

### **What Comes Next**
🔮 **Enhanced spread modeling** - Integrate Rothermel physics-based fire behavior  
🔮 **Historical fire database** - Train on real MODIS archive data (not synthetic)  
🔮 **Multi-class classification** - Detect fire intensity levels (low/medium/high)  
🔮 **Video stream processing** - Real-time analysis of drone/surveillance feeds  
🔮 **Mobile application** - iOS/Android app for field use by firefighters  
🔮 **Smoke detection** - Early warning before visible flames appear  

**PyroWatch demonstrates the power of multimodal AI for wildfire management — combining deep learning, satellite remote sensing, and meteorological intelligence into a unified decision support platform.**

---

## SLIDE 12: System Architecture Diagram

```
┌─────────────────────────────────────────────────────────┐
│                  PYROWATCH SYSTEM                       │
├─────────────────────────────────────────────────────────┤
│                                                         │
│  ┌──────────────┐  ┌──────────────┐  ┌─────────────┐  │
│  │   IMAGE      │  │   SATELLITE  │  │   WEATHER   │  │
│  │   MODULE     │  │    MODULE    │  │   MODULE    │  │
│  ├──────────────┤  ├──────────────┤  ├─────────────┤  │
│  │ • ViT Model  │  │ • MODIS API  │  │ • Weather   │  │
│  │ • Attention  │  │ • Hotspot    │  │   Fetcher   │  │
│  │   Heatmap    │  │   Filter     │  │ • Risk RF   │  │
│  │ • 96.3% Acc  │  │ • Confidence │  │ • ~95% Acc  │  │
│  └──────────────┘  └──────────────┘  └─────────────┘  │
│         ↓                  ↓                  ↓         │
│  ┌─────────────────────────────────────────────────┐  │
│  │           FUSION & VISUALIZATION ENGINE         │  │
│  ├─────────────────────────────────────────────────┤  │
│  │ • Streamlit Dashboard                           │  │
│  │ • Folium Interactive Maps                       │  │
│  │ • Plotly Analytics Charts                       │  │
│  │ • Fire Spread Geometric Model                   │  │
│  │ • PDF Report Generation                         │  │
│  └─────────────────────────────────────────────────┘  │
│                                                         │
└─────────────────────────────────────────────────────────┘
```

---

## SLIDE 13: Live Demo Screenshots

### **Command Center Tab**
- Interactive map showing global fire hotspots
- Weather metrics (temperature, humidity, wind)
- Risk prediction badge (High/Low)
- Fire spread cone visualization

### **AI Vision Lab Tab**
- Image upload interface
- Fire detection results with confidence
- Side-by-side attention heatmap comparison
- Alert history log

### **Analytics Tab**
- Daily hotspot trend line chart
- Confidence distribution histogram
- Summary statistics cards
- Date range selector

---

## SLIDE 14: Technical Achievements

### **Model Performance**
- Vision Transformer: **96.3% accuracy**
- Random Forest: **~95% accuracy**
- Attention heatmap: **0.12 seconds**
- End-to-end latency: **<10 seconds**

### **Data Integration**
- NASA FIRMS: **Real-time** global coverage
- OpenWeatherMap: **Live** weather updates
- Kaggle Dataset: **1,887 images** (730 fire, 1,157 no-fire)

### **System Capabilities**
- **Multimodal fusion** of 4 data sources
- **Explainable AI** with attention visualization
- **Interactive dashboard** with 3 tabs
- **Production-ready** with error handling

---

## SLIDE 15: Thank You

### **PyroWatch: Fire Detection System**

**Team Members:**
- SPARSH KUMAR - S24CSEU0428
- SHIVRAJ SINGH BHATI - S24CSEU0511
- SHIKHAR SHUKHA - S24CSEU412
- SAMAGRA GULATI - S24CSEU0451

**Key Achievements:**
✅ 96.3% fire detection accuracy  
✅ Real-time satellite & weather integration  
✅ Multimodal AI fusion architecture  
✅ Interactive web dashboard  

**Questions?**

---

