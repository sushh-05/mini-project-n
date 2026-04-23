# Saree Blouse Matcher - Object Detection System

## Overview
AI-powered system that detects and highlights matching blouses for sarees in real-time using computer vision.

## Features
- 🎯 **Object Detection**: Detects matching blouses in images with multiple items
- 🎨 **Color Analysis**: Extracts and analyzes color palettes using K-means clustering
- 🤖 **AI Matching**: Uses CLIP embeddings + metadata scoring for hybrid recommendations
- 📷 **Real-time Camera**: Supports webcam input for live detection
- 🖼️ **Composite Testing**: Grid-based detection for testing with multiple blouses

## Quick Start

### 1. Install Dependencies
```bash
cd saree-blouse-matcher
pip install -r requirements.txt
```

### 2. Prepare Your Catalog
Edit `data/catalog.csv` with your saree and blouse items.

### 3. Extract Embeddings
```bash
python scripts/extract_embeddings.py
```

### 4. (Optional) Create Composite Test Image
```bash
python scripts/create_composite.py --cols 3 --size 300
```

### 5. Start the API
```bash
uvicorn api:app --host 0.0.0.0 --port 8000 --reload
```

### 6. Open Frontend
Open `frontend.html` in your browser
