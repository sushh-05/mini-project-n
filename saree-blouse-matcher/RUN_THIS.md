# 🚀 QUICK TEST GUIDE - Run This!

## ✅ Step-by-Step Testing Instructions

### 1️⃣ First Time Setup (5 minutes)

```powershell
# Navigate to project folder
cd saree-blouse-matcher

# Install dependencies (if not done)
pip install -r requirements.txt

# Extract embeddings from your images
python scripts/extract_embeddings.py

# Generate initial training data
python scripts/train_matcher.py
```

**Expected Output:**
```
✓ Saved embeddings to: outputs/embeddings.npy
✓ Saved filtered catalog to: outputs/catalog_with_status.csv
✓ Created X training pairs
```

### 2️⃣ Start the API Server

```powershell
# Start the FastAPI server
uvicorn api:app --host 0.0.0.0 --port 8000 --reload
```

**You should see:**
```
INFO:     Started server process
INFO:     Uvicorn running on http://0.0.0.0:8000
✓ Loaded custom YOLO model (or pretrained)
✓ Loaded X training pairs
```

**Keep this terminal running!**

### 3️⃣ Open the Frontend

**Option A: Direct Open (Simple)**
```powershell
# Open in default browser
start frontend.html
```

**Option B: HTTP Server (Better for camera)**
```powershell
# In a NEW terminal:
python -m http.server 8080

# Then visit: http://localhost:8080/frontend.html
```

### 4️⃣ Test the System

1. **Upload Reference Saree**
   - Click "Choose File" under Step 1
   - Select: `data/saree-green.jpeg`
   - Click **"Store Saree Image"**
   - Wait for success ✓

2. **Create Test Composite** (optional)
   ```powershell
   # In a NEW terminal:
   python scripts/create_composite.py --cols 2
   # Creates: data/composite_blouses.jpg
   ```

3. **Detect Matching Blouse**
   - Choose detection method: **YOLO** or **Grid**
   - Upload composite image (or individual blouse image)
   - Click **"Detect Matching Blouse"**
   - See green box around matched blouse! 🎯

4. **Provide Feedback**
   - Click **✓ Good Match** or **✗ Wrong Match**
   - System learns from your feedback
   - Check training stats below

## 🎯 Expected Results

### With Your Images:
- **Black saree** → Gold blouse (90%+)
- **Green saree** → Gold/Green blouse (85%+)  
- **Purple saree** → Gold/Purple blouse (85%+)
- **Silver saree** → Silver blouse (95%+)

## 🔍 Verify Everything Works

### Test API Directly:
```powershell
# Health check
curl http://localhost:8000/health

# Expected response:
# {
#   "status": "ok",
#   "yolo_available": true,
#   "detection_methods": ["grid", "yolo"]
# }

# View training stats
curl http://localhost:8000/training_stats
```

### Test with Sample Images:
```powershell
# Store a saree
curl -X POST http://localhost:8000/store_saree -F "file=@data/saree-green.jpeg"

# Note the "saree_id" from response, then:
curl -X POST http://localhost:8000/detect_blouse_yolo ^
  -F "file=@data/blouse-gold.jpeg" ^
  -F "saree_id=YOUR_SAREE_ID_HERE" ^
  --output result.jpg

# View result
start result.jpg
```

## 🐛 Troubleshooting

### "Missing file: outputs/catalog_with_status.csv"
```powershell
python scripts/extract_embeddings.py
```

### "YOLO model not available"
```powershell
pip install ultralytics opencv-python
# Restart API
```

### "Cannot connect to API"
- Make sure API is running: `uvicorn api:app --reload`
- Check port 8000 is not in use
- Visit: http://localhost:8000/docs

### "Import errors" in editor
- These are IDE warnings, NOT real errors
- Python can't find packages until installed
- Run: `pip install -r requirements.txt`
- Restart VS Code if needed

### Frontend not loading
- Use HTTP server: `python -m http.server 8080`
- Don't use `file://` protocol
- Visit: http://localhost:8080/frontend.html

## 📊 What Each File Does

| File | Purpose | When to Run |
|------|---------|-------------|
| `api.py` | FastAPI server | Always (via uvicorn) |
| `frontend.html` | Web UI | Open in browser |
| `scripts/extract_embeddings.py` | Generate embeddings | Once, or when catalog changes |
| `scripts/train_matcher.py` | Create training pairs | Once, or to reset training |
| `scripts/create_composite.py` | Make test grid | For testing only |
| `scripts/train_yolo.py` | Train YOLO model | Optional, for custom model |

## ✅ Complete Test Checklist

- [ ] Installed dependencies: `pip install -r requirements.txt`
- [ ] Extracted embeddings: `python scripts/extract_embeddings.py`
- [ ] Generated training: `python scripts/train_matcher.py`
- [ ] Started API: `uvicorn api:app --reload`
- [ ] Opened frontend: `start frontend.html`
- [ ] Uploaded saree and got it stored ✓
- [ ] Detected blouse and saw green box ✓
- [ ] Provided feedback (Good/Bad match) ✓
- [ ] Checked training stats ✓

## 🎉 Success!

If all steps work, you have:
- ✅ Working object detection (YOLO + Grid)
- ✅ AI-powered matching (CLIP + metadata)
- ✅ Learning system (improves with feedback)
- ✅ Web UI with camera support
- ✅ Production-ready API

## 📚 Next Steps

1. **Test with More Images**
   - Try different saree-blouse combinations
   - Provide feedback to train the system

2. **Use Camera Detection**
   - Start with HTTP server
   - Click "Use Camera" button
   - Point at real blouses

3. **Train Custom YOLO**
   - Download fashion dataset
   - Label your own blouses
   - Run `python scripts/train_yolo.py`

4. **Deploy to Production**
   - Add more catalog items
   - Collect user feedback
   - Fine-tune for your market

---

**Having issues?** Check:
- [QUICKSTART.md](QUICKSTART.md) - Detailed setup
- [HOW_IT_WORKS.md](HOW_IT_WORKS.md) - Technical explanation
- [DATASETS.md](DATASETS.md) - Training data sources
- [YOLO_ADDED.md](YOLO_ADDED.md) - YOLO guide

**Still stuck?** Run: `python setup_check.py` for diagnostics
