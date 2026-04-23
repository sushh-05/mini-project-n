# 📱 Android App Deployment Guide

## ✅ Your app is now PWA-ready!

### **Step 1: Test Locally (Now)**

1. Make sure your servers are running:
   ```powershell
   # Terminal 1 - Backend API
   cd saree-blouse-matcher
   python api.py
   
   # Terminal 2 - Frontend (from project root)
   python -m http.server 8080 --directory saree-blouse-matcher
   ```

2. Open Chrome on your Android phone
3. Visit: `http://YOUR_PC_IP:8080/frontend.html`
   - Find your PC IP: Run `ipconfig` and look for IPv4 Address
   - Make sure phone is on same WiFi network

4. Tap the three dots (⋮) → **"Add to Home Screen"**
5. Done! App icon appears on your home screen 🎉

---

## 🌐 Step 2: Deploy to Internet (Free)

### **Option A: Render.com (Recommended - Free)**

1. Create account at [render.com](https://render.com)

2. Create `render.yaml` in your project:
   ```yaml
   services:
     - type: web
       name: saree-matcher-api
       runtime: python
       buildCommand: pip install -r requirements.txt
       startCommand: uvicorn api:app --host 0.0.0.0 --port $PORT
   ```

3. Push to GitHub and connect to Render
4. Get your URL: `https://your-app.onrender.com`

### **Option B: Railway.app (Also Free)**

1. Create account at [railway.app](https://railway.app)
2. Connect your GitHub repo
3. Railway auto-detects Python and deploys!

### **Option C: PythonAnywhere (Free tier)**

1. Sign up at [pythonanywhere.com](https://www.pythonanywhere.com)
2. Upload your code
3. Configure WSGI for FastAPI

---

## 📦 Step 3: Update Frontend URL

After deployment, edit `frontend.html`:
```javascript
const API_URL = 'https://your-deployed-url.com';  // Change this!
```

---

## 🎯 Installation on Android

1. Open Chrome on Android
2. Visit your deployed URL
3. Tap **"Add to Home Screen"**
4. Icon appears like a native app!

**Features:**
- ✅ Works offline (cached)
- ✅ Full camera access
- ✅ Splash screen
- ✅ No app store needed!

---

## 🔧 Need a Real Android App? (Optional)

If you need Play Store distribution, I can help create an Android WebView wrapper.

---

## 📝 Current Status

- ✅ PWA manifest created
- ✅ Service worker added
- ✅ Install prompt ready
- ⏳ Icons needed (using placeholders)
- ⏳ Deploy backend to cloud

**Next: Deploy backend so anyone can access it!**
