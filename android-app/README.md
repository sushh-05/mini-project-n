# Android App Setup Guide

## Step-by-Step Instructions:

### 1. Open Android Studio
- File → New → New Project
- Select "Empty Views Activity"
- Name: `SareeBlouseMatcher`
- Package: `com.yourname.sareematcher`
- Language: Kotlin
- Minimum SDK: API 24 (Android 7.0)

### 2. Replace Files

Copy these files to your Android Studio project:

**MainActivity.kt** → `app/src/main/java/com/yourname/sareematcher/MainActivity.kt`
**activity_main.xml** → `app/src/main/res/layout/activity_main.xml`
**AndroidManifest.xml** → `app/src/main/AndroidManifest.xml`

### 3. Update the URL in MainActivity.kt

Line 36:
```kotlin
webView.loadUrl("http://172.16.103.162:8080/frontend.html")
```

Change to your actual URL (local network or deployed URL)

### 4. Build APK

1. Click **Build** menu → **Build Bundle(s) / APK(s)** → **Build APK(s)**
2. Wait 2-3 minutes
3. APK will be in: `app/build/outputs/apk/debug/app-debug.apk`

### 5. Install on Phone

**Option A:** Transfer APK via USB and install

**Option B:** In Android Studio:
- Connect phone via USB
- Enable USB Debugging on phone
- Click green "Run" button ▶️

### 6. For Production APK (signed for distribution):

**Build** → **Generate Signed Bundle / APK** → Follow wizard

---

## Features Included:

✅ Camera permission handling
✅ Full WebView with JavaScript
✅ Back button navigation
✅ Local storage support
✅ Camera access from WebView
✅ Offline capable (with service worker)

---

## Alternative: Faster Option with AppGyver/Thunkable

If Android Studio is too complex, try **no-code** builders:
- **Thunkable** - Drag & drop, add WebView component
- **AppGyver** - Free, export APK directly
- Takes 10 minutes instead of hours

Let me know which approach you prefer!
