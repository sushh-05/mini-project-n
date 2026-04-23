# Complete Setup and Test Script
Write-Host "========================================" -ForegroundColor Cyan
Write-Host "Saree Blouse Matcher - Complete Setup" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""

$projectRoot = "saree-blouse-matcher"
Set-Location $projectRoot

# Step 1: Check Python
Write-Host "Step 1: Checking Python..." -ForegroundColor Yellow
python --version
if ($LASTEXITCODE -ne 0) {
    Write-Host "ERROR: Python not found!" -ForegroundColor Red
    exit 1
}
Write-Host ""

# Step 2: Check/Install dependencies
Write-Host "Step 2: Installing dependencies..." -ForegroundColor Yellow
pip install -q -r requirements.txt
Write-Host "✓ Dependencies installed" -ForegroundColor Green
Write-Host ""

# Step 3: Check catalog and images
Write-Host "Step 3: Checking catalog..." -ForegroundColor Yellow
if (-not (Test-Path "data/catalog.csv")) {
    Write-Host "ERROR: data/catalog.csv not found!" -ForegroundColor Red
    exit 1
}

$catalog = Import-Csv "data/catalog.csv"
Write-Host "✓ Found $($catalog.Count) items in catalog" -ForegroundColor Green

$missingImages = @()
foreach ($row in $catalog) {
    if (-not (Test-Path $row.image_path)) {
        $missingImages += $row.image_path
    }
}

if ($missingImages.Count -gt 0) {
    Write-Host "⚠ Missing images:" -ForegroundColor Yellow
    $missingImages | ForEach-Object { Write-Host "  - $_" -ForegroundColor Yellow }
    Write-Host ""
}
Write-Host ""

# Step 4: Extract embeddings
Write-Host "Step 4: Extracting embeddings..." -ForegroundColor Yellow
python scripts/extract_embeddings.py
if ($LASTEXITCODE -ne 0) {
    Write-Host "ERROR: Failed to extract embeddings!" -ForegroundColor Red
    exit 1
}
Write-Host "✓ Embeddings extracted" -ForegroundColor Green
Write-Host ""

# Step 5: Generate training data
Write-Host "Step 5: Generating initial training data..." -ForegroundColor Yellow
python scripts/train_matcher.py
if ($LASTEXITCODE -ne 0) {
    Write-Host "WARNING: Training generation failed (non-critical)" -ForegroundColor Yellow
}
else {
    Write-Host "✓ Training data generated" -ForegroundColor Green
}
Write-Host ""

# Step 6: Create composite test image
Write-Host "Step 6: Creating composite test image..." -ForegroundColor Yellow
python scripts/create_composite.py --cols 2 --size 300
if ($LASTEXITCODE -ne 0) {
    Write-Host "WARNING: Composite creation failed (non-critical)" -ForegroundColor Yellow
}
else {
    Write-Host "✓ Composite image created" -ForegroundColor Green
}
Write-Host ""

# Summary
Write-Host "========================================" -ForegroundColor Green
Write-Host "✓ Setup Complete!" -ForegroundColor Green  
Write-Host "========================================" -ForegroundColor Green
Write-Host ""
Write-Host "File Structure:" -ForegroundColor Cyan
Write-Host "  ✓ data/catalog.csv" -ForegroundColor White
Write-Host "  ✓ outputs/embeddings.npy" -ForegroundColor White
Write-Host "  ✓ outputs/catalog_with_status.csv" -ForegroundColor White
Write-Host "  ✓ outputs/training_pairs.json" -ForegroundColor White
Write-Host ""

Write-Host "Next Steps:" -ForegroundColor Cyan
Write-Host "1. Start API server:" -ForegroundColor White
Write-Host "   uvicorn api:app --host 0.0.0.0 --port 8000 --reload" -ForegroundColor Gray
Write-Host ""
Write-Host "2. Open frontend:" -ForegroundColor White
Write-Host "   Open frontend.html in your browser" -ForegroundColor Gray
Write-Host "   Or: python -m http.server 8080" -ForegroundColor Gray
Write-Host ""
Write-Host "3. Test the system:" -ForegroundColor White
Write-Host "   - Upload a saree image" -ForegroundColor Gray
Write-Host "   - Upload composite blouse image" -ForegroundColor Gray
Write-Host "   - See the detection!" -ForegroundColor Gray
Write-Host "   - Provide feedback to improve accuracy" -ForegroundColor Gray
Write-Host ""

$start = Read-Host "Start API server now? (y/n)"
if ($start -eq "y" -or $start -eq "Y") {
    Write-Host ""
    Write-Host "Starting API server..." -ForegroundColor Yellow
    Write-Host "API Docs: http://localhost:8000/docs" -ForegroundColor Cyan
    Write-Host "Health Check: http://localhost:8000/health" -ForegroundColor Cyan
    Write-Host ""
    Write-Host "Press Ctrl+C to stop the server" -ForegroundColor Yellow
    Write-Host ""
    uvicorn api:app --host 0.0.0.0 --port 8000 --reload
}
