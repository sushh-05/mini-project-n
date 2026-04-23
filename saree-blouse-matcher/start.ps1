# Saree Blouse Matcher - PowerShell Startup Script

Write-Host "========================================" -ForegroundColor Cyan
Write-Host "Saree Blouse Matcher - Quick Start" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""

Set-Location saree-blouse-matcher

# Check if virtual environment exists
if (-not (Test-Path "venv")) {
    Write-Host "Creating virtual environment..." -ForegroundColor Yellow
    python -m venv venv
    Write-Host ""
}

# Activate virtual environment
Write-Host "Activating virtual environment..." -ForegroundColor Yellow
& "venv\Scripts\Activate.ps1"
Write-Host ""

# Install dependencies
Write-Host "Installing dependencies..." -ForegroundColor Yellow
pip install -r requirements.txt
Write-Host ""

# Run setup check
Write-Host "Running setup check..." -ForegroundColor Yellow
python setup_check.py
Write-Host ""

# Ask if user wants to extract embeddings
$extract = Read-Host "Extract embeddings now? (y/n)"
if ($extract -eq "y" -or $extract -eq "Y") {
    Write-Host ""
    Write-Host "Extracting embeddings..." -ForegroundColor Yellow
    python scripts\extract_embeddings.py
    Write-Host ""
}

# Ask if user wants to create composite
$composite = Read-Host "Create composite test image? (y/n)"
if ($composite -eq "y" -or $composite -eq "Y") {
    Write-Host ""
    Write-Host "Creating composite..." -ForegroundColor Yellow
    python scripts\create_composite.py
    Write-Host ""
}

Write-Host "========================================" -ForegroundColor Green
Write-Host "Setup Complete!" -ForegroundColor Green
Write-Host "========================================" -ForegroundColor Green
Write-Host ""
Write-Host "Next steps:" -ForegroundColor Cyan
Write-Host "1. Start API server: uvicorn api:app --reload" -ForegroundColor White
Write-Host "2. Open frontend.html in your browser" -ForegroundColor White
Write-Host ""

$start = Read-Host "Start API server now? (y/n)"
if ($start -eq "y" -or $start -eq "Y") {
    Write-Host ""
    Write-Host "Starting API server..." -ForegroundColor Yellow
    Write-Host "Visit: http://localhost:8000/docs" -ForegroundColor Cyan
    Write-Host "Frontend: Open frontend.html in your browser" -ForegroundColor Cyan
    Write-Host ""
    uvicorn api:app --host 0.0.0.0 --port 8000 --reload
}
