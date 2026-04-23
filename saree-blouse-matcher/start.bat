@echo off
echo ========================================
echo Saree Blouse Matcher - Quick Start
echo ========================================
echo.

cd saree-blouse-matcher

REM Check if virtual environment exists
if not exist "venv" (
    echo Creating virtual environment...
    python -m venv venv
    echo.
)

REM Activate virtual environment
echo Activating virtual environment...
call venv\Scripts\activate.bat
echo.

REM Install dependencies
echo Installing dependencies...
pip install -r requirements.txt
echo.

REM Run setup check
echo Running setup check...
python setup_check.py
echo.

REM Ask if user wants to extract embeddings
set /p EXTRACT="Extract embeddings now? (y/n): "
if /i "%EXTRACT%"=="y" (
    echo.
    echo Extracting embeddings...
    python scripts\extract_embeddings.py
    echo.
)

REM Ask if user wants to create composite
set /p COMPOSITE="Create composite test image? (y/n): "
if /i "%COMPOSITE%"=="y" (
    echo.
    echo Creating composite...
    python scripts\create_composite.py
    echo.
)

echo ========================================
echo Setup Complete!
echo ========================================
echo.
echo Next steps:
echo 1. Start API server: uvicorn api:app --reload
echo 2. Open frontend.html in your browser
echo.
echo Press any key to start the API server now...
pause > nul

echo.
echo Starting API server...
echo Visit: http://localhost:8000/docs
echo Open frontend: http://localhost:8000/../frontend.html
echo.
uvicorn api:app --host 0.0.0.0 --port 8000 --reload
