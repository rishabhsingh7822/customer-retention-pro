#!/usr/bin/env python3
"""
RETAIN Platform - Automated Diagnostic Script
Run this to check if everything is configured correctly
"""

import os
import sys

def color_text(text, color_code):
    """Add color to terminal output"""
    return f"\033[{color_code}m{text}\033[0m"

def green(text): return color_text(text, "92")
def red(text): return color_text(text, "91")
def yellow(text): return color_text(text, "93")
def blue(text): return color_text(text, "94")

print("\n" + "="*70)
print(blue("RETAIN PLATFORM - DIAGNOSTIC TOOL"))
print("="*70 + "\n")

# Test 1: Python version
print("[1/8] Checking Python version...")
py_version = sys.version_info
if py_version.major == 3 and py_version.minor >= 8:
    print(green(f"  ✓ Python {py_version.major}.{py_version.minor}.{py_version.micro}"))
else:
    print(red(f"  ✗ Python {py_version.major}.{py_version.minor} (need 3.8+)"))

# Test 2: Required files
print("\n[2/8] Checking required files...")
required_files = {
    'main.py': 'Streamlit dashboard',
    'api.py': 'FastAPI backend',
    '.env': 'Environment variables',
    '.streamlit/secrets.toml': 'Streamlit secrets',
    'models/xgboost_churn.pkl': 'ML model',
    'models/model_metadata.json': 'Model metadata',
    'models/customer_database.csv': 'Customer data'
}

missing_files = []
for filepath, description in required_files.items():
    if os.path.exists(filepath):
        print(green(f"  ✓ {filepath}"))
    else:
        print(yellow(f"  ⚠ {filepath} (missing - {description})"))
        missing_files.append(filepath)

# Test 3: Check .env format
print("\n[3/8] Checking .env configuration...")
if os.path.exists('.env'):
    with open('.env', 'r') as f:
        env_content = f.read()
    
    if 'GEMINI_API_KEY' in env_content:
        print(green("  ✓ .env contains GEMINI_API_KEY"))
        
        # Check format
        for line in env_content.split('\n'):
            if line.strip().startswith('GEMINI_API_KEY'):
                if '=' in line:
                    key_value = line.split('=', 1)[1].strip().strip('"').strip("'")
                    if key_value.startswith('AIza'):
                        print(green(f"  ✓ Key format looks valid (starts with AIza)"))
                        print(green(f"  ✓ Key length: {len(key_value)} characters"))
                    elif key_value == 'your_api_key_here':
                        print(red("  ✗ Still using placeholder - replace with real key"))
                    else:
                        print(yellow(f"  ⚠ Key doesn't start with 'AIza' (may be invalid)"))
                else:
                    print(red("  ✗ Invalid format (missing '=')"))
    else:
        print(red("  ✗ .env doesn't contain GEMINI_API_KEY"))
else:
    print(yellow("  ⚠ .env file not found"))

# Test 4: Check secrets.toml format
print("\n[4/8] Checking .streamlit/secrets.toml...")
if os.path.exists('.streamlit/secrets.toml'):
    with open('.streamlit/secrets.toml', 'r') as f:
        secrets_content = f.read()
    
    if 'GEMINI_API_KEY' in secrets_content:
        print(green("  ✓ secrets.toml contains GEMINI_API_KEY"))
        
        # Check format
        if secrets_content.strip().startswith('['):
            print(yellow("  ⚠ May have section headers (should be flat)"))
        
        for line in secrets_content.split('\n'):
            if 'GEMINI_API_KEY' in line and '=' in line:
                key_value = line.split('=', 1)[1].strip().strip('"').strip("'")
                if key_value.startswith('AIza'):
                    print(green(f"  ✓ Key format looks valid"))
    else:
        print(yellow("  ⚠ secrets.toml doesn't contain GEMINI_API_KEY"))
else:
    print(yellow("  ⚠ .streamlit/secrets.toml not found"))

# Test 5: Check if main.py is fixed version
print("\n[5/8] Checking main.py version...")
if os.path.exists('main.py'):
    with open('main.py', 'r', encoding='utf-8') as f:
        main_content = f.read()
    
    # Check for old error message
    if 'st.error("⚠️ GEMINI_API_KEY missing' in main_content:
        print(red("  ✗ OLD VERSION detected!"))
        print(red("  ✗ Shows error on every page"))
        print(yellow("  → Replace with FIXED version from download"))
    elif 'ai_active = configure_genai()' in main_content:
        print(green("  ✓ FIXED VERSION detected"))
        print(green("  ✓ No annoying errors on every page"))
    else:
        print(yellow("  ⚠ Unable to determine version"))
else:
    print(red("  ✗ main.py not found"))

# Test 6: Check if api.py has CORS
print("\n[6/8] Checking api.py configuration...")
if os.path.exists('api.py'):
    with open('api.py', 'r', encoding='utf-8') as f:
        api_content = f.read()
    
    if 'app.add_middleware' in api_content and 'CORSMiddleware' in api_content:
        print(green("  ✓ CORS middleware configured"))
    else:
        print(yellow("  ⚠ CORS middleware not configured"))
        print(yellow("  → Replace with FIXED version if you need API"))
else:
    print(yellow("  ⚠ api.py not found"))

# Test 7: Check installed packages
print("\n[7/8] Checking installed packages...")
packages = {
    'streamlit': 'Streamlit dashboard framework',
    'google.generativeai': 'Gemini AI',
    'fastapi': 'API framework',
    'pandas': 'Data processing',
    'plotly': 'Visualizations',
    'joblib': 'Model loading',
    'dotenv': 'Environment variables'
}

missing_packages = []
for package, description in packages.items():
    try:
        if package == 'dotenv':
            __import__('dotenv')
        else:
            __import__(package)
        print(green(f"  ✓ {package}"))
    except ImportError:
        print(red(f"  ✗ {package} (missing - {description})"))
        missing_packages.append(package)

# Test 8: Try to load API key
print("\n[8/8] Testing API key loading...")
try:
    from dotenv import load_dotenv
    load_dotenv()
    
    api_key = os.getenv("GEMINI_API_KEY")
    if api_key and api_key != 'your_api_key_here':
        print(green(f"  ✓ API key loaded from .env"))
        print(green(f"  ✓ Key: {api_key[:20]}..."))
    else:
        print(yellow("  ⚠ API key not found or is placeholder"))
except Exception as e:
    print(red(f"  ✗ Error loading .env: {e}"))

# Summary
print("\n" + "="*70)
print(blue("DIAGNOSTIC SUMMARY"))
print("="*70)

issues = []

if missing_files:
    issues.append(f"Missing files: {', '.join(missing_files)}")

if missing_packages:
    issues.append(f"Missing packages: {', '.join(missing_packages)}")

if os.path.exists('main.py'):
    with open('main.py', 'r', encoding='utf-8') as f:
        if 'st.error("⚠️ GEMINI_API_KEY missing' in f.read():
            issues.append("main.py is OLD version (needs replacement)")

if not os.path.exists('.env') and not os.path.exists('.streamlit/secrets.toml'):
    issues.append("No API key configuration found")

if issues:
    print(red("\n⚠ ISSUES FOUND:"))
    for i, issue in enumerate(issues, 1):
        print(red(f"  {i}. {issue}"))
    
    print(yellow("\n📋 ACTION ITEMS:"))
    if 'main.py is OLD version' in str(issues):
        print(yellow("  1. Replace main.py with FIXED version"))
    if missing_packages:
        print(yellow(f"  2. Install packages: pip install {' '.join(missing_packages)}"))
    if 'No API key' in str(issues):
        print(yellow("  3. Add GEMINI_API_KEY to .env or .streamlit/secrets.toml"))
else:
    print(green("\n✅ ALL CHECKS PASSED!"))
    print(green("\nYour dashboard should work correctly."))
    print(blue("\nNext step: Run 'streamlit run main.py'"))

print("\n" + "="*70 + "\n")
