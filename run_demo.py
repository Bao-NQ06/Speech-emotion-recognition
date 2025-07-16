#!/usr/bin/env python3
"""
Demo Launcher Script
Simple script to launch the Streamlit demo interface
"""

import subprocess
import sys
import os

def check_requirements():
    """Check if required packages are installed"""
    try:
        import streamlit
        import plotly
        print("✅ All required packages are installed")
        return True
    except ImportError as e:
        print(f"❌ Missing required package: {e}")
        print("Please install requirements: pip install -r requirements.txt")
        return False

def check_model():
    """Check if the trained model exists"""
    if not os.path.exists("enhanced_best_model.pth"):
        print("❌ Model file 'enhanced_best_model.pth' not found!")
        print("Please ensure the trained model is in the current directory.")
        return False
    print("✅ Model file found")
    return True

def main():
    """Launch the demo"""
    print("🚀 Audio Emotion Recognition Demo Launcher")
    print("="*50)
    
    # Check requirements
    if not check_requirements():
        return
    
    # Check model
    if not check_model():
        return
    
    print("🎵 Starting Streamlit demo...")
    print("The demo will open in your default web browser")
    print("Press Ctrl+C to stop the demo")
    print("="*50)
    
    # Launch Streamlit
    try:
        subprocess.run([sys.executable, "-m", "streamlit", "run", "demo.py"])
    except KeyboardInterrupt:
        print("\n👋 Demo stopped by user")
    except Exception as e:
        print(f"❌ Error launching demo: {e}")

if __name__ == "__main__":
    main() 