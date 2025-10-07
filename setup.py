#!/usr/bin/env python3
"""
Setup script for Virtual Patient Monitor
Installs required dependencies and verifies the environment
"""

import subprocess
import sys
import os

def install_requirements():
    """Install all requirements from requirements.txt"""
    print("Installing requirements...")
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
        print("✓ Requirements installed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"✗ Failed to install requirements: {e}")
        return False

def verify_installation():
    """Verify that key packages are installed"""
    print("\nVerifying installation...")
    
    required_packages = [
        'streamlit',
        'pandas', 
        'numpy',
        'plotly',
        'scikit-learn',
        'tensorflow',
        'paho-mqtt',
        'websockets',
        'schedule'
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package.replace('-', '_'))
            print(f"✓ {package}")
        except ImportError:
            print(f"✗ {package} - Missing")
            missing_packages.append(package)
    
    if missing_packages:
        print(f"\nMissing packages: {', '.join(missing_packages)}")
        return False
    else:
        print("\n✓ All packages installed successfully!")
        return True

def main():
    """Main setup function"""
    print("Virtual Patient Monitor - Setup")
    print("=" * 50)
    
    if not os.path.exists("requirements.txt"):
        print("✗ requirements.txt not found!")
        return False
    
    # Install requirements
    if not install_requirements():
        return False
    
    # Verify installation
    if not verify_installation():
        print("\nSome packages failed to install. You may need to install them manually.")
        return False
    
    print("\n" + "=" * 50)
    print("✓ Setup complete! You can now run the Virtual Patient Monitor.")
    print("\nTo start the demo:")
    print("  python3 demo.py")
    print("\nTo start the Streamlit dashboard:")
    print("  streamlit run src/visualization/streamlit_dashboard.py")
    
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
