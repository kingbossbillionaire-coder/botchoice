# Alternative entry point for Streamlit Cloud deployment
# This file ensures compatibility with different deployment platforms

import sys
import os

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import and run main app
from app import main

if __name__ == "__main__":
    main()