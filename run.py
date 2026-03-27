"""Entry point for RetailMind Product Intelligence Agent.
Run this file to launch the Streamlit application:
    python run.py
"""
import subprocess
import sys
import os

if __name__ == "__main__":
    # Launch Streamlit app from the project root directory
    project_dir = os.path.dirname(os.path.abspath(__file__))
    subprocess.run(
        [sys.executable, "-m", "streamlit", "run", "app.py",
         "--server.headless", "true"],
        cwd=project_dir
    )
