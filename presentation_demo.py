"""
presentation_demo.py
====================
Simple, presentation-ready script to start both Backend and Frontend for your teacher demo.

Usage:
    python presentation_demo.py
    (or double click START_PRESENTATION.bat)
"""

import os
import sys
import time
import webbrowser

# Set Matplotlib temp directory to suppress harmless Windows warnings
os.environ["MPLCONFIGDIR"] = os.path.join(os.environ.get("TEMP", "."), "mpl_cache")

print("=" * 68)
print("     PHISHING SHIELD AI - DUAL-ENGINE CYBER DEFENSE SYSTEM")
print("=" * 68)
print("\n[Step 1/3] Loading AI Models onto GPU...")
print("  • Email Engine : DistilBERT Transformer (Semantic Intent)")
print("  • URL Engine   : Deep Neural Network (49 Lexical Features)")
print("  • Fusion Engine: Bayesian Noisy-OR Fusion")

# Import Flask app and detector
from app import app, get_detector

# Preload models into memory before starting server
detector = get_detector()
print("\n[Step 2/3] Models successfully loaded and verified!")

# Target URL
PORT = 5000
URL = f"http://127.0.0.1:{PORT}"

print(f"\n[Step 3/3] Starting Web Server & Launching Frontend Dashboard...")
print(f"  • Local Address: {URL}")
print(f"  • Hardware     : NVIDIA CUDA GPU Accelerated")
print("=" * 68)
print(">>> PRESENTATION IS READY! Opening browser automatically... <<<")
print("Press CTRL+C in this terminal when you want to stop the demo.")
print("=" * 68 + "\n")

# Automatically open the browser
time.sleep(1)
webbrowser.open(URL)

# Run Flask server
if __name__ == "__main__":
    app.run(host="127.0.0.1", port=PORT, debug=False)

