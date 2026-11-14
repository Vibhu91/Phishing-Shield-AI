"""
run_backend.py
==============
Starts only the Flask REST API & ML Backend (DistilBERT + URL DNN + Fusion Engine).
Serves API endpoints on http://127.0.0.1:5000/api/...
"""

import os
import sys

# Suppress harmless Matplotlib warning on Windows
os.environ["MPLCONFIGDIR"] = os.path.join(os.environ.get("TEMP", "."), "mpl_cache")

print("=" * 65)
print("       [BACKEND] PHISHING SHIELD AI - INFERENCE ENGINE")
print("=" * 65)
print("[*] Loading AI Models into GPU memory...")
print("    • Email Model: DistilBERT Transformer")
print("    • URL Model  : Deep Neural Network (49 features)")
print("    • Fusion     : Bayesian Noisy-OR Model")

from app import app, get_detector

# Preload detector
get_detector()

PORT = int(os.environ.get("PORT", 5000))

print("\n[✓] Backend Models Loaded Successfully!")
print(f"[🚀] Backend REST API running at: http://127.0.0.1:{PORT}")
print("     Endpoints ready:")
print("     - POST http://127.0.0.1:5000/api/predict/url")
print("     - POST http://127.0.0.1:5000/api/predict/email")
print("     - GET  http://127.0.0.1:5000/api/health")
print("=" * 65)
print("Press CTRL+C to stop the backend.")
print("=" * 65 + "\n")

if __name__ == "__main__":
    app.run(host="127.0.0.1", port=PORT, debug=False)

