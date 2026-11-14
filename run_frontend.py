"""
run_frontend.py
===============
Checks if Backend is active, then launches the Frontend Dashboard in your browser.
"""

import sys
import time
import urllib.request
import webbrowser

BACKEND_URL = "http://127.0.0.1:5000"
HEALTH_URL = f"{BACKEND_URL}/api/health"

print("=" * 65)
print("       [FRONTEND] PHISHING SHIELD AI - CYBER DASHBOARD")
print("=" * 65)
print(f"[*] Checking connection to Backend at {BACKEND_URL}...")

# Check if backend is alive
backend_ready = False
try:
    with urllib.request.urlopen(HEALTH_URL, timeout=2) as response:
        if response.status == 200:
            backend_ready = True
except Exception:
    backend_ready = False

if backend_ready:
    print("[✓] Backend is ONLINE and healthy!")
else:
    print("[!] Warning: Backend is not running yet.")
    print("    Make sure to run 'python run_backend.py' in a separate terminal.")

print(f"\n[🚀] Opening Frontend Cyber Dashboard in browser: {BACKEND_URL}")
print("=" * 65)

time.sleep(1)
webbrowser.open(BACKEND_URL)

print("[✓] Dashboard opened in your default web browser.")

