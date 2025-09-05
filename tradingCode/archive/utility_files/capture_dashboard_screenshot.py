#!/usr/bin/env python3
"""Simple script to run main.py and capture dashboard screenshot"""

import subprocess
import time
import os
from datetime import datetime

print("=" * 70)
print("RUNNING MAIN.PY WITH DASHBOARD")
print("=" * 70)
print()

# Run main.py as a subprocess
cmd = ["python", "main.py", "ES", "simpleSMA"]
print(f"Command: {' '.join(cmd)}")
print("Starting dashboard (this takes about 30 seconds)...")
print()

# Start the process
process = subprocess.Popen(
    cmd,
    stdout=subprocess.PIPE,
    stderr=subprocess.STDOUT,
    text=True,
    cwd="C:/Users/skyeAM/SkyeAM Dropbox/SAMresearch/ABtoPython/tradingCode"
)

# Monitor output
print("Console output:")
print("-" * 40)

important_lines = []
start_time = time.time()
dashboard_launched = False

while True:
    line = process.stdout.readline()
    if not line:
        if process.poll() is not None:
            break
        continue
    
    line = line.strip()
    
    # Look for important messages
    if any(keyword in line for keyword in ["Trade indices", "Safe viewport", "Trade arrows", "Data length", "Dashboard", "SUCCESS", "ERROR", "WARNING"]):
        print(f"[{time.time()-start_time:.1f}s] {line}")
        important_lines.append(line)
    
    if "Ultimate Dashboard launched" in line or "dashboard" in line.lower():
        dashboard_launched = True
    
    # Give dashboard time to fully render
    if dashboard_launched and time.time() - start_time > 20:
        print("\n[INFO] Dashboard should be visible now")
        print("[INFO] Please take a screenshot manually")
        print("[INFO] Save it as: 2025-08-08_2/main_py_manual_verification.png")
        break
    
    # Timeout after 60 seconds
    if time.time() - start_time > 60:
        print("\n[TIMEOUT] Process took too long")
        break

print("\n" + "=" * 70)
print("IMPORTANT MESSAGES FOUND:")
print("-" * 40)
for line in important_lines:
    print(line)

print("\n" + "=" * 70)
print("VERIFICATION CHECKLIST:")
print("-" * 40)
print("1. Did the dashboard appear? (Y/N)")
print("2. Are candlesticks visible? (Y/N)")
print("3. Are trade arrows visible on the main chart? (Y/N)")
print("4. Does the trade list show correct dates? (Y/N)")
print("5. Is there an 'Invalid indices' error? (Y/N)")
print()
print("Take a screenshot and save as:")
print("C:\\Users\\skyeAM\\SkyeAM Dropbox\\SAMresearch\\ABtoPython\\2025-08-08_2\\main_py_manual_verification.png")