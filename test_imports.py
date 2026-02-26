#!/usr/bin/env python
import sys
print("Test 1: Basic imports", flush=True)

try:
    from flask import Flask
    print("✓ Flask imported", flush=True)
except Exception as e:
    print(f"✗ Flask import failed: {e}", flush=True)
    sys.exit(1)

print("Test 2: Data imports", flush=True)
try:
    import pandas as pd
    print("✓ Pandas imported", flush=True)
except Exception as e:
    print(f"✗ Pandas import failed: {e}", flush=True)
    sys.exit(1)

print("Test 3: Loading CSV", flush=True)
try:
    data = pd.read_csv('energy_data.csv')
    print(f"✓ CSV loaded: {len(data)} rows", flush=True)
except Exception as e:
    print(f"✗ CSV load failed: {e}", flush=True)
    sys.exit(1)

print("Test 4: ML imports", flush=True)
try:
    from sklearn.ensemble import RandomForestRegressor
    import numpy as np
    print("✓ ML libraries imported", flush=True)
except Exception as e:
    print(f"✗ ML import failed: {e}", flush=True)
    sys.exit(1)

print("Test 5: Model training", flush=True)
try:
    data['Month_sin'] = np.sin(2 * np.pi * data['Month'] / 12)
    data['Month_cos'] = np.cos(2 * np.pi * data['Month'] / 12)
    X = data[['Year','Month','Population','Industrial_Growth','Month_sin','Month_cos']]
    y = data['Energy_Consumption']
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X, y)
    print("✓ Model trained", flush=True)
except Exception as e:
    print(f"✗ Model training failed: {e}", flush=True)
    sys.exit(1)

print("All tests passed!", flush=True)
