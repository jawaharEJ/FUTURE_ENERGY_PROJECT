import sys
import time

steps = [
    ("Flask", lambda: __import__('flask')),
    ("Pandas", lambda: __import__('pandas')),
    ("CSV load", lambda: __import__('pandas').read_csv('energy_data.csv')),
    ("Numpy", lambda: __import__('numpy')),
    ("Sklearn imports", lambda: __import__('sklearn.ensemble', fromlist=['RandomForestRegressor'])),
]

for name, step in steps:
    try:
        print(f"[{time.time()}] Testing {name}...", flush=True)
        start = time.time()
        result = step()
        elapsed = time.time() - start
        print(f"[{time.time()}] ✓ {name} OK ({elapsed:.2f}s)", flush=True)
    except Exception as e:
        print(f"[{time.time()}] ✗ {name} FAILED: {e}", flush=True)
        sys.exit(1)

# Now test model training
print(f"[{time.time()}] Testing ML pipeline...", flush=True)
try:
    import pandas as pd
    import numpy as np
    from sklearn.ensemble import RandomForestRegressor
    
    print(f"[{time.time()}] Loading CSV...", flush=True)
    data = pd.read_csv('energy_data.csv')
    
    print(f"[{time.time()}] Preparing features...", flush=True)
    data['Month_sin'] = np.sin(2 * np.pi * data['Month'] / 12)
    data['Month_cos'] = np.cos(2 * np.pi * data['Month'] / 12)
    
    X = data[['Year','Month','Population','Industrial_Growth','Month_sin','Month_cos']]
    y = data['Energy_Consumption']
    
    print(f"[{time.time()}] Training model (this may take a minute)...", flush=True)
    start = time.time()
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X, y)
    elapsed = time.time() - start
    print(f"[{time.time()}] ✓ Model trained ({elapsed:.2f}s)", flush=True)
except Exception as e:
    print(f"[{time.time()}] ✗ ML pipeline FAILED: {e}", flush=True)
    sys.exit(1)

print(f"[{time.time()}] All tests passed!", flush=True)
