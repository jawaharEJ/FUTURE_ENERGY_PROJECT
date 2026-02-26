import sys
import os

# Redirect output to file
log_file = open('app_startup.log', 'w', buffering=1)
sys.stdout = log_file
sys.stderr = log_file

print("=" * 50, flush=True)
print("App startup log", flush=True)
print("=" * 50, flush=True)

try:
    print("Step 1: Starting imports...", flush=True)
    from flask import Flask, render_template, request, redirect, url_for, flash, session, jsonify, make_response
    from werkzeug.security import generate_password_hash, check_password_hash
    import datetime
    import io
    import pandas as pd
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.cluster import KMeans
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import os
    import json
    import plotly.graph_objects as go
    import plotly.express as px
    import seaborn as sns
    import numpy as np
    print("✓ All imports successful", flush=True)
    
    print("\nStep 2: Creating Flask app...", flush=True)
    app = Flask(__name__)
    app.secret_key = 'your_secret_key'
    print("✓ Flask app created", flush=True)
    
    print("\nStep 3: Loading data...", flush=True)
    data = pd.read_csv('energy_data.csv')
    print(f"✓ Data loaded: {len(data)} rows", flush=True)
    
    print("\nStep 4: Preparing training data...", flush=True)
    data['Month_sin'] = np.sin(2 * np.pi * data['Month'] / 12)
    data['Month_cos'] = np.cos(2 * np.pi * data['Month'] / 12)
    X = data[['Year','Month','Population','Industrial_Growth','Month_sin','Month_cos']]
    y = data['Energy_Consumption']
    print("✓ Training data prepared", flush=True)
    
    print("\nStep 5: Training model (this may take a minute)...", flush=True)
    model = RandomForestRegressor(n_estimators=10, random_state=42)
    model.fit(X, y)
    print("✓ Model trained", flush=True)
    
    print("\nStep 6: Starting Flask server...", flush=True)
    port = int(os.environ.get('PORT', 5000))
    print(f"Server starting on port {port}", flush=True)
    app.run(host='0.0.0.0', port=port, debug=False)
    
except Exception as e:
    import traceback
    print(f"\n✗ ERROR: {e}", flush=True)
    print(traceback.format_exc(), flush=True)
    sys.exit(1)
