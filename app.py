from flask import Flask, render_template, request, redirect, url_for, flash, session, jsonify, make_response
from werkzeug.security import generate_password_hash, check_password_hash
import datetime
import io
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.cluster import KMeans
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import os
import json
import plotly.graph_objects as go
import plotly.express as px
import seaborn as sns
import numpy as np

print("Starting app")

app = Flask(__name__)
app.secret_key = os.environ.get('SECRET_KEY', 'your_secret_key')  # Use environment variable in production

# Initialize data and model as None - will load on first request
data = None
model = None

def load_data_and_model():
    """Load CSV and train model - called on first request"""
    global data, model
    if data is not None and model is not None:
        return  # Already loaded
    
    try:
        print("Loading energy data...")
        # Try multiple paths
        csv_paths = ['energy_data.csv', './energy_data.csv', '/app/energy_data.csv']
        data_loaded = False
        
        for csv_path in csv_paths:
            if os.path.exists(csv_path):
                print(f"Found CSV at: {csv_path}")
                data = pd.read_csv(csv_path)
                data_loaded = True
                break
        
        if not data_loaded:
            print(f"WARNING: energy_data.csv not found in paths: {csv_paths}")
            print(f"Current directory: {os.getcwd()}")
            print(f"Files in current directory: {os.listdir('.')}")
            # Create dummy data if CSV not found
            data = pd.DataFrame({
                'Year': [2020, 2021, 2022],
                'Month': [1, 1, 1],
                'Population': [1000000, 1100000, 1200000],
                'Industrial_Growth': [0.05, 0.06, 0.07],
                'Energy_Consumption': [100, 110, 120]
            })
            print("Using dummy data")
        
        # Add seasonal features
        data['Month_sin'] = np.sin(2 * np.pi * data['Month'] / 12)
        data['Month_cos'] = np.cos(2 * np.pi * data['Month'] / 12)
        
        X = data[['Year','Month','Population','Industrial_Growth','Month_sin','Month_cos']]
        y = data['Energy_Consumption']
        
        print("Training model...")
        model = RandomForestRegressor(n_estimators=5, random_state=42)
        model.fit(X, y)
        print("Model trained successfully")
        
    except Exception as e:
        print(f"ERROR loading data/model: {e}")
        import traceback
        traceback.print_exc()
        raise

# revenue per kWh (configurable) in Indian Rupees
KWH_RATE = 10  # ₹10 per kWh

USERS_FILE = 'users.json'
PREDICTIONS_FILE = 'predictions.json'

def load_users():
    if os.path.exists(USERS_FILE):
        with open(USERS_FILE, 'r') as f:
            return json.load(f)
    return {}

def save_users(users):
    with open(USERS_FILE, 'w') as f:
        json.dump(users, f)

@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        name = request.form['name']
        email = request.form['email']
        username = request.form['username']
        password = request.form['password']
        users = load_users()
        if username in users:
            flash('Username already exists')
            return redirect(url_for('register'))
        users[username] = {'name': name, 'email': email, 'password': generate_password_hash(password)}
        save_users(users)
        flash('Registration successful, please login')
        return redirect(url_for('login'))
    return render_template('register.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        users = load_users()
        if username in users and check_password_hash(users[username]['password'], password):
            session['user'] = username
            return redirect(url_for('home'))
        else:
            flash('Invalid credentials')
    return render_template('login.html')

@app.route('/logout')
def logout():
    session.pop('user', None)
    return redirect(url_for('login'))

@app.route('/')
def home():
    load_data_and_model()  # Ensure data is loaded
    if 'user' not in session:
        return redirect(url_for('login'))
    # summary statistics
    avg_energy = None
    latest = None
    yoy = None
    try:
        avg_energy = round(data['Energy_Consumption'].mean(), 2)
        latest = data.iloc[-1]['Energy_Consumption']
        # compute year-over-year change comparing last 12 months average
        if len(data) >= 12:
            last12 = data.tail(12)['Energy_Consumption'].mean()
            prev12 = data.tail(24).head(12)['Energy_Consumption'].mean() if len(data) >= 24 else None
            if prev12 and prev12 != 0:
                yoy = round((last12 - prev12) / prev12 * 100, 2)
    except Exception:
        pass

    # load recent predictions
    recent = []
    try:
        if os.path.exists(PREDICTIONS_FILE):
            with open(PREDICTIONS_FILE, 'r') as f:
                recent = json.load(f)
    except Exception:
        recent = []

    # small analytics: monthly averages and last 12 months for sparkline
    monthly_avg = {}
    last12_labels = []
    last12_values = []
    try:
        monthly_avg = data.groupby('Month')['Energy_Consumption'].mean().round(2).to_dict()
        last12 = data.tail(12)
        last12_labels = [str(int(r['Year'])) + '-' + f"{int(r['Month']):02d}" for r in last12.to_dict(orient='records')]
        last12_values = list(last12['Energy_Consumption'].round(2).tolist())
    except Exception:
        monthly_avg = {}
        last12_labels = []
        last12_values = []

    # compute revenue metrics
    total_revenue = round(avg_energy * 365 * KWH_RATE, 2) if avg_energy else 0
    monthly_potential = round(avg_energy * 30 * KWH_RATE, 2) if avg_energy else 0
    latest_revenue = round(latest * 30 * KWH_RATE, 2) if latest else 0

    return render_template('index.html', avg_energy=avg_energy, latest_energy=latest, yoy=yoy, recent_predictions=recent, monthly_avg=monthly_avg, last12_labels=last12_labels, last12_values=last12_values, total_revenue=total_revenue, monthly_potential=monthly_potential, latest_revenue=latest_revenue, kwh_rate=KWH_RATE)

@app.route('/predict', methods=['POST'])
def predict():
    load_data_and_model()  # Ensure data is loaded
    year = int(request.form['year'])
    month = int(request.form['month'])
    population = float(request.form['population'])
    growth = float(request.form['growth'])
    month_sin = np.sin(2 * np.pi * month / 12)
    month_cos = np.cos(2 * np.pi * month / 12)
    prediction = model.predict([[year, month, population, growth, month_sin, month_cos]])
    pred_value = round(prediction[0], 2)

    # Generate plot
    plt.figure(figsize=(8, 5))
    plt.plot(data['Year'] + data['Month']/12, data['Energy_Consumption'], label='Historical Data', marker='o')
    plt.scatter(year + month/12, pred_value, color='red', label=f'Prediction: {pred_value}', s=100)
    plt.xlabel('Year')
    plt.ylabel('Energy Consumption (GWh)')
    plt.title('Energy Consumption Prediction')
    plt.legend()
    plt.grid(True)
    plot_path = os.path.join('static', 'prediction_plot.png')
    plt.savefig(plot_path)
    plt.close()

    # save prediction to file
    pred_record = {
        'user': session.get('user'),
        'year': year,
        'month': month,
        'population': population,
        'growth': growth,
        'prediction': pred_value,
        'timestamp': datetime.datetime.utcnow().isoformat() + 'Z'
    }
    try:
        if os.path.exists(PREDICTIONS_FILE):
            with open(PREDICTIONS_FILE, 'r') as f:
                preds = json.load(f)
        else:
            preds = []
        preds.insert(0, pred_record)
        # keep last 50 predictions
        preds = preds[:50]
        with open(PREDICTIONS_FILE, 'w') as f:
            json.dump(preds, f, indent=2)
    except Exception:
        pass

    # load recent predictions to display
    recent = []
    try:
        if os.path.exists(PREDICTIONS_FILE):
            with open(PREDICTIONS_FILE, 'r') as f:
                recent = json.load(f)
    except Exception:
        recent = []

    # recompute summary for display after prediction
    try:
        avg_energy = round(data['Energy_Consumption'].mean(), 2)
        latest = data.iloc[-1]['Energy_Consumption']
        if len(data) >= 12:
            last12 = data.tail(12)['Energy_Consumption'].mean()
            prev12 = data.tail(24).head(12)['Energy_Consumption'].mean() if len(data) >= 24 else None
            yoy = round((last12 - prev12) / prev12 * 100, 2) if prev12 and prev12 != 0 else None
        else:
            yoy = None
    except Exception:
        avg_energy = None
        latest = None
        yoy = None

    total_revenue = round(avg_energy * 365 * KWH_RATE, 2) if avg_energy else 0
    monthly_potential = round(avg_energy * 30 * KWH_RATE, 2) if avg_energy else 0
    latest_revenue = round(latest * 30 * KWH_RATE, 2) if latest else 0

    monthly_avg = {}
    last12_labels = []
    last12_values = []
    try:
        monthly_avg = data.groupby('Month')['Energy_Consumption'].mean().round(2).to_dict()
        last12 = data.tail(12)
        last12_labels = [str(int(r['Year'])) + '-' + f"{int(r['Month']):02d}" for r in last12.to_dict(orient='records')]
        last12_values = list(last12['Energy_Consumption'].round(2).tolist())
    except Exception:
        monthly_avg = {}
        last12_labels = []
        last12_values = []

    return render_template('index.html', prediction=pred_value, plot=True, recent_predictions=recent, avg_energy=avg_energy, latest_energy=latest, yoy=yoy, monthly_avg=monthly_avg, last12_labels=last12_labels, last12_values=last12_values, total_revenue=total_revenue, monthly_potential=monthly_potential, latest_revenue=latest_revenue, kwh_rate=KWH_RATE)

@app.route('/admin')
def admin():
    load_data_and_model()  # Ensure data is loaded
    if 'user' not in session or session['user'] != 'admin':
        return redirect(url_for('login'))
    users = load_users()
    total_users = len(users)
    total_predictions = len(data)
    return render_template('admin.html', total_users=total_users, total_predictions=total_predictions, users=users)


@app.route('/data')
def data_api():
    load_data_and_model()  # Ensure data is loaded
    # return historical CSV data for charts
    try:
        records = data.to_dict(orient='records')
        return jsonify(records)
    except Exception:
        return jsonify([])


@app.route('/export/data.csv')
def export_data_csv():
    load_data_and_model()  # Ensure data is loaded
    try:
        csv = data.to_csv(index=False)
        resp = make_response(csv)
        resp.headers['Content-Type'] = 'text/csv'
        resp.headers['Content-Disposition'] = 'attachment; filename=energy_data.csv'
        return resp
    except Exception:
        return redirect(url_for('home'))


@app.route('/export/data.json')
def export_data_json():
    load_data_and_model()  # Ensure data is loaded
    try:
        records = data.to_dict(orient='records')
        return jsonify(records)
    except Exception:
        return jsonify([])


@app.route('/export/predictions.csv')
def export_predictions_csv():
    try:
        if not os.path.exists(PREDICTIONS_FILE):
            return redirect(url_for('home'))
        with open(PREDICTIONS_FILE, 'r') as f:
            preds = json.load(f)
        df = pd.DataFrame(preds)
        csv = df.to_csv(index=False)
        resp = make_response(csv)
        resp.headers['Content-Type'] = 'text/csv'
        resp.headers['Content-Disposition'] = 'attachment; filename=predictions.csv'
        return resp
    except Exception:
        return redirect(url_for('home'))


@app.route('/export/predictions.json')
def export_predictions_json():
    try:
        if not os.path.exists(PREDICTIONS_FILE):
            return jsonify([])
        with open(PREDICTIONS_FILE, 'r') as f:
            preds = json.load(f)
        return jsonify(preds)
    except Exception:
        return jsonify([])


@app.context_processor
def inject_current_year():
    return {'current_year': datetime.datetime.now().year}

@app.route('/chat')
def chat():
    return render_template('chat.html')


@app.route('/insights')
def insights():
    load_data_and_model()  # Ensure data is loaded
    # graph algorithms: clustering and trend analysis
    try:
        # k-means clustering on energy consumption
        X_cluster = data[['Energy_Consumption']].values
        kmeans = KMeans(n_clusters=3, random_state=42)
        clusters = kmeans.fit_predict(X_cluster)
        data_copy = data.copy()
        data_copy['Cluster'] = clusters
        
        # cluster stats
        cluster_stats = {}
        for i in range(3):
            cluster_data = data_copy[data_copy['Cluster'] == i]['Energy_Consumption']
            cluster_stats[f'Cluster {i}'] = {
                'count': len(cluster_data),
                'avg': round(cluster_data.mean(), 2),
                'min': round(cluster_data.min(), 2),
                'max': round(cluster_data.max(), 2),
                'revenue': round(cluster_data.mean() * 30 * KWH_RATE, 2)
            }
        
        # trend: is consumption increasing?
        first_half_avg = data.head(len(data)//2)['Energy_Consumption'].mean()
        second_half_avg = data.tail(len(data)//2)['Energy_Consumption'].mean()
        trend = 'Increasing' if second_half_avg > first_half_avg else 'Decreasing'
        trend_pct = round((second_half_avg - first_half_avg) / first_half_avg * 100, 2) if first_half_avg else 0
        
        return jsonify({
            'clusters': cluster_stats,
            'trend': trend,
            'trend_pct': trend_pct,
            'recommendation': f'Energy {trend.lower()} by {abs(trend_pct)}%. Consider optimization strategies.' if trend_pct != 0 else 'Energy stable.'
        })
    except Exception as e:
        return jsonify({'error': str(e)})


@app.route('/autologin')
def autologin():
    # Temporary helper: set session as the primary user for quick access during development
    session['user'] = 'jawahar'
    return redirect(url_for('home'))


@app.route('/whoami')
def whoami():
    # debug endpoint: returns current session user
    return jsonify({'user': session.get('user')})


# ============= DATA VISUALIZATION FEATURES (Sem 5) =============

@app.route('/visualization')
def visualization():
    load_data_and_model()  # Ensure data is loaded
    if 'user' not in session:
        return redirect(url_for('login'))
    
    # Get data for charts
    monthly_data = data.groupby('Month')['Energy_Consumption'].mean().to_dict()
    monthly_labels = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
    monthly_values = [monthly_data.get(i, 0) for i in range(1, 13)]
    
    # Year data
    yearly_data = data.groupby('Year')['Energy_Consumption'].mean().to_dict()
    yearly_labels = sorted(yearly_data.keys())
    yearly_values = [yearly_data[y] for y in yearly_labels]
    
    return render_template('visualization.html', 
                         monthly_labels=monthly_labels,
                         monthly_values=monthly_values,
                         yearly_labels=yearly_labels,
                         yearly_values=yearly_values)


@app.route('/reports')
def reports():
    """Custom Report Builder"""
    if 'user' not in session:
        return redirect(url_for('login'))
    return render_template('reports.html')


@app.route('/api/generate-report', methods=['POST'])
def generate_report():
    """Generate custom report based on filters"""
    load_data_and_model()  # Ensure data is loaded
    try:
        filters = request.json
        filtered_data = data.copy()
        
        # Apply filters
        if filters.get('year_start'):
            filtered_data = filtered_data[filtered_data['Year'] >= int(filters['year_start'])]
        if filters.get('year_end'):
            filtered_data = filtered_data[filtered_data['Year'] <= int(filters['year_end'])]
        if filters.get('month'):
            filtered_data = filtered_data[filtered_data['Month'] == int(filters['month'])]
        
        if filtered_data.empty:
            return jsonify({'error': 'No data found for selected filters'})
        
        # Calculate metrics
        report = {
            'total_records': len(filtered_data),
            'avg_consumption': round(filtered_data['Energy_Consumption'].mean(), 2),
            'max_consumption': round(filtered_data['Energy_Consumption'].max(), 2),
            'min_consumption': round(filtered_data['Energy_Consumption'].min(), 2),
            'std_dev': round(filtered_data['Energy_Consumption'].std(), 2),
            'avg_population': round(filtered_data['Population'].mean(), 2),
            'avg_growth': round(filtered_data['Industrial_Growth'].mean(), 2),
            'total_revenue': round(filtered_data['Energy_Consumption'].mean() * len(filtered_data) * KWH_RATE, 2),
            'data_summary': filtered_data.describe().to_dict()
        }
        return jsonify(report)
    except Exception as e:
        return jsonify({'error': str(e)})


@app.route('/api/export-report', methods=['POST'])
def export_report():
    """Export custom report as CSV"""
    load_data_and_model()  # Ensure data is loaded
    try:
        filters = request.json
        filtered_data = data.copy()
        
        # Apply same filters
        if filters.get('year_start'):
            filtered_data = filtered_data[filtered_data['Year'] >= int(filters['year_start'])]
        if filters.get('year_end'):
            filtered_data = filtered_data[filtered_data['Year'] <= int(filters['year_end'])]
        if filters.get('month'):
            filtered_data = filtered_data[filtered_data['Month'] == int(filters['month'])]
        
        csv = filtered_data.to_csv(index=False)
        resp = make_response(csv)
        resp.headers['Content-Type'] = 'text/csv'
        resp.headers['Content-Disposition'] = f'attachment; filename=energy_report_{datetime.datetime.now().strftime("%Y%m%d_%H%M%S")}.csv'
        return resp
    except Exception as e:
        return redirect(url_for('reports'))

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    print(f"Starting Flask app on http://localhost:{port}/")
    print("Press Ctrl+C to stop the server")
    app.run(host='0.0.0.0', port=port, debug=False)