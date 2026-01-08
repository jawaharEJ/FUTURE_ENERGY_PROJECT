from flask import Flask, render_template, request, redirect, url_for, flash, session
import pandas as pd
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import os
import json

print("Starting app")

app = Flask(__name__)
app.secret_key = 'your_secret_key'  # Change this to a random secret key

data = pd.read_csv('energy_data.csv')

X = data[['Year','Month','Population','Industrial_Growth']]
y = data['Energy_Consumption']

model = LinearRegression()
model.fit(X, y)

USERS_FILE = 'users.json'

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
        users[username] = {'name': name, 'email': email, 'password': password}
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
        if username in users and users[username]['password'] == password:
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
    if 'user' not in session:
        return redirect(url_for('login'))
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    year = int(request.form['year'])
    month = int(request.form['month'])
    population = float(request.form['population'])
    growth = float(request.form['growth'])
    prediction = model.predict([[year, month, population, growth]])
    pred_value = round(prediction[0], 2)

    # Generate plot
    plt.figure(figsize=(8, 5))
    plt.plot(data['Year'] + data['Month']/12, data['Energy_Consumption'], label='Historical Data', marker='o')
    plt.scatter(year + month/12, pred_value, color='red', label=f'Prediction: {pred_value}', s=100)
    plt.xlabel('Year')
    plt.ylabel('Energy Consumption (kWh)')
    plt.title('Energy Consumption Prediction')
    plt.legend()
    plt.grid(True)
    plot_path = os.path.join('static', 'prediction_plot.png')
    plt.savefig(plot_path)
    plt.close()

    return render_template('index.html', prediction=pred_value, plot=True)

@app.route('/admin')
def admin():
    if 'user' not in session or session['user'] != 'admin':
        return redirect(url_for('login'))
    users = load_users()
    total_users = len(users)
    total_predictions = len(data)
    return render_template('admin.html', total_users=total_users, total_predictions=total_predictions, users=users)

@app.route('/chat')
def chat():
    return render_template('chat.html')

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)