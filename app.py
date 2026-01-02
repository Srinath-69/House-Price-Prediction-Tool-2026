from flask import Flask, render_template, request, jsonify, session, redirect, url_for
from flask_sqlalchemy import SQLAlchemy
from werkzeug.security import generate_password_hash, check_password_hash
import pandas as pd
import numpy as np
import pickle
import os
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
from datetime import datetime
import pytz

app = Flask(__name__)
app.secret_key = 'your_secret_key'

# Database config
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///users.db'
db = SQLAlchemy(app)

# User model
class User(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(150), unique=True, nullable=False)
    password = db.Column(db.String(200), nullable=False)

# Prediction History model
class PredictionHistory(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    predicted_price = db.Column(db.Float, nullable=False)
    timestamp = db.Column(db.DateTime, default=datetime.utcnow)

    user = db.relationship('User', backref=db.backref('history', lazy=True))

# Create the database tables
with app.app_context():
    db.create_all()

# Function to train the model
def train_model():
    # Check if data directory exists, if not create it
    if not os.path.exists('data'):
        os.makedirs('data')
    
    # Check if model directory exists, if not create it
    if not os.path.exists('model'):
        os.makedirs('model')
    
    # Check if dataset exists, if not generate it
    if not os.path.exists('data/house_prices.csv'):
        exec(open('dataset_generator.py').read())
    
    # Load the dataset
    data = pd.read_csv('data/house_prices.csv')
    
    # Split features and target
    X = data.drop('price', axis=1)
    y = data['price']
    
    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Standardize the features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Create and train the model
    model = LinearRegression()
    model.fit(X_train_scaled, y_train)
    
    # Evaluate the model
    y_pred = model.predict(X_test_scaled)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    print(f"Model trained with MSE: {mse:.2f} and R² score: {r2:.2f}")
    
    # Save the model and scaler
    with open('model/model.pkl', 'wb') as f:
        pickle.dump(model, f)
    
    with open('model/scaler.pkl', 'wb') as f:
        pickle.dump(scaler, f)
    
    return model, scaler, mse, r2

# Load or train the model
if not (os.path.exists('model/model.pkl') and os.path.exists('model/scaler.pkl')):
    model, scaler, mse, r2 = train_model()
else:
    # Load the model and scaler
    with open('model/model.pkl', 'rb') as f:
        model = pickle.load(f)
    
    with open('model/scaler.pkl', 'rb') as f:
        scaler = pickle.load(f)

@app.route('/')
def homepage():
    return render_template('home.html')

@app.route('/')
def home():
    if 'username' not in session:
        return redirect(url_for('register'))  # Redirect to register page first
    # Load dataset statistics for display
    USD_TO_INR = 86  # Update this rate as needed
    if os.path.exists('data/house_prices.csv'):
        data = pd.read_csv('data/house_prices.csv')
        stats = {
            'count': len(data),
            'avg_price': f"₹{data['price'].mean() * USD_TO_INR:,.2f}",
            'min_price': f"₹{data['price'].min() * USD_TO_INR:,.2f}",
            'max_price': f"₹{data['price'].max() * USD_TO_INR:,.2f}",
            'avg_area': f"{data['area'].mean():.0f} sq ft",
            'avg_bedrooms': f"{data['bedrooms'].mean():.1f}",
            'avg_bathrooms': f"{data['bathrooms'].mean():.1f}"
        }
    else:
        stats = {}
    
    return render_template('index.html', stats=stats)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get input data from the form
        area = float(request.form['area'])
        bedrooms = int(request.form['bedrooms'])
        bathrooms = int(request.form['bathrooms'])
        age = int(request.form['age'])
        garage = int(request.form['garage'])
        yard_size = float(request.form['yard_size'])
        distance_to_city = float(request.form['distance_to_city'])
        
        # Create a dataframe with the input
        input_data = pd.DataFrame({
            'area': [area],
            'bedrooms': [bedrooms],
            'bathrooms': [bathrooms],
            'age': [age],
            'garage': [garage],
            'yard_size': [yard_size],
            'distance_to_city': [distance_to_city]
        })
        
        # Scale the input data
        input_scaled = scaler.transform(input_data)
        
        # Make prediction
        prediction = model.predict(input_scaled)[0]
        USD_TO_INR = 86
        prediction_inr = prediction * USD_TO_INR
        formatted_prediction = f"₹{prediction_inr:,.2f}"

        # Save to history if user is logged in
        if 'username' in session:
            user = User.query.filter_by(username=session['username']).first()
            if user:
                history = PredictionHistory(
                    user_id=user.id,
                    predicted_price=prediction_inr
                )
                db.session.add(history)
                db.session.commit()

        session['last_inputs'] = {
            'area': area,
            'bedrooms': bedrooms,
            'bathrooms': bathrooms,
            'age': age,
            'garage': garage,
            'yard_size': yard_size,
            'distance_to_city': distance_to_city,
            'predicted_price': prediction_inr,
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }

        data = pd.read_csv('data/house_prices.csv')
        stats = {
            'count': len(data),
            'avg_price': f"₹{data['price'].mean() * USD_TO_INR:,.2f}",
            'min_price': f"₹{data['price'].min() * USD_TO_INR:,.2f}",
            'max_price': f"₹{data['price'].max() * USD_TO_INR:,.2f}",
            'avg_area': f"{data['area'].mean():.0f} sq ft",
            'avg_bedrooms': f"{data['bedrooms'].mean():.1f}",
            'avg_bathrooms': f"{data['bathrooms'].mean():.1f}"
        }

        return render_template('index.html', prediction=formatted_prediction, stats=stats)
    except Exception as e:
        return render_template('index.html', error=str(e))

@app.route('/retrain', methods=['GET'])
def retrain():
    try:
        model, scaler, mse, r2 = train_model()
        return jsonify({'success': True, 'mse': mse, 'r2': r2})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@app.route('/login', methods=['GET', 'POST'])
def login():
    error = None
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        user = User.query.filter_by(username=username).first()
        if user and check_password_hash(user.password, password):
            session['username'] = username
            return redirect(url_for('index'))
        else:
            error = 'Invalid username or password'
    return render_template('login.html', error=error)

@app.route('/index')
def index():
    USD_TO_INR = 86  # Update this rate as needed
    if os.path.exists('data/house_prices.csv'):
        data = pd.read_csv('data/house_prices.csv')
        stats = {
            'count': len(data),
            'avg_price': f"₹{data['price'].mean() * USD_TO_INR:,.2f}",
            'min_price': f"₹{data['price'].min() * USD_TO_INR:,.2f}",
            'max_price': f"₹{data['price'].max() * USD_TO_INR:,.2f}",
            'avg_area': f"{data['area'].mean():.0f} sq ft",
            'avg_bedrooms': f"{data['bedrooms'].mean():.1f}",
            'avg_bathrooms': f"{data['bathrooms'].mean():.1f}"
        }
    else:
        stats = {}

    return render_template('index.html', stats=stats)

@app.route('/register', methods=['GET', 'POST'])
def register():
    error = None
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        confirm_password = request.form['confirm_password']
        if User.query.filter_by(username=username).first():
            error = 'Username already exists'
        elif password != confirm_password:
            error = 'Passwords do not match'
        else:
            hashed_password = generate_password_hash(password)
            new_user = User(username=username, password=hashed_password)
            db.session.add(new_user)
            db.session.commit()
            return redirect(url_for('login'))
    return render_template('register.html', error=error)

@app.route('/logout')
def logout():
    session.pop('username', None)
    return redirect(url_for('homepage'))

@app.route('/users')
def users():
    all_users = User.query.all()
    return render_template('users.html', users=all_users)

@app.route('/history')
def history():
    if 'username' not in session:
        return redirect(url_for('login'))
    user = User.query.filter_by(username=session['username']).first()
    if user:
        history = PredictionHistory.query.filter_by(user_id=user.id).order_by(PredictionHistory.timestamp.desc()).all()
        # Convert UTC to IST
        ist = pytz.timezone('Asia/Kolkata')
        for h in history:
            h.timestamp = h.timestamp.replace(tzinfo=pytz.utc).astimezone(ist)
    else:
        history = []
    return render_template('history.html', history=history)

if __name__ == '__main__':
    app.run(debug=True)
