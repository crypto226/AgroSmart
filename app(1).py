from flask import Flask, request, jsonify, render_template, redirect, url_for, session
import pandas as pd

app = Flask(__name__, static_folder='static', template_folder='templates')
app.secret_key = 'your_secret_key_here'  

df = pd.read_csv("Fertilizer_Prediction_gpt(1).csv")

@app.route('/')
def login():
    if session.get('logged_in'):
        return redirect(url_for('home'))
    return render_template('login.html')

@app.route('/login', methods=['POST'])
def do_login():
    data = request.form
    username = data.get('username')
    password = data.get('password')
    if username == "farmer" and password == "12345":
        session['logged_in'] = True
        return redirect(url_for('home'))
    else:
        error = "Invalid username or password"
        return render_template('login.html', error=error)

@app.route('/home')
def home():
    if not session.get('logged_in'):
        return redirect(url_for('login'))
    return render_template('home.html')

@app.route('/recommend')
def recommend():
    if not session.get('logged_in'):
        return redirect(url_for('login'))
    return render_template('recommendation.html')

@app.route('/carbon-footprint')
def carbon_footprint():
    if not session.get('logged_in'):
        return redirect(url_for('login'))
    return render_template('login.html')

@app.route('/logout')
def logout():
    session.clear()
    return redirect(url_for('login'))

@app.route('/predict', methods=['POST'])
def predict():
    if not session.get('logged_in'):
        return jsonify({'error': 'Unauthorized'}), 401
    data = request.get_json()
    try:
        input_row = {
            'Temperature': float(data['Temperature']),
            'Humidity': float(data['Humidity']),
            'Moisture': float(data['Moisture']),
            'Nitrogen': float(data['Nitrogen']),
            'Phosphorus': float(data['Phosphorus']),
            'Potassium': float(data['Potassium']),
            'Soil Type': data['Soil Type'],
            'Crop Type': data['Crop Type']
        }
        matched = df[
            (df['Temperature'] == input_row['Temperature']) &
            (df['Humidity'] == input_row['Humidity']) &
            (df['Moisture'] == input_row['Moisture']) &
            (df['Nitrogen'] == input_row['Nitrogen']) &
            (df['Phosphorus'] == input_row['Phosphorus']) &
            (df['Potassium'] == input_row['Potassium']) &
            (df['Soil Type'] == input_row['Soil Type']) &
            (df['Crop Type'] == input_row['Crop Type'])
        ]
        if not matched.empty:
            fertilizer_name = matched.iloc[0]['Fertilizer Name']
            return jsonify({'recommendation': fertilizer_name, 'confidence': 100.0})
        else:
            return jsonify({'error': 'No exact match found in dataset.'}), 404
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)




