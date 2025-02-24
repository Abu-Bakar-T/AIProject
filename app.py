from flask import Flask, render_template, request, jsonify
import pickle
import numpy as np
from waitress import serve  # Import waitress

app = Flask(__name__)

# Load trained model and scaler
with open('house_price_model.pkl', 'rb') as model_file:
    model = pickle.load(model_file)
with open('scaler.pkl', 'rb') as scaler_file:
    scaler = pickle.load(scaler_file)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get form data
        form_data = [
            int(request.form['bedrooms']),
            int(request.form['bathrooms']),
            float(request.form['living_in_m2']),
            int(request.form['nice_view']),
            int(request.form['perfect_condition']),
            int(request.form['grade']),
            int(request.form['has_basement']),
            int(request.form['renovated']),
            int(request.form['has_lavatory']),
            int(request.form['single_floor']),
            int(request.form['month']),
            int(request.form['quartile_zone'])
        ]
        
        # Convert to NumPy array
        features = np.array(form_data).reshape(1, -1)
        
        # Scale the 'living_in_m2' column
        features[:, 2] = scaler.transform(features[:, 2].reshape(-1, 1))
        
        # Predict
        prediction = model.predict(features)
        predicted_price = round(prediction[0], 2)
        
        return render_template('index.html', prediction_text=f'Estimated House Price: ${predicted_price}')
    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8000)