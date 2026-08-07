# app.py
from flask import Flask, request, jsonify, render_template_string
import pandas as pd
import numpy as np
import joblib
import os

app = Flask(__name__)

# Load model once (cached across serverless invocations)
MODEL_PATH = os.path.join(os.path.dirname(__file__), 'models', 'rf_model.pkl')
model = None
if os.path.exists(MODEL_PATH):
    model = joblib.load(MODEL_PATH)

# Feature engineering (same as before)
def prepare_features(row):
    prob_1 = 1 / row['odd_1']
    prob_X = 1 / row['odd_X']
    prob_2 = 1 / row['odd_2']
    margin = prob_1 + prob_X + prob_2
    return {
        'prob_1': prob_1,
        'prob_X': prob_X,
        'prob_2': prob_2,
        'margin': margin,
        'norm_prob_1': prob_1 / margin,
        'norm_prob_X': prob_X / margin,
        'norm_prob_2': prob_2 / margin,
    }

# Simple HTML template
HTML_TEMPLATE = '''
<!DOCTYPE html>
<html>
<head>
    <title>Uganda Betting Predictions</title>
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <style>
        body { font-family: sans-serif; max-width: 600px; margin: 2rem auto; padding: 0 1rem; }
        h1 { color: #2e7d32; }
        label { display: block; margin-top: 1rem; }
        input { width: 100%; padding: 0.5rem; margin: 0.3rem 0; }
        button { background: #2e7d32; color: white; padding: 0.7rem 1.5rem; border: none; border-radius: 4px; margin-top: 1rem; cursor: pointer; }
        .result { margin-top: 2rem; padding: 1rem; background: #e8f5e9; border-radius: 4px; }
        .error { color: #c62828; }
    </style>
</head>
<body>
    <h1>⚽ Uganda Betting Prediction</h1>
    <p>Enter 1X2 odds to get a model prediction.</p>
    <form method="POST" action="/predict">
        <label>Home Team</label>
        <input type="text" name="home_team" placeholder="e.g. KCCA" required>
        <label>Away Team</label>
        <input type="text" name="away_team" placeholder="e.g. Vipers" required>
        <label>Odd for Home Win (1)</label>
        <input type="number" step="0.01" name="odd_1" required>
        <label>Odd for Draw (X)</label>
        <input type="number" step="0.01" name="odd_X" required>
        <label>Odd for Away Win (2)</label>
        <input type="number" step="0.01" name="odd_2" required>
        <button type="submit">Predict</button>
    </form>
    {% if prediction %}
    <div class="result">
        <h3>{{ home_team }} vs {{ away_team }}</h3>
        <p><strong>Prediction:</strong> {{ prediction }} ({{ confidence }}% confidence)</p>
    </div>
    {% endif %}
    {% if error %}
    <p class="error">{{ error }}</p>
    {% endif %}
</body>
</html>
'''

@app.route('/')
def home():
    return render_template_string(HTML_TEMPLATE)

@app.route('/predict', methods=['POST'])
def predict():
    if model is None:
        return render_template_string(HTML_TEMPLATE, error="Model not found. Please train and save it first.")
    try:
        home = request.form.get('home_team', '')
        away = request.form.get('away_team', '')
        odds = {
            'odd_1': float(request.form['odd_1']),
            'odd_X': float(request.form['odd_X']),
            'odd_2': float(request.form['odd_2']),
        }
    except (ValueError, KeyError):
        return render_template_string(HTML_TEMPLATE, error="Invalid odds values.")

    row = pd.Series(odds)
    features = prepare_features(row)
    X = pd.DataFrame([features])
    expected = ['prob_1', 'prob_X', 'prob_2', 'margin', 'norm_prob_1', 'norm_prob_X', 'norm_prob_2']
    X = X[expected]

    pred = model.predict(X)[0]
    proba = model.predict_proba(X)[0]
    conf = max(proba) * 100
    outcome_map = {1: "Home Win", 0: "Draw", 2: "Away Win"}
    prediction_text = outcome_map.get(pred, "Unknown")

    return render_template_string(
        HTML_TEMPLATE,
        home_team=home,
        away_team=away,
        prediction=prediction_text,
        confidence=round(conf, 1)
    )

# Vercel serverless requires a callable `app`
if __name__ == '__main__':
    app.run()