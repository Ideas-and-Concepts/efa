import joblib
model = joblib.load('models/rf_model.pkl')
# today_features = ... from scraper + historical data
# predictions = model.predict(today_features)
# print predictions