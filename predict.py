import joblib
import pandas as pd

model = joblib.load("models/aqi_model.pkl")

sample_input = pd.DataFrame([{
    'AQI_today': 210,
    'PM2.5_today': 115.5,
    'PM10_today': 180.0,
    'AQI_3day_avg': 195.0,
    'AQI_7day_avg': 185.0,
    'Month': 11,
    'DayOfWeek': 3
}])

predicted_aqi = model.predict(sample_input)[0]

print("=" * 40)
print(f" Predicted Tomorrow's AQI: {predicted_aqi:.2f}")
print("=" * 40)
