def get_aqi_alert(aqi):
    """
    Convert an AQI value into an air quality category
    and a corresponding alert message.
    """

    if aqi <= 50:
        category = "Good"
        alert = "Air quality is good. Outdoor activities are safe."

    elif aqi <= 100:
        category = "Satisfactory"
        alert = "Air quality is acceptable for most people."

    elif aqi <= 200:
        category = "Moderate"
        alert = "Sensitive individuals should reduce prolonged outdoor exposure."

    elif aqi <= 300:
        category = "Poor"
        alert = "Air pollution is high. Sensitive groups should limit outdoor activity."

    elif aqi <= 400:
        category = "Very Poor"
        alert = "Air pollution is very high. Reduce outdoor activities."

    else:
        category = "Severe"
        alert = "Severe air pollution detected. Avoid prolonged outdoor exposure."

    return {
        "aqi": aqi,
        "category": category,
        "alert": alert
    }


if __name__ == "__main__":

    sample_aqi = 275

    result = get_aqi_alert(sample_aqi)

    print("Predicted AQI:", result["aqi"])
    print("AQI Category:", result["category"])
    print("Alert:", result["alert"])