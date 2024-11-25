
import pickle
from flask import Flask
from flask import request, jsonify


with open('fetal_health_model.bin', 'rb') as f_in:
    dv, xgb_model = pickle.load(f_in)

app = Flask('health_classification')

@app.route('/predict', methods=['POST'])
def predict():
    fetus = request.get_json()  # Get input data from the request

    X = dv.transform([fetus])  # Transform the data using the vectorizer

    # Get the full class probabilities (for all classes)
    y_pred_proba = xgb_model.predict_proba(X)[0]

    # Get the predicted class (most likely class)
    y_pred_class = xgb_model.predict(X)[0]

    # Get the specific probability for class 1 (assuming 1 is the class of interest)
    y_pred_specific = float(y_pred_proba[1])  # Probability of class 1

    # Logic for classification decision (custom thresholds can be set here)
    if y_pred_class == 0:  # If the most likely class is 0
        health_classification = 'normal'
    elif y_pred_class == 1:  # If the most likely class is 1
        health_classification = 'suspect'
    elif y_pred_class == 2:  # If the most likely class is 2
        health_classification = 'pathological'

    result = {
        'health_classification': health_classification,
        'probabilities': {
            'normal': float(y_pred_proba[0]),
            'suspect': float(y_pred_proba[1]),
            'pathological': float(y_pred_proba[2])
        },
        'specific_class_probability': y_pred_specific
    }

    return jsonify(result)


if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0', port=9696)

