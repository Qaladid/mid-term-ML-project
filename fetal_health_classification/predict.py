
import pickle

with open('fetal_health_model.bin', 'rb') as f_in:
    dv, xgb_model = pickle.load(f_in)


fetus = {
    'baseline value': 140.0,
    'accelerations': 0.004,
    'fetal_movement': 0.0,
    'uterine_contractions': 0.004,
    'light_decelerations': 0.0,
    'severe_decelerations': 0.0,
    'prolongued_decelerations': 0.0,
    'abnormal_short_term_variability': 80.0,
    'mean_value_of_short_term_variability': 0.2,
    'percentage_of_time_with_abnormal_long_term_variability': 36.0,
    'mean_value_of_long_term_variability': 2.2,
    'histogram_width': 18.0,
    'histogram_min': 140.0,
    'histogram_max': 158.0,
    'histogram_number_of_peaks': 1.0,
    'histogram_number_of_zeroes': 0.0,
    'histogram_mode': 147.0,
    'histogram_mean': 148.0,
    'histogram_median': 149.0,
    'histogram_variance': 1.0,
    'histogram_tendency': 0.0
 }

X = dv.transform([fetus])
y_pred_proba = xgb_model.predict_proba(X)[0]  # All class probabilities
y_pred_specific = xgb_model.predict_proba(X)[0, 1]  # Probability of class 1
y_pred_class = xgb_model.predict(X)[0]  # Most likely class

print('Health Class Probabilities:', y_pred_proba)
print('Probability of Suspect Class:', y_pred_specific)
print('Predicted Health Class:', y_pred_class)





