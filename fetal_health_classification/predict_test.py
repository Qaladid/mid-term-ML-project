#!/usr/bin/env python
# coding: utf-8

import requests

url = 'http://localhost:9696/predict'

fetus = {
    "baseline value": 140.0,
    "accelerations": 0.004,
    "fetal_movement": 0.0,
    "uterine_contractions": 0.004,
    "light_decelerations": 0.0,
    "severe_decelerations": 0.0,
    "prolongued_decelerations": 0.0,
    "abnormal_short_term_variability": 80.0,
    "mean_value_of_short_term_variability": 0.2,
    "percentage_of_time_with_abnormal_long_term_variability": 36.0,
    "mean_value_of_long_term_variability": 2.2,
    "histogram_width": 18.0,
    "histogram_min": 140.0,
    "histogram_max": 158.0,
    "histogram_number_of_peaks": 1.0,
    "histogram_number_of_zeroes": 0.0,
    "histogram_mode": 147.0,
    "histogram_mean": 148.0,
    "histogram_median": 149.0,
    "histogram_variance": 1.0,
    "histogram_tendency": 0.0
 }


response = requests.post(url, json=fetus).json()
print(response)

# Handle the response to generate the message based on the classification
health_classification = response['health_classification']
probabilities = response['probabilities']
max_class_label = health_classification  # This is what you already got from the response
max_class_prob = probabilities[max_class_label]

# Custom message based on the classification
if max_class_label == 'pathological':
    custom_message = f"The fetus needs more care. The probability of being pathological is {max_class_prob*100:.2f}%."
elif max_class_label == 'normal':
    custom_message = f"The fetus is healthy. The probability of being normal is {max_class_prob*100:.2f}%. Keep monitoring."
else:  # suspect
    custom_message = f"Alarm! The fetus is in a suspect state. Probability of being suspect is {max_class_prob*100:.2f}%. Caution is needed."

# Print the response and the custom message
print(response)
print("Custom Message:", custom_message)





