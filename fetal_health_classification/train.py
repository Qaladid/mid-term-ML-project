#!/usr/bin/env python
# coding: utf-8

import pandas as pd
import numpy as np
import pickle

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction import DictVectorizer
from sklearn.metrics import precision_score
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score
from sklearn.metrics import classification_report
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import cross_val_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import roc_auc_score
import xgboost as xgb

import matplotlib.pyplot as plt
import seaborn as sns

# Data preparation and Cleaning
df = pd.read_csv('fetal_health.csv')

df.isnull().sum()

# Applying label encoding to the target variable
df['fetal_health'] = df['fetal_health'].astype(int)

fetal_health = {
    1: 'normal', 
    2: 'suspect', 
    3: 'pathological'
}

df.fetal_health = df.fetal_health.map(fetal_health)



# Split the data into training (80%) and test (20%)
df_full_train, df_test = train_test_split(df, test_size=0.2, random_state=1)

# Split the training data into training (75%) and validation (25%)
df_train, df_val = train_test_split(df_full_train, test_size=0.25, random_state=1)

df_full_train = df_full_train.reset_index(drop=True)
df_train = df_train.reset_index(drop=True)
df_val = df_val.reset_index(drop=True)
df_test = df_test.reset_index(drop=True)

# Extract target labels from the original DataFrame (this part is correct)
y_train = df_train['fetal_health'].values
y_val = df_val['fetal_health'].values
y_test = df_test['fetal_health'].values

del df_train['fetal_health']
del df_val['fetal_health']
del df_test['fetal_health']


# Convert DataFrames to dictionaries
dicts_train = df_train.to_dict(orient='records')
dicts_val = df_val.to_dict(orient='records')
dicts_test = df_test.to_dict(orient='records')

# Initialize DictVectorizer
dv = DictVectorizer(sparse=False)

# Convert dictionaries to feature arrays (numerical format)
X_train = dv.fit_transform(dicts_train)
X_val = dv.transform(dicts_val)
X_test = dv.transform(dicts_test)

# Initialize LabelEncoder
le = LabelEncoder()

# Fit the encoder on training labels
y_train_encoded = le.fit_transform(y_train)
y_val_encoded = le.transform(y_val)  # Make sure to transform y_val using the same encoder

# Initialize the model

# Parameters
xgb_params = {
    'objective': 'multi:softprob',  # Use softprob for probabilities
    'num_class': len(np.unique(y_train_encoded)),  # Number of unique classes
    'learning_rate': 0.01,
    'max_depth': 1,  # Can be tuned
    'min_child_weight': 1,
    'n_estimators': 100,  # Number of trees
    'seed': 42,
    'verbosity': 1
}

xgb_model = xgb.XGBClassifier(**xgb_params)

# Train the XGBoost model
xgb_model.fit(X_train, y_train_encoded)

# Train AUC
y_pred_train = xgb_model.predict_proba(X_train)
train_auc = roc_auc_score(y_train_encoded, y_pred_train, multi_class='ovr')
print('Train AUC:', train_auc)

# Validation AUC
y_pred_val = xgb_model.predict_proba(X_val)
val_auc = roc_auc_score(y_val_encoded, y_pred_val, multi_class='ovr')
print('Validation AUC:', val_auc)


# Save the model
with open('fetal_health_model.bin', 'wb') as f_out:
    pickle.dump((dv, xgb_model), f_out)









