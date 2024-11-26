# Mid-Term ML Project: Fetal Health Classification

## Overview

The **Fetal Health Classification** project aims to predict the health status of a fetus based on medical data. The model classifies the fetus into one of three categories:

- **Normal**: Healthy fetal condition.
- **Suspect**: Potential health concerns that require monitoring.
- **Pathological**: Critical health condition that requires immediate medical intervention.

This project involves training a machine learning model to classify the fetal health status based on features such as fetal heart rate, uterine contractions, and maternal health data.

### Problem Description

The dataset for this project contains the following key features:

- Fetal heart rate (FHR)
- Uterine contractions
- Maternal health data (including various maternal health indicators)

The objective of the project is to use this data to predict the fetal health condition. The process involves:

- Preprocessing the dataset to handle missing values and outliers.
- Extracting features that contribute to the prediction.
- Training a classifier using the XGBoost algorithm.
- Deploying the trained model via a Flask API, which is then containerized using Docker and deployed on **Heroku**.

---

## Technologies Used

- **Python 3.11**: The programming language used for data processing, model building, and deployment.
- **XGBoost**: A gradient boosting framework used for classification tasks.
- **Pandas, NumPy**: Libraries used for data manipulation, analysis, and preprocessing.
- **Flask**: A lightweight web framework for creating the API to serve the model.
- **Docker**: Containerization tool used to ensure consistency across different environments.
- **Pipenv**: A Python dependency management tool used to manage project dependencies and virtual environments.
- **Heroku**: Cloud platform used to deploy the trained model.

---

## Project Setup

### Prerequisites

To run the project locally, make sure you have the following installed:

- Python 3.11
- Docker
- Pipenv (for managing Python environments)

### Step 1: Clone the Repository

Clone the project repository/folder to your local machine:

```bash
git clone https://github.com/your-username/mid-term-ML-project.git
cd mid-term-ML-project
```

### Step 2: Install Dependencies

Install project dependencies using Pipenv:

```bash
pipenv install
```
Activate the virtual environment:

```bash
pipenv shell
```

### Step 3: Train the Model

After setting up the environment, navigate to the project directory where your dataset is stored. Use the following Python script to train the XGBoost model:

```bash
python train.py
```
This will train and save the model to a file.


### Step 4: Serve the Model with Flask

Once the model is trained, you can serve it using Flask. Run the following command to start the Flask API:

```bash
python predict.py
```

The model will be available at http://127.0.0.1:9696/ by default.

## Deployment on Heroku

### Step 1: Prepare the Project for Heroku

To deploy the application on Heroku, we need to include the following files:

1. `Procfile`: This file tells Heroku how to run the application.
    In the Procfile, add the following content:

```bash
      web: waitress-serve --listen=0.0.0.0:$PORT app:app
```
This uses `waitress` to serve the Flask application.

2. `requirements`.txt: This file lists the dependencies for the project.
  To generate the `requirements.txt` file, run the following command:

```bash
  pip freeze > requirements.txt
```

### Step 2: Deploy to Heroku

Once you've prepared the necessary files, follow these steps to deploy the project to Heroku:

1. Login to Heroku:
```bash
heroku login
```

2. Create a new Heroku app:
```bash
heroku create fetal-health-classification-dp
```
here, `dp` in the name `create fetal-health-classification-dp`, i abbreviated as deployment.

3. Initialize a Git repository and commit the changes:
```bash
git init
git add .
git commit -m "Initial commit"
```

4. Push the app to Heroku:
```bash 
git push heroku master
```

5. Open the app in your browser:
```bash
heroku open
```

URL to the service i deployed:
```bash
https://fetal-health-classification-dp-dfad861fc4dc.herokuapp.com/ 
```