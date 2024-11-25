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
- Deploying the trained model via a Flask API, which is then containerized using Docker.

---

## Technologies Used

- **Python 3.11**: The programming language used for data processing, model building, and deployment.
- **XGBoost**: A gradient boosting framework used for classification tasks.
- **Pandas, NumPy**: Libraries used for data manipulation, analysis, and preprocessing.
- **Flask**: A lightweight web framework for creating the API to serve the model.
- **Docker**: Containerization tool used to ensure consistency across different environments.
- **Pipenv**: A Python dependency management tool used to manage project dependencies and virtual environments.

---

## Project Setup

### Prerequisites

To run the project locally, make sure you have the following installed:

- Python 3.11
- Docker
- Pipenv (for managing Python environments)

### Step 1: Clone the Repository

Clone the project repository to your local machine:

```bash
git clone https://github.com/your-username/mid-term-ML-project.git
cd mid-term-ML-project
