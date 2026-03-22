# 🔐 Phishing Website Detection System (MLOps Project)

## 📌 Overview

This project is an end-to-end Machine Learning system designed to detect phishing websites using a structured MLOps pipeline. It goes beyond simple model training by incorporating data ingestion, validation, transformation, model training, evaluation, deployment, and monitoring.

---

## 🚀 Key Features

* Modular ML pipeline (Ingestion → Validation → Transformation → Training → Evaluation → Deployment)
* Custom logging and exception handling
* MongoDB integration for data storage
* Data validation with schema checks and data drift detection
* KNN-based missing value imputation
* Multiple model training with hyperparameter tuning
* MLflow experiment tracking
* FastAPI-based deployment
* Batch prediction support
* AWS S3 integration for artifact storage

---

## 🧠 Project Architecture

```
MongoDB → Data Ingestion → Data Validation → Data Transformation → Model Training → Model Evaluation → Model Pusher → FastAPI Deployment → Prediction
```

---

## 📂 Project Structure

```
NetworkSecurity/
│
├── components/            # Core pipeline steps
│   ├── data_ingestion.py
│   ├── data_validation.py
│   ├── data_transformation.py
│   ├── model_trainer.py
│
├── entity/
│   ├── config_entity.py   # Config classes (input settings)
│   ├── artifact_entity.py # Artifact classes (outputs of steps)
│
├── constants/             # All constant values (paths, names)
├── utils/                 # Utility functions
│   ├── common.py
│   ├── ml_utils.py
│
├── pipeline/              # Training & prediction pipelines
├── app.py                 # FastAPI application
├── logging.py             # Logging configuration
├── exception.py           # Custom exception handling
├── setup.py               # Project packaging
└── requirements.txt
```

---

## ⚙️ Core Concepts

### 1. Config, Component, Artifact Pattern

Each pipeline step follows a structured design:

* **Config**: Defines input parameters and file paths
* **Component**: Executes the logic
* **Artifact**: Stores output metadata (paths, status)

---

### 2. Data Ingestion

* Fetches data from MongoDB
* Converts JSON data to Pandas DataFrame
* Saves raw data and splits into train/test datasets

---

### 3. Data Validation

* Validates schema consistency
* Detects data drift using statistical tests
* Ensures reliability of incoming data

---

### 4. Data Transformation

* Separates features and target
* Handles missing values using KNN Imputer
* Prepares data for model training

---

### 5. Model Training

* Trains multiple ML models
* Performs hyperparameter tuning using GridSearchCV
* Selects best-performing model

---

### 6. Model Evaluation

* Compares model performance on train and test data
* Detects overfitting and underfitting

---

### 7. Model Pusher

* Saves trained model and preprocessing object
* Prepares artifacts for deployment

---

## 📊 Experiment Tracking (MLflow)

* Logs parameters, metrics, and models
* Enables comparison of multiple experiments
* Integrated with DagsHub for remote tracking

---

## 🌐 Deployment (FastAPI)

### Endpoints:

* `/train` → Triggers full training pipeline
* `/predict` → Accepts CSV input and returns predictions

---

## ☁️ Cloud Integration

* AWS S3 for artifact storage
* Supports scalable deployment

---

## 🔍 Logging & Exception Handling

### Logging

* Centralized logging system
* Tracks pipeline execution steps

### Exception Handling

* Custom exception class
* Captures file name and line number
* Improves debugging in production systems

---

## 📦 Installation

```bash
pip install -r requirements.txt
```

---

## ▶️ Running the Project

### Start FastAPI server

```bash
uvicorn app:app --reload
```

### Access API docs

```
http://127.0.0.1:8000/docs
```

---

## 📈 Use Case

This system is designed for detecting phishing websites based on structured input features. It can be extended to other classification problems with minimal changes.

---

## 💡 Future Improvements

* Real-time prediction API
* CI/CD pipeline integration
* Kubernetes deployment
* Advanced feature engineering

---

## 🧾 Conclusion

This project demonstrates a production-ready ML system with modular design, scalability, and deployment capabilities. It highlights best practices in MLOps, including experiment tracking, pipeline structuring, and cloud integration.

---

## 👨‍💻 Author

Khursh-eed
