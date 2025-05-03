# IoT Traffic Volume Prediction with MLOps

This project demonstrates a full MLOps workflow for predicting traffic volume using an XGBoost regression model. Built with **ZenML**, the pipeline includes steps for data ingestion, preprocessing, training, evaluation, and deployment using **MLflow**.

---

## Problem Statement

Predict the number of vehicles using historical IoT traffic sensor data with features like `Hour`, `Weekday`, `Day`, `Month`, and `Week`.

---

## 🛠️ Tech Stack

- **ZenML** – For pipeline orchestration
- **XGBoost** – Regressor model
- **MLflow** – Model deployment and tracking
- **Docker** – Containerized pipeline steps
- **pandas / scikit-learn** – Data handling and preprocessing

---

## ⚙️ Pipelines

### 1. Training Pipeline

> Trains and evaluates the model.

**Steps**:

- Load data
- Clean/split data
- Train model (`XGBRegressor`)
- Evaluate model (RMSE)

```python
training_pipeline(data_path="data/train_ML_IOT.csv")
```

---

### 2. Continuous Deployment Pipeline

> Trains a model and deploys it via MLflow if RMSE < `max_rmse`.

**Steps**:

- Ingest and clean data
- Train and evaluate model
- Trigger deployment based on RMSE
- Deploy model using MLflow

```python
continuous_deployment_pipeline(max_rmse=50.0)
```

---

## 🐳 Docker Integration

Each step in the pipeline runs in a containerized environment using ZenML's `DockerSettings`.

---

## Getting Started

### 1. Clone and Install

```bash
git clone https://github.com/lokk798/iot-traffic-mlops.git
cd iot-traffic-mlops
pip install -r requirements.txt
```

### 2. Initialize ZenML Repository

```bash
zenml init
zenml login --local

```

### 3. Register Stack Components

```bash
zenml integration install mlflow -y
zenml experiment-tracker register mlflow_tracker_customer --flavor=mlflow
zenml model-deployer register mlflow_customer --flavor=mlflow
zenml stack register mlflow_stack_customer -a default -o default -d mlflow_customer -e mlflow_tracker_customer --set


```

### 4. Run Pipelines

```bash
python run_training.py
python run_deployment.py
```

---

## 📌 Future Improvements

- Monitor drift in deployed models.
- Add CI/CD via GitHub Actions.

---

## Author

**Loukmane Daoudi** – [LinkedIn](https://www.linkedin.com/in/loukmane-daoudi/)

---

## 📄 License

This project is licensed under the MIT License.
