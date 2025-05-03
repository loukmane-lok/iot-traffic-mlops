# iot-traffic-mlops

pip install zenml["server"]
zenml login --local

### mlflow integration

zenml integration install mlflow -y
zenml experiment-tracker register mlflow_tracker_customer --flavor=mlflow
zenml model-deployer register mlflow_customer --flavor=mlflow
zenml stack register mlflow_stack_customer -a default -o default -d mlflow_customer -e mlflow_tracker_customer --set
