================================================================================
                    DELIVERY 5 - HOW TO RUN LOCALLY
================================================================================

PREREQUISITES
-------------
- Python 3.11 installed
- pip (Python package manager)


STEP 1: INSTALL DEPENDENCIES
----------------------------
Open PowerShell/Command Prompt in the Delivery5 folder and run:

    pip install -r requirements.txt


STEP 2: RUN THE TRAINING SCRIPT
-------------------------------
This trains 4 models and logs everything to MLflow:

    python train_model.py

Expected output:
- 4 models trained (RandomForest, LogisticRegression, SVM, KNN)
- Best model saved to app/model.joblib
- Metadata saved to app/model_meta.json
- Model registered as "IrisModel" in MLflow


STEP 3: LAUNCH MLFLOW UI
------------------------
Start the MLflow dashboard:

    mlflow ui --backend-store-uri ./mlruns --port 5000

Open in browser: http://localhost:5000

You should see:
- Experiment: iris-model-zoo
- 4 runs with metrics and artifacts
- Registered model: IrisModel v1


STEP 4: RUN STREAMLIT APP
-------------------------
In a NEW terminal (keep MLflow running), start the app:

    streamlit run app/app.py

Open in browser: http://localhost:8501

The app shows:
- Iris flower prediction interface
- Footer with version, model name, MLflow run ID, accuracy, and MLflow link


================================================================================

[X] MLflow UI shows experiment "iris-model-zoo" with 4 runs
[X] MLflow UI shows registered model "IrisModel" v1
[X] Files exist: app/model.joblib and app/model_meta.json
[X] Streamlit app displays model metadata in footer

================================================================================
