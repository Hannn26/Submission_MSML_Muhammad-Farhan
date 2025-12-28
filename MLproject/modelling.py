import mlflow
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, log_loss
import random
import numpy as np
import os
import warnings
import sys
 
if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    np.random.seed(40)

    if 'MLFLOW_RUN_ID' in os.environ:
        del os.environ['MLFLOW_RUN_ID']

 
    file_path = sys.argv[3] if len(sys.argv) > 3 else os.path.join(os.path.dirname(os.path.abspath(__file__)), "train_mobile_processed.csv")
    data = pd.read_csv(file_path)


    X_train, X_test, y_train, y_test = train_test_split(
        data.drop('price_range', axis=1),
        data['price_range'],
        random_state=42,
        test_size=0.2
    )


    input_example = X_train.iloc[0:5]
    n_estimators = int(sys.argv[1]) if len(sys.argv) > 1 else 505
    max_depth = int(sys.argv[2]) if len(sys.argv) > 2 else 37
    with mlflow.start_run():
    
        mlflow.autolog()

        model = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, random_state=42)
        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test) 


        accuracy = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, average='weighted')
        precision = precision_score(y_test, y_pred, average='weighted')
        recall = recall_score(y_test, y_pred, average='weighted')
        logloss = log_loss(y_test, y_pred_proba)

        print(f"Accuracy: {accuracy}")
        print(f"F1 Score: {f1}")
    
        mlflow.log_metric("accuracy", accuracy)
        mlflow.log_metric("f1_score", f1)
        mlflow.log_metric("precision", precision)
        mlflow.log_metric("recall", recall)
        mlflow.log_metric("log_loss", logloss)

        mlflow.sklearn.log_model(
            sk_model=model,
            artifact_path="model",
            input_example=input_example

        )

