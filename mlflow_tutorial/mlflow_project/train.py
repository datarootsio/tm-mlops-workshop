import sys

import mlflow
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import f1_score, matthews_corrcoef, precision_score, recall_score
from sklearn.model_selection import cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# to run the project
# mlflow run ./src-mlflow-project --env-manager=local -P \
# csv_file=../data/heart.csv -P  max_depth=5 --experiment-id= ..


def parse_parameters():
    csv_file = str(sys.argv[1])
    max_depth = int(sys.argv[2])
    max_features = int(sys.argv[3])
    return csv_file, max_depth, max_features


if __name__ == "__main__":
    # parameters
    csv_file, max_depth, max_features = parse_parameters()

    data = pd.read_csv(csv_file)

    cat_attr = ["sex", "cp", "fbs", "restecg", "exang", "slope"]
    num_attr = ["age", "trestbps", "chol", "thalach", "oldpeak", "ca", "thal"]

    num_pipeline = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("std_scaler", StandardScaler()),
        ]
    )

    full_pipeline = ColumnTransformer(
        [("num", num_pipeline, num_attr), ("cat", OneHotEncoder(), cat_attr)]
    )

    X_train = data.drop("target", axis=1)
    y_train = data.target

    X_train = full_pipeline.fit_transform(X_train)

    mlflow.sklearn.autolog()
    with mlflow.start_run():
        model = RandomForestClassifier(random_state=42)

        y_preds = cross_val_predict(model, X_train, y_train, cv=3)

        mlflow.log_metric("Precision", precision_score(y_train, y_preds))
        mlflow.log_metric("Recall", recall_score(y_train, y_preds))
        mlflow.log_metric("F1 score", f1_score(y_train, y_preds))
        mlflow.log_metric("Matthews Correlation", matthews_corrcoef(y_train, y_preds))
