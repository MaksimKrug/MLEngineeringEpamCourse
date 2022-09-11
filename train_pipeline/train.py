import json

import joblib
import mlflow
import optuna
import pandas as pd
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC

if __name__ == "__main__":
    """
    DATA
    """
    # load data
    df = pd.read_csv("data/preprocessed_data.csv")
    # split
    train_data, test_data = train_test_split(df.sample(10000), random_state=42)
    # TF-IDF
    vectorizer = TfidfVectorizer()
    X_train = vectorizer.fit_transform(train_data["comment_text"])
    X_test = vectorizer.transform(test_data["comment_text"])
    joblib.dump(vectorizer, "vectorizer.joblib")
    # preprocess target
    target_columns = [
        "toxic",
    ]
    y_train = train_data[target_columns].values
    y_test = test_data[target_columns].values

    """
    HYPERPARAMETER OPTIMIZATION
    """

    def objective(trial):
        # params
        model_name = trial.suggest_categorical(
            "model_name", ["RandomForest", "LogisticRegression", "SVC"]
        )
        if model_name == "RandomForest":
            params = {
                "max_depth": trial.suggest_int("max_depth", 2, 32),
                "n_estimators": trial.suggest_int("n_estimators", 50, 150),
                "random_state": trial.suggest_categorical("random_state", [42]),
            }
            model = RandomForestClassifier(**params)
        elif model_name == "LogisticRegression":
            params = {
                "C": trial.suggest_float("C", 1e-8, 1e2),
                "penalty": trial.suggest_categorical(
                    "penalty",
                    [
                        "l1",
                        "l2",
                    ],
                ),
                "solver": trial.suggest_categorical("solver", ["liblinear"]),
                "random_state": trial.suggest_categorical("random_state", [42]),
            }
            model = LogisticRegression(**params)
        elif model_name == "SVC":
            params = {
                "C": trial.suggest_float("C", 1e-8, 1e2),
                "kernel": trial.suggest_categorical(
                    "kernel", ["linear", "rbf", "sigmoid"]
                ),
                "random_state": trial.suggest_categorical("random_state", [42]),
            }
            model = SVC(**params)

        # train model
        model.fit(X_train, y_train.ravel())

        # metrics
        y_pred = model.predict(X_test)
        metrics = {
            "accuracy": accuracy_score(y_test, y_pred),
            "f1": f1_score(y_test, y_pred, average="macro"),
        }

        # mlflow log        
        mlflow.set_tracking_uri("http://mlflow_back:5000/")
        with mlflow.start_run() as mlrun:
            mlflow.set_tag("Model Name", model_name)
            mlflow.log_params(params)
            mlflow.log_metrics(metrics)

        return metrics["f1"]

    # hyperparameters optimization
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=10, timeout=60 * 2)


    """
    SAVE THE BEST MODEL
    """
    print("Best Params:")
    print(study.best_params)
    print("-"*20)

    # params
    params = study.best_params
    model_name = params.pop("model_name")    
    mlflow.log_params(params)
    mlflow.set_tag("Model Name", model_name)
    # train model
    if model_name == "RandomForest":
        model = RandomForestClassifier(**params)
    elif model_name == "LogisticRegression":
        model = LogisticRegression(**params)
    elif model_name == "SVC":
        model = SVC(**params)
    model.fit(X_train, y_train.ravel())
    mlflow.sklearn.log_model(model, model_name)
    
    # metrics
    y_pred = model.predict(X_test)
    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "f1": f1_score(y_test, y_pred, average="macro"),
    }
    mlflow.log_metrics(metrics)
    # get the most important words
    if model_name == "RandomForest":
        all_words = vectorizer.get_feature_names_out()
        words_importnaces = model.feature_importances_
        the_most_important_words = sorted(
            list(zip(all_words, words_importnaces)), key=lambda x: x[1], reverse=True
        )
        temp_data = pd.DataFrame(the_most_important_words, columns=["Words", "Importance"])
        fig = sns.barplot(x="Words", y="Importance", data=temp_data[:10])
        fig.get_figure().savefig("feature_importances.png")
        mlflow.log_figure(fig.get_figure(), "feature_importances.png")
