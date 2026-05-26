import pandas as pd
import numpy as np
import joblib
import sys
from pathlib import Path

from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer, make_column_selector
from sklearn.preprocessing import FunctionTransformer, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression

from sklearn.ensemble import (
    HistGradientBoostingRegressor,
    HistGradientBoostingClassifier,
)
from sklearn.metrics import (
    r2_score,
    mean_absolute_error,
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
)

import seaborn as sns
import matplotlib.pyplot as plt

from utils import add_engagement



BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "model"
DATA_PATH = BASE_DIR.parent / "data" / "student_performance (1).csv"
FEATURE_COLUMNS = [
    "weekly_self_study_hours",
    "attendance_percentage",
    "class_participation",
]
REQUIRED_COLUMNS = FEATURE_COLUMNS + ["total_score"]
RANDOM_STATE = 42
TEST_SIZE = 0.2
TRAIN_SAMPLE_SIZE = 120_000
CV_SAMPLE_SIZE = 30_000

feature_engineering = FunctionTransformer(add_engagement)


def score_to_grade(scores):
    scores = np.asarray(scores)
    return np.select(
        [
            scores >= 95,
            scores >= 90,
            scores >= 85,
            scores >= 80,
            scores >= 75,
            scores >= 70,
            scores >= 60,
        ],
        ["A+", "A", "B+", "B", "C+", "C", "D"],
        default="F",
    )


GRADE_LABELS = ["A+", "A", "B+", "B", "C+", "C", "D", "F"]


class ScoreThresholdClassifier(BaseEstimator, ClassifierMixin):
    """Predict fine-grained grades by learning total score, then thresholding."""

    def __init__(self, regressor):
        self.regressor = regressor

    def fit(self, X, y):
        if not isinstance(y, pd.Series):
            y = pd.Series(y)
        self.classes_ = np.array(GRADE_LABELS)
        numeric_target = y.map(self._grade_to_score_proxy).astype(float)
        self.regressor_ = clone(self.regressor)
        self.regressor_.fit(X, numeric_target)
        return self

    def predict(self, X):
        predicted_scores = self.regressor_.predict(X)
        return score_to_grade(predicted_scores)

    @staticmethod
    def _grade_to_score_proxy(label):
        grade_midpoints = {
            "A+": 97.5,
            "A": 92.0,
            "B+": 87.0,
            "B": 82.0,
            "C+": 77.0,
            "C": 72.0,
            "D": 65.0,
            "F": 45.0,
        }
        return grade_midpoints[label]


# Make pickles loadable when this file is run as a script.
ScoreThresholdClassifier.__module__ = "train"
sys.modules.setdefault("train", sys.modules[__name__])


def print_model_metrics(model_name, y_true, y_pred):
    accuracy = accuracy_score(y_true, y_pred)
    balanced_accuracy = balanced_accuracy_score(y_true, y_pred)

    print(f"\n----- {model_name} -----")
    print("Accuracy:", accuracy)
    print("Balanced Accuracy:", balanced_accuracy)
    print("\nClassification Report:")
    print(classification_report(y_true, y_pred, labels=GRADE_LABELS, zero_division=0))

    return {
        "name": model_name,
        "accuracy": accuracy,
        "balanced_accuracy": balanced_accuracy,
        "predictions": y_pred,
    }


def load_training_data():
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Training data not found: {DATA_PATH}")

    df = pd.read_csv(DATA_PATH)
    missing_columns = sorted(set(REQUIRED_COLUMNS) - set(df.columns))
    if missing_columns:
        raise ValueError(f"Training data is missing columns: {missing_columns}")

    return df


def make_preprocessors():
    num_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="mean")),
    ])

    scaled_num_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="mean")),
        ("scaler", StandardScaler()),
    ])

    preprocessor = ColumnTransformer([
        ("num", num_pipeline, make_column_selector(dtype_include=np.number))
    ])

    scaled_preprocessor = ColumnTransformer([
        ("num", scaled_num_pipeline, make_column_selector(dtype_include=np.number))
    ])

    return preprocessor, scaled_preprocessor


def make_models(preprocessor, scaled_preprocessor):
    return {
        "Logistic Regression": Pipeline([
            ("feature_engineering", feature_engineering),
            ("preprocessing", scaled_preprocessor),
            ("model", LogisticRegression(
                max_iter=3000,
                class_weight="balanced",
                random_state=RANDOM_STATE,
            ))
        ]),
        "HistGradientBoosting": Pipeline([
            ("feature_engineering", feature_engineering),
            ("preprocessing", preprocessor),
            ("model", HistGradientBoostingClassifier(
                learning_rate=0.05,
                max_depth=6,
                max_iter=220,
                min_samples_leaf=8,
                l2_regularization=0.1,
                random_state=RANDOM_STATE,
            ))
        ]),
        "Score Threshold": Pipeline([
            ("feature_engineering", feature_engineering),
            ("preprocessing", preprocessor),
            ("model", ScoreThresholdClassifier(
                HistGradientBoostingRegressor(
                    learning_rate=0.06,
                    max_iter=220,
                    max_leaf_nodes=31,
                    l2_regularization=0.05,
                    random_state=RANDOM_STATE,
                )
            ))
        ]),
    }


def make_cv_data(X_train, y_train_g):
    class_counts = y_train_g.value_counts()
    min_class_count = int(class_counts.min())
    if min_class_count < 2:
        raise ValueError(
            "Each grade needs at least 2 training rows for stratified validation. "
            f"Class counts: {class_counts.to_dict()}"
        )

    cv_splits = min(5, min_class_count)
    cv = StratifiedKFold(n_splits=cv_splits, shuffle=True, random_state=RANDOM_STATE)

    cv_sample_size = min(len(X_train), CV_SAMPLE_SIZE)
    if cv_sample_size < len(X_train):
        X_cv, _, y_cv, _ = train_test_split(
            X_train,
            y_train_g,
            train_size=cv_sample_size,
            random_state=RANDOM_STATE,
            stratify=y_train_g,
        )
    else:
        X_cv = X_train
        y_cv = y_train_g

    return X_cv, y_cv, cv


def sample_training_frame(df, y_grade):
    if len(df) <= TRAIN_SAMPLE_SIZE:
        return df, y_grade

    stratify_target = y_grade if y_grade.value_counts().min() >= 2 else None
    sampled_df, _, sampled_y_grade, _ = train_test_split(
        df,
        y_grade,
        train_size=TRAIN_SAMPLE_SIZE,
        random_state=RANDOM_STATE,
        stratify=stratify_target,
    )

    return sampled_df.reset_index(drop=True), sampled_y_grade.reset_index(drop=True)


def main():
    # ---------------- LOAD DATA ----------------
    df = load_training_data()

    y_score = df["total_score"]

    # Train the classifier on the same fine-grained grade buckets used by the app.
    y_grade = pd.Series(score_to_grade(y_score), index=df.index)
    df, y_grade = sample_training_frame(df, y_grade)

    # Keep training features aligned with PredictionInput in main.py.
    X = df[FEATURE_COLUMNS]
    y_score = df["total_score"]

    stratify_target = y_grade if y_grade.value_counts().min() >= 2 else None

    # ---------------- PREPROCESSING ----------------
    preprocessor, scaled_preprocessor = make_preprocessors()

    # ---------------- SPLIT ----------------
    X_train, X_test, y_train_s, y_test_s, y_train_g, y_test_g = train_test_split(
        X,
        y_score,
        y_grade,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=stratify_target,
    )

    # ---------------- REGRESSION MODEL ----------------
    reg_model = Pipeline([
        ("feature_engineering", feature_engineering),
        ("preprocessing", preprocessor),
        ("model", HistGradientBoostingRegressor(
            learning_rate=0.06,
            max_iter=300,
            max_leaf_nodes=31,
            l2_regularization=0.05,
            random_state=RANDOM_STATE,
        ))
    ])

    reg_model.fit(X_train, y_train_s)
    y_pred_s = reg_model.predict(X_test)
    y_pred_g_from_score = score_to_grade(y_pred_s)

    print("----- REGRESSION -----")
    print("R2 Score:", r2_score(y_test_s, y_pred_s))
    print("MAE:", mean_absolute_error(y_test_s, y_pred_s))
    print("Grade Accuracy From Predicted Score:", accuracy_score(y_test_g, y_pred_g_from_score))
    print("Balanced Accuracy From Predicted Score:", balanced_accuracy_score(y_test_g, y_pred_g_from_score))

    # ---------------- CLASSIFICATION MODELS ----------------
    models = make_models(preprocessor, scaled_preprocessor)
    X_cv, y_cv, cv = make_cv_data(X_train, y_train_g)

    results = []
    trained_models = {}

    for model_name, model in models.items():
        cv_scores = cross_val_score(
            model,
            X_cv,
            y_cv,
            cv=cv,
            scoring="balanced_accuracy",
            n_jobs=1,
        )

        print(f"\n{model_name} CV Balanced Accuracy: {cv_scores.mean():.4f} +/- {cv_scores.std():.4f}")

        model.fit(X_train, y_train_g)
        y_pred_g = model.predict(X_test)

        trained_models[model_name] = model
        result = print_model_metrics(model_name, y_test_g, y_pred_g)
        result["cv_balanced_accuracy"] = cv_scores.mean()
        results.append(result)

    best_result = max(
        results,
        key=lambda result: (
            result["cv_balanced_accuracy"],
            result["balanced_accuracy"],
            result["accuracy"],
        ),
    )
    best_model_name = best_result["name"]
    clf_model = trained_models[best_model_name]

    print("\n----- BEST CLASSIFIER -----")
    print("Selected Model:", best_model_name)
    print("Selection Metric: CV balanced accuracy, then test balanced accuracy, then accuracy")
    print("Best CV Balanced Accuracy:", best_result["cv_balanced_accuracy"])
    print("Best Accuracy:", best_result["accuracy"])
    print("Best Balanced Accuracy:", best_result["balanced_accuracy"])

    # ---------------- CONFUSION MATRIX ----------------
    cm = confusion_matrix(y_test_g, best_result["predictions"], labels=GRADE_LABELS)

    MODEL_DIR.mkdir(exist_ok=True)
    plt.figure(figsize=(6, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=GRADE_LABELS, yticklabels=GRADE_LABELS)
    plt.title(f"Confusion Matrix - {best_model_name}")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.tight_layout()
    plt.savefig(MODEL_DIR / "confusion_matrix.png", dpi=150)
    plt.close()

    # ---------------- SAVE MODELS ----------------
    joblib.dump(reg_model, MODEL_DIR / "score_model.pkl", compress=3)
    joblib.dump(clf_model, MODEL_DIR / "grade_model.pkl", compress=3)

    print("\nModels saved successfully!")


if __name__ == "__main__":
    main()
