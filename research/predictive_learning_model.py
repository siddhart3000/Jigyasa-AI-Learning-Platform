from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from sklearn.ensemble import (
    GradientBoostingRegressor,
    RandomForestRegressor,
)

from sklearn.linear_model import LinearRegression

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)

from sklearn.model_selection import (
    cross_val_score,
    train_test_split,
)

from sklearn.pipeline import Pipeline

from sklearn.preprocessing import StandardScaler


# =========================================================
# CONFIG
# =========================================================

FEATURES = [
    "study_time_minutes",
    "quiz_attempts",
    "quiz_accuracy",
    "ai_questions_asked",
    "videos_watched",
    "practice_attempts",
    "engagement_score",
    "topic_mastery",
    "study_consistency",
]

TARGET = "exam_score"


# =========================================================
# METRICS
# =========================================================

def evaluate_model(
    model_name: str,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    train_r2: float,
    cv_score: float,
) -> dict:

    mae = mean_absolute_error(y_true, y_pred)

    rmse = mean_squared_error(
        y_true,
        y_pred,
        squared=False
    )

    r2 = r2_score(
        y_true,
        y_pred
    )

    return {
        "Model": model_name,
        "MAE": round(float(mae), 4),
        "RMSE": round(float(rmse), 4),
        "R2_Test": round(float(r2), 4),
        "R2_Train": round(float(train_r2), 4),
        "CV_R2_Mean": round(float(cv_score), 4),
    }


# =========================================================
# FEATURE IMPORTANCE
# =========================================================

def export_feature_importance(
    model,
    model_name: str,
    output_dir: Path
) -> None:

    if not hasattr(model, "feature_importances_"):
        return

    importance_df = pd.DataFrame(
        {
            "Feature": FEATURES,
            "Importance": model.feature_importances_,
        }
    )

    importance_df = importance_df.sort_values(
        by="Importance",
        ascending=False
    )

    out_file = output_dir / f"{model_name.lower().replace(' ', '_')}_feature_importance.csv"

    importance_df.to_csv(
        out_file,
        index=False
    )

    print(f"\nSaved feature importance -> {out_file}")


# =========================================================
# PREDICTION EXPORT
# =========================================================

def export_predictions(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    model_name: str,
    output_dir: Path
) -> None:

    pred_df = pd.DataFrame(
        {
            "Actual_Exam_Score": y_true,
            "Predicted_Exam_Score": np.round(y_pred, 2),
            "Prediction_Error": np.round(y_true - y_pred, 2),
        }
    )

    out_file = output_dir / f"{model_name.lower().replace(' ', '_')}_predictions.csv"

    pred_df.to_csv(
        out_file,
        index=False
    )

    print(f"Saved predictions -> {out_file}")


# =========================================================
# MAIN
# =========================================================

def main() -> None:

    # =====================================================
    # ARGUMENTS
    # =====================================================

    ap = argparse.ArgumentParser()

    ap.add_argument(
        "--data",
        type=str,
        default="research/student_learning_data.csv"
    )

    ap.add_argument(
        "--out",
        type=str,
        default="research/model_results.csv"
    )

    ap.add_argument(
        "--modeldir",
        type=str,
        default="research/models"
    )

    ap.add_argument(
        "--seed",
        type=int,
        default=42
    )

    args = ap.parse_args()

    # =====================================================
    # LOAD DATA
    # =====================================================

    df = pd.read_csv(args.data)

    print("\nDataset Loaded Successfully")
    print(f"Rows: {len(df)}")
    print(f"Columns: {len(df.columns)}")

    # =====================================================
    # HANDLE MISSING VALUES
    # =====================================================

    df = df.fillna(
        df.median(numeric_only=True)
    )

    # =====================================================
    # FEATURES + TARGET
    # =====================================================

    X = df[FEATURES]

    y = df[TARGET]

    # =====================================================
    # TRAIN TEST SPLIT
    # =====================================================

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=args.seed,
    )

    # =====================================================
    # MODELS
    # =====================================================

    models = [

        (
            "Linear Regression",

            Pipeline([
                ("scaler", StandardScaler()),
                ("model", LinearRegression())
            ])
        ),

        (
            "Random Forest",

            RandomForestRegressor(
                n_estimators=300,
                max_depth=15,
                min_samples_split=4,
                min_samples_leaf=2,
                random_state=args.seed,
            )
        ),

        (
            "Gradient Boosting",

            GradientBoostingRegressor(
                n_estimators=300,
                learning_rate=0.05,
                max_depth=5,
                random_state=args.seed,
            )
        ),
    ]

    # =====================================================
    # OUTPUT DIRECTORIES
    # =====================================================

    result_output = Path(args.out)

    result_output.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    model_dir = Path(args.modeldir)

    model_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    # =====================================================
    # TRAINING LOOP
    # =====================================================

    results = []

    best_model = None

    best_r2 = -999

    best_model_name = ""

    print("\n================================================")
    print("MODEL TRAINING + EVALUATION")
    print("================================================")

    for name, model in models:

        print(f"\nTraining: {name}")

        # =================================================
        # CROSS VALIDATION
        # =================================================

        cv_scores = cross_val_score(
            model,
            X,
            y,
            cv=5,
            scoring="r2"
        )

        cv_mean = np.mean(cv_scores)

        # =================================================
        # TRAIN
        # =================================================

        model.fit(
            X_train,
            y_train
        )

        # =================================================
        # PREDICTIONS
        # =================================================

        train_pred = model.predict(X_train)

        test_pred = model.predict(X_test)

        # =================================================
        # SCORES
        # =================================================

        train_r2 = r2_score(
            y_train,
            train_pred
        )

        metrics = evaluate_model(
            model_name=name,
            y_true=y_test.to_numpy(),
            y_pred=test_pred,
            train_r2=train_r2,
            cv_score=cv_mean,
        )

        results.append(metrics)

        # =================================================
        # SAVE BEST MODEL
        # =================================================

        if metrics["R2_Test"] > best_r2:

            best_r2 = metrics["R2_Test"]

            best_model = model

            best_model_name = name

        # =================================================
        # EXPORT PREDICTIONS
        # =================================================

        export_predictions(
            y_true=y_test.to_numpy(),
            y_pred=test_pred,
            model_name=name,
            output_dir=result_output.parent
        )

        # =================================================
        # FEATURE IMPORTANCE
        # =================================================

        actual_model = model

        if isinstance(model, Pipeline):
            actual_model = model.named_steps["model"]

        export_feature_importance(
            actual_model,
            name,
            result_output.parent
        )

    # =====================================================
    # RESULTS DATAFRAME
    # =====================================================

    results_df = pd.DataFrame(results)

    results_df = results_df.sort_values(
        by="R2_Test",
        ascending=False
    )

    results_df.to_csv(
        result_output,
        index=False
    )

    # =====================================================
    # SAVE BEST MODEL
    # =====================================================

    best_model_path = model_dir / "best_exam_score_model.pkl"

    joblib.dump(
        best_model,
        best_model_path
    )

    # =====================================================
    # CONSOLE OUTPUT
    # =====================================================

    print("\n================================================")
    print("FINAL MODEL PERFORMANCE")
    print("================================================\n")

    print(results_df.to_string(index=False))

    print("\n================================================")
    print("BEST MODEL")
    print("================================================")

    print(f"\nBest Model: {best_model_name}")
    print(f"Best Test R2: {best_r2:.4f}")

    print(f"\nSaved best model -> {best_model_path}")

    print(f"\nSaved evaluation results -> {result_output}")

    print("\nResearch-grade ML pipeline completed successfully.\n")


# =========================================================
# ENTRY
# =========================================================

if __name__ == "__main__":
    main()