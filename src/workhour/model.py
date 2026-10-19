"""Model definitions, evaluation and persistence."""
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import TransformedTargetRegressor
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import RepeatedKFold, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

MODEL_PATH = Path(__file__).resolve().parents[2] / "models" / "effort_model.joblib"
RANDOM_STATE = 42


def build_candidates() -> dict[str, object]:
    """Candidate estimators. Effort is right-skewed, so all models fit log(effort)."""

    def wrap(estimator, scale: bool = False):
        steps = [("impute", SimpleImputer(strategy="median"))]
        if scale:
            steps.append(("scale", StandardScaler()))
        steps.append(("model", estimator))
        return TransformedTargetRegressor(
            regressor=Pipeline(steps), func=np.log1p, inverse_func=np.expm1
        )

    return {
        "ridge": wrap(Ridge(alpha=1.0), scale=True),
        "random_forest": wrap(
            RandomForestRegressor(n_estimators=300, min_samples_leaf=2, random_state=RANDOM_STATE)
        ),
        "gradient_boosting": wrap(
            GradientBoostingRegressor(n_estimators=200, max_depth=2, learning_rate=0.05,
                                      subsample=0.8, random_state=RANDOM_STATE)
        ),
    }


def cross_validate_models(X: pd.DataFrame, y: pd.Series) -> pd.DataFrame:
    """Repeated 5-fold CV for every candidate; returns mean MAE, RMSE and R2 per model."""
    cv = RepeatedKFold(n_splits=5, n_repeats=5, random_state=RANDOM_STATE)
    scoring = {
        "mae": "neg_mean_absolute_error",
        "rmse": "neg_root_mean_squared_error",
        "r2": "r2",
    }
    rows = []
    for name, est in build_candidates().items():
        res = cross_validate(est, X, y, cv=cv, scoring=scoring)
        rows.append({
            "model": name,
            "mae": -res["test_mae"].mean(),
            "rmse": -res["test_rmse"].mean(),
            "r2": res["test_r2"].mean(),
            "r2_std": res["test_r2"].std(),
        })
    return pd.DataFrame(rows).sort_values("mae").reset_index(drop=True)


def holdout_metrics(model, X_test: pd.DataFrame, y_test: pd.Series) -> dict[str, float]:
    pred = model.predict(X_test)
    return {
        "mae": float(mean_absolute_error(y_test, pred)),
        "rmse": float(np.sqrt(mean_squared_error(y_test, pred))),
        "r2": float(r2_score(y_test, pred)),
    }


def save_model(model, feature_names: list[str], path: Path = MODEL_PATH) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"model": model, "features": feature_names}, path)


def load_model(path: Path = MODEL_PATH) -> tuple[object, list[str]]:
    bundle = joblib.load(path)
    return bundle["model"], bundle["features"]


def predict_effort(model, features: list[str], inputs: dict[str, float]) -> float:
    """Predict effort for a single project; inputs are validated against the training features."""
    missing = set(features) - set(inputs)
    if missing:
        raise ValueError(f"Missing input features: {sorted(missing)}")
    row = pd.DataFrame([[inputs[f] for f in features]], columns=features)
    return float(model.predict(row)[0])
