"""Train, evaluate and persist the effort model.  Usage: python -m workhour.train"""
import json
from pathlib import Path

from sklearn.model_selection import train_test_split

from workhour.data import load_dataset, split_xy
from workhour.model import (
    RANDOM_STATE,
    build_candidates,
    cross_validate_models,
    holdout_metrics,
    save_model,
)

REPORT_PATH = Path(__file__).resolve().parents[2] / "models" / "metrics.json"


def main() -> None:
    X, y = split_xy(load_dataset())
    cv_table = cross_validate_models(X, y)
    print("Repeated 5-fold CV (lower MAE/RMSE is better):")
    print(cv_table.round(3).to_string(index=False))

    best = cv_table.loc[0, "model"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE
    )
    model = build_candidates()[best].fit(X_train, y_train)
    holdout = holdout_metrics(model, X_test, y_test)
    print(f"\nBest model: {best}\nHold-out (20%): {holdout}")

    final = build_candidates()[best].fit(X, y)  # refit on all data for serving
    save_model(final, list(X.columns))
    REPORT_PATH.write_text(json.dumps(
        {"best_model": best, "cv": cv_table.round(4).to_dict("records"), "holdout": holdout},
        indent=2))


if __name__ == "__main__":
    main()
