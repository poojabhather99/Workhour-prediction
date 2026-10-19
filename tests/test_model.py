import numpy as np
import pytest

from workhour.data import load_dataset, split_xy
from workhour.model import build_candidates, predict_effort


@pytest.fixture(scope="module")
def xy():
    return split_xy(load_dataset())


def test_dataset_shape_and_no_nans(xy):
    X, y = xy
    assert len(X) == len(y) == 81
    assert not X.isna().any().any()
    assert "id" not in X.columns and "Effort" not in X.columns


@pytest.mark.parametrize("name", ["ridge", "random_forest", "gradient_boosting"])
def test_candidates_fit_and_predict_positive(xy, name):
    X, y = xy
    model = build_candidates()[name].fit(X, y)
    pred = model.predict(X)
    assert pred.shape == (len(X),) and np.all(pred > 0)


def test_predict_effort_validates_inputs(xy):
    X, y = xy
    model = build_candidates()["ridge"].fit(X, y)
    row = X.iloc[0].to_dict()
    assert predict_effort(model, list(X.columns), row) > 0
    row.pop("Length")
    with pytest.raises(ValueError):
        predict_effort(model, list(X.columns), row)


def test_predict_does_not_mutate_training_data(xy):
    X, y = xy
    before = X.copy()
    model = build_candidates()["ridge"].fit(X, y)
    predict_effort(model, list(X.columns), X.iloc[0].to_dict())
    assert X.equals(before)
