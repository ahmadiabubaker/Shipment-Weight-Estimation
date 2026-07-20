import os
import sys

import joblib
import pytest
from fastapi.testclient import TestClient

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from shipment_weight.data_gen import generate_shipments  # noqa: E402
from shipment_weight.train import make_pipeline, MODEL_CANDIDATES, split_data  # noqa: E402


@pytest.fixture(scope="module")
def client(tmp_path_factory):
    df = generate_shipments(n_shipments=500, seed=1)
    X_train, X_test, y_train, y_test = split_data(df, seed=1)
    pipe = make_pipeline(MODEL_CANDIDATES["ridge"])
    pipe.fit(X_train, y_train)
    residual_std = float((y_test.values - pipe.predict(X_test)).std())

    model_path = tmp_path_factory.mktemp("models") / "model.joblib"
    joblib.dump(
        {"pipeline": pipe, "model_type": "ridge", "model_version": "test", "residual_std": residual_std, "trained_on_rows": len(X_train)},
        model_path,
    )
    os.environ["MODEL_PATH"] = str(model_path)

    from api.main import app

    with TestClient(app) as c:
        yield c


def test_health(client):
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json()["model_loaded"] is True


def test_predict_returns_weight_and_confidence(client):
    payload = {
        "theoretical_weight_oz": 100.0,
        "item_count": 5,
        "total_item_volume_in3": 800.0,
        "item_categories": "electronics,apparel",
        "category_mode": "electronics",
        "carton_type": "L_16x12x10",
        "ship_method": "GROUND",
        "packing_material": "bubble_wrap",
        "num_missing_catalog_weights": 0,
        "category_avg_weight_error_oz": 0.5,
    }
    resp = client.post("/v1/predict", json=payload)
    assert resp.status_code == 200
    body = resp.json()
    assert "predicted_weight_oz" in body
    assert body["confidence"]["interval_oz"][0] < body["predicted_weight_oz"] < body["confidence"]["interval_oz"][1]


def test_predict_handles_unseen_categories_gracefully(client):
    payload = {
        "theoretical_weight_oz": 50.0,
        "item_count": 2,
        "total_item_volume_in3": 100.0,
        "item_categories": "never_seen_category",
        "category_mode": "never_seen_category",
        "carton_type": "MYSTERY_BOX",
        "ship_method": "DRONE",
        "packing_material": "glitter",
        "num_missing_catalog_weights": 0,
        "category_avg_weight_error_oz": 0.0,
    }
    resp = client.post("/v1/predict", json=payload)
    assert resp.status_code == 200


def test_predict_rejects_invalid_input(client):
    payload = {
        "theoretical_weight_oz": -5.0,
        "item_count": 5,
        "total_item_volume_in3": 800.0,
        "item_categories": "electronics",
        "category_mode": "electronics",
        "carton_type": "L_16x12x10",
        "ship_method": "GROUND",
        "packing_material": "bubble_wrap",
    }
    resp = client.post("/v1/predict", json=payload)
    assert resp.status_code == 422
