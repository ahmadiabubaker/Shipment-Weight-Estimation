"""Regression test for the original serving-time skew bug.

The bug: the API computed fill_ratio against a synthetic 4-carton lookup
table (always NaN for real box names -> silently imputed to the training
median), hardcoded void_volume_in3 to 0.0 (no box volume in the request),
and had no way to compute category_avg_weight_error_oz at all (the
train-fold map was never saved anywhere). All three failures were silent --
no error, no NaN in the response.

This test builds a small model bundle the same way training does (pipeline
+ a fabricated category_error_map, mirroring what scripts/train_real_data.py
now persists) and calls the public library entry point with a REAL,
non-synthetic box_name and realistic item lines. If any of the three
features regress to their old broken values (NaN, 0.0, or an unapplied
default), this test fails.
"""
import joblib
import numpy as np
import pytest

from shipment_weight.data_gen import generate_shipments
from shipment_weight.features import ALL_FEATURES
from shipment_weight.predict import Box, ItemLine, ShipmentWeightPredictor, Shipment, build_feature_frame
from shipment_weight.rounding import round_to_billing_tier
from shipment_weight.train import MODEL_CANDIDATES, make_pipeline, split_data


@pytest.fixture(scope="module")
def bundle_path(tmp_path_factory):
    df = generate_shipments(n_shipments=400, seed=11)
    X_train, X_test, y_train, y_test = split_data(df, seed=11)
    pipe = make_pipeline(MODEL_CANDIDATES["ridge"])
    pipe.fit(X_train, y_train)
    residual_std = float((y_test.values - pipe.predict(X_test)).std())

    bundle = {
        "pipeline": pipe,
        "model_type": "ridge",
        "model_version": "test-v0",
        "residual_std": residual_std,
        "trained_on_rows": len(X_train),
        "feature_list": ALL_FEATURES,
        # Fabricated category->mean-residual map, mirroring what
        # scripts/train_real_data.py now persists into the real bundle.
        "category_error_map": {"electronics": 0.42, "apparel": -0.10},
        "category_error_global_mean": 0.05,
    }
    path = tmp_path_factory.mktemp("models") / "model.joblib"
    joblib.dump(bundle, path)
    return path


def _real_world_shipment() -> Shipment:
    """A shipment built the way a real caller would: a real warehouse box
    name/dimensions (not one of the synthetic S_/M_/L_/XL_ CARTON_CAPACITY
    keys) and raw item lines, not pre-aggregated features."""
    return Shipment(
        items=[
            ItemLine(category="electronics", quantity=2, unit_weight_lbs=0.9, length_in=6, width_in=5, height_in=3),
            ItemLine(category="electronics", quantity=1, unit_weight_lbs=0.7, length_in=5, width_in=4, height_in=2),
            ItemLine(category="apparel", quantity=1, unit_weight_lbs=0.4, length_in=10, width_in=8, height_in=2),
        ],
        box=Box(box_name="14x10x8", length_in=14.0, width_in=10.0, height_in=8.0, tare_weight_lbs=0.5),
        ship_method="GROUND",
        shipment_id="TEST-001",
    )


def test_build_feature_frame_computes_real_fill_ratio_and_void_volume():
    shipment = _real_world_shipment()
    df = build_feature_frame(
        [shipment],
        category_error_map={"electronics": 0.42, "apparel": -0.10},
        category_error_global_mean=0.05,
    )

    # carton_type is the real box name, not a synthetic CARTON_CAPACITY key --
    # confirms the input this test exercises is the one that broke before.
    assert df["carton_type"].iloc[0] == "14x10x8"

    box_volume_in3 = 14.0 * 10.0 * 8.0
    item_volume_in3 = (2 * 6 * 5 * 3) + (1 * 5 * 4 * 2) + (1 * 10 * 8 * 2)

    fill_ratio = df["fill_ratio"].iloc[0]
    void_volume = df["void_volume_in3"].iloc[0]

    assert not np.isnan(fill_ratio), "fill_ratio regressed to NaN (synthetic CARTON_CAPACITY lookup, no real box volume)"
    assert fill_ratio == pytest.approx(min(item_volume_in3 / box_volume_in3, 1.5))

    assert void_volume > 0, "void_volume_in3 regressed to the old hardcoded 0.0"
    assert void_volume == pytest.approx(box_volume_in3 - item_volume_in3)

    # category_mode is "electronics" (2 of 3 lines) -> should pick up the
    # saved map's electronics value, not the global mean and not a caller guess.
    assert df["category_mode"].iloc[0] == "electronics"
    assert df["category_avg_weight_error_oz"].iloc[0] == pytest.approx(0.42)


def test_build_feature_frame_falls_back_to_global_mean_for_unseen_category():
    shipment = Shipment(
        items=[ItemLine(category="never_seen", quantity=1, unit_weight_lbs=1.0, length_in=4, width_in=4, height_in=4)],
        box=Box(box_name="8x8x8", length_in=8.0, width_in=8.0, height_in=8.0, tare_weight_lbs=0.2),
        ship_method="GROUND",
    )
    df = build_feature_frame([shipment], category_error_map={"electronics": 0.42}, category_error_global_mean=0.05)
    assert df["category_avg_weight_error_oz"].iloc[0] == pytest.approx(0.05)


def test_predictor_end_to_end_with_real_box_name(bundle_path):
    predictor = ShipmentWeightPredictor.load(bundle_path)
    result = predictor.predict(_real_world_shipment())

    assert result.shipment_id == "TEST-001"
    assert result.predicted_weight_oz > 0
    assert result.confidence_interval_oz[0] < result.predicted_weight_oz < result.confidence_interval_oz[1]

    # The exact regression check: these must not be the old broken values.
    assert not np.isnan(result.features["fill_ratio"])
    assert result.features["void_volume_in3"] > 0
    assert result.features["category_avg_weight_error_oz"] == pytest.approx(0.42)


def test_predictor_includes_rounded_label_field(bundle_path):
    """predicted_weight_lbs_for_label must be present alongside the existing,
    unchanged predicted_weight_lbs, and must equal what
    shipment_weight.rounding.round_to_billing_tier produces for this
    prediction's own precise value -- proves the field is derived from the
    shared rounding function (not a second, possibly-drifted implementation)
    and follows the verified rule: >=1lb rounds to the nearest whole pound,
    <1lb stays unrounded."""
    predictor = ShipmentWeightPredictor.load(bundle_path)
    result = predictor.predict(_real_world_shipment())

    assert hasattr(result, "predicted_weight_lbs_for_label")
    expected_label_lbs, expected_sub_1lb = round_to_billing_tier(result.predicted_weight_lbs)
    assert result.predicted_weight_lbs_for_label == pytest.approx(float(expected_label_lbs))

    if expected_sub_1lb:
        # Precise prediction was < 1lb: label field must stay unrounded,
        # i.e. identical to the precise value (e.g. 0.7 lbs stays 0.7).
        assert result.predicted_weight_lbs_for_label == pytest.approx(result.predicted_weight_lbs)
    else:
        # Precise prediction was >= 1lb: label field must be a whole number
        # (e.g. a precise 12.34 lbs prediction would become 12.0).
        assert result.predicted_weight_lbs_for_label == round(result.predicted_weight_lbs_for_label)

    # Confidence interval must stay based on the PRECISE prediction, not the
    # rounded label value -- rounding is a presentation step, not a
    # statistical one (see WeightPrediction's confidence_interval_oz comment).
    precise_oz = result.predicted_weight_lbs * 16.0
    assert result.confidence_interval_oz[0] < precise_oz < result.confidence_interval_oz[1]


def test_rounding_examples_from_spec():
    """The exact worked examples from the rounding spec: a precise 12.34 lbs
    prediction rounds to 12 for the label; a precise 0.7 lbs prediction
    stays unrounded. Exercised directly against the shared rounding
    function (predict.py delegates to this same function -- see
    test_predictor_includes_rounded_label_field for the delegation check)."""
    rounded, _ = round_to_billing_tier(12.34)
    assert float(rounded) == 12.0

    rounded, _ = round_to_billing_tier(0.7)
    assert float(rounded) == 0.7


def test_predict_batch_matches_single_predict(bundle_path):
    predictor = ShipmentWeightPredictor.load(bundle_path)
    shipments = [_real_world_shipment(), _real_world_shipment()]
    batch_results = predictor.predict_batch(shipments)
    single_result = predictor.predict(shipments[0])

    assert len(batch_results) == 2
    assert batch_results[0].predicted_weight_oz == pytest.approx(single_result.predicted_weight_oz)


def test_load_raises_clear_error_for_missing_model(tmp_path):
    missing = tmp_path / "does_not_exist.joblib"
    with pytest.raises(FileNotFoundError):
        ShipmentWeightPredictor.load(missing)
