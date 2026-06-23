import pandas as pd

from shipment_weight.data_gen import generate_shipments
from shipment_weight.features import ALL_FEATURES, TARGET, add_derived_features, build_preprocessor


def test_add_derived_features_columns():
    df = generate_shipments(n_shipments=50, seed=1)
    out = add_derived_features(df)
    for col in ["weight_per_item_oz", "fill_ratio", "num_categories", "has_unknown_category"]:
        assert col in out.columns


def test_add_derived_features_is_idempotent():
    df = generate_shipments(n_shipments=50, seed=1)
    once = add_derived_features(df)
    twice = add_derived_features(once)
    pd.testing.assert_frame_equal(once, twice)


def test_preprocessor_handles_unseen_categories():
    df = add_derived_features(generate_shipments(n_shipments=300, seed=5))
    X, y = df[ALL_FEATURES], df[TARGET]

    preprocessor = build_preprocessor()
    preprocessor.fit(X, y)

    unseen_row = X.iloc[[0]].copy()
    unseen_row["carton_type"] = "NEVER_SEEN_CARTON"
    unseen_row["category_mode"] = "never_seen_category"

    transformed = preprocessor.transform(unseen_row)
    assert transformed.shape[0] == 1


def test_preprocessor_handles_missing_numeric():
    df = add_derived_features(generate_shipments(n_shipments=300, seed=6))
    X, y = df[ALL_FEATURES], df[TARGET]

    preprocessor = build_preprocessor()
    preprocessor.fit(X, y)

    row_with_nan = X.iloc[[0]].copy()
    row_with_nan["total_item_volume_in3"] = None

    transformed = preprocessor.transform(row_with_nan)
    assert transformed.shape[0] == 1
