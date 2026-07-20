import numpy as np

from shipment_weight.data_gen import CARTON_TYPES, generate_shipments


def test_generate_shipments_shape_and_dtypes():
    df = generate_shipments(n_shipments=200, seed=1)
    assert len(df) == 200
    expected_cols = {
        "theoretical_weight_oz",
        "actual_weight_oz",
        "carton_type",
        "item_count",
        "total_item_volume_in3",
        "item_categories",
        "category_mode",
        "ship_method",
        "packing_material",
        "num_missing_catalog_weights",
        "category_avg_weight_error_oz",
    }
    assert expected_cols.issubset(df.columns)


def test_no_nulls_in_core_columns():
    df = generate_shipments(n_shipments=500, seed=2)
    assert df["theoretical_weight_oz"].isna().sum() == 0
    assert df["actual_weight_oz"].isna().sum() == 0


def test_weights_are_positive_and_above_tare():
    df = generate_shipments(n_shipments=500, seed=3)
    tares = {name: tare for name, _, tare in CARTON_TYPES}
    min_tare = df["carton_type"].map(tares)
    assert (df["theoretical_weight_oz"] > 0).all()
    assert (df["actual_weight_oz"] >= min_tare).all()


def test_error_compounds_with_item_count():
    df = generate_shipments(n_shipments=4000, seed=4)
    abs_err = (df["actual_weight_oz"] - df["theoretical_weight_oz"]).abs()
    low_count_err = abs_err[df["item_count"] <= 3].mean()
    high_count_err = abs_err[df["item_count"] >= 10].mean()
    assert high_count_err > low_count_err


def test_deterministic_with_seed():
    df1 = generate_shipments(n_shipments=50, seed=99)
    df2 = generate_shipments(n_shipments=50, seed=99)
    assert np.allclose(df1["actual_weight_oz"], df2["actual_weight_oz"])
