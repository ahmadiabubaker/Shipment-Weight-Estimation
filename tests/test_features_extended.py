import numpy as np
import pandas as pd
import pytest

from shipment_weight.features_extended import (
    add_extended_features,
    add_target_encodings,
    expanding_target_encode,
    finalize_train_fold_stats,
    top_category_by_weight,
    train_fold_only_encode,
)


def _toy_df():
    return pd.DataFrame(
        {
            "order_date": pd.to_datetime(
                ["2026-01-01", "2026-01-01", "2026-01-02", "2026-01-02", "2026-01-03", "2026-01-03"]
            ),
            "box_name": ["A", "A", "A", "B", "A", "B"],
            "residual": [1.0, 3.0, 5.0, 10.0, 100.0, -4.0],
        }
    )


def test_expanding_target_encode_matches_manual_strictly_prior_mean():
    df = _toy_df()
    encoded = expanding_target_encode(df, "box_name", "residual", "order_date")

    for i in range(len(df)):
        prior = df[df["order_date"] < df.loc[i, "order_date"]]
        prior_group = prior[prior["box_name"] == df.loc[i, "box_name"]]
        if len(prior_group) > 0:
            expected = prior_group["residual"].mean()
        elif len(prior) > 0:
            expected = prior["residual"].mean()
        else:
            expected = 0.0
        assert encoded[i] == pytest.approx(expected), f"row {i}: leak or wrong cold-start fallback"


def test_expanding_target_encode_same_day_rows_never_see_each_other():
    # Both rows on 2026-01-01 have no prior data at all -> both must be the
    # 0.0 cold-start fallback, never each other's same-day residual.
    df = _toy_df()
    encoded = expanding_target_encode(df, "box_name", "residual", "order_date")
    assert encoded[0] == 0.0
    assert encoded[1] == 0.0


def test_expanding_target_encode_future_mutation_does_not_change_past_rows():
    """The core no-leakage guarantee: changing a later date's residuals must
    not change the encoding of any earlier-dated row."""
    df = _toy_df()
    encoded_before = expanding_target_encode(df, "box_name", "residual", "order_date")

    df_mutated = df.copy()
    last_date = df_mutated["order_date"].max()
    df_mutated.loc[df_mutated["order_date"] == last_date, "residual"] = 999_999.0
    encoded_after = expanding_target_encode(df_mutated, "box_name", "residual", "order_date")

    earlier_mask = (df["order_date"] < last_date).values
    np.testing.assert_array_equal(encoded_before[earlier_mask], encoded_after[earlier_mask])


def test_expanding_target_encode_no_row_uses_same_day_or_later_data():
    """Direct assertion of the required property: for every row, the
    encoding must be reproducible using ONLY rows strictly before that
    row's date -- i.e. dropping all same-day-or-later rows from the input
    before encoding a given row's group must not change that row's value."""
    df = _toy_df()
    encoded = expanding_target_encode(df, "box_name", "residual", "order_date")

    for i in range(len(df)):
        strictly_prior = df[df["order_date"] < df.loc[i, "order_date"]]
        # Re-encode using a frame that physically cannot contain same-day-or-later data
        probe = pd.concat([strictly_prior, df.loc[[i]]], ignore_index=True)
        probe_encoded = expanding_target_encode(probe, "box_name", "residual", "order_date")
        assert encoded[i] == pytest.approx(probe_encoded.iloc[-1])


def test_expanding_target_encode_cold_start_group_uses_global_fallback():
    df = _toy_df()
    encoded = expanding_target_encode(df, "box_name", "residual", "order_date")
    # Row 3 (index 3) is box "B"'s first-ever appearance, on 2026-01-02.
    # No prior "B" rows exist, so it must fall back to the global mean of
    # all strictly-prior rows (both "A" residuals from 2026-01-01: 1.0, 3.0).
    assert encoded[3] == pytest.approx(2.0)


def test_add_target_encodings_columns_present():
    df = pd.DataFrame(
        {
            "order_date": pd.to_datetime(["2026-01-01", "2026-01-02", "2026-01-03"]),
            "carton_type": ["box1", "box1", "box2"],
            "ship_method": ["ground", "ground", "air"],
            "actual_weight_oz": [100.0, 110.0, 200.0],
            "theoretical_weight_oz": [90.0, 90.0, 180.0],
        }
    )
    out = add_target_encodings(df)
    assert "box_name_target_enc_oz" in out.columns
    assert "ship_method_target_enc_oz" in out.columns
    assert out["box_name_target_enc_oz"].isnull().sum() == 0


def test_top_category_by_weight_picks_heaviest_not_most_frequent():
    lines = pd.DataFrame(
        {
            "shipment_number": ["S1"] * 4,
            "item_id": ["I1", "I2", "I3", "I4"],
            "item_quantity": [5, 1, 1, 1],
            "category": ["light_accessory", "light_accessory", "light_accessory", "heavy_item"],
            "theoretical_item_weight_lbs": [0.1, 0.1, 0.1, 20.0],
        }
    )
    out = top_category_by_weight(lines)
    row = out[out["shipment_number"] == "S1"].iloc[0]
    # light_accessory: 5*0.1 + 0.1 + 0.1 = 0.7 lbs total; heavy_item: 20.0 lbs.
    # Mode-by-count would pick light_accessory (3 lines); weight-mode must pick heavy_item.
    assert row["top_category_by_weight"] == "heavy_item"


def test_add_extended_features_handles_zero_volume_shipment():
    df = pd.DataFrame(
        {
            "shipment_number": ["S1", "S2"],
            "total_item_volume_in3": [0.0, 100.0],
            "box_volume_in3": [50.0, 200.0],
            "theoretical_cargo_weight_lbs": [5.0, 10.0],
            "num_categories": [1, 2],
        }
    )
    lines = pd.DataFrame(
        {
            "shipment_number": ["S1", "S2"],
            "item_id": ["I1", "I2"],
            "item_quantity": [1, 1],
            "category": ["cat_a", "cat_b"],
            "theoretical_item_weight_lbs": [5.0, 10.0],
        }
    )
    out = add_extended_features(df, lines)
    # zero-volume shipment's density is left NaN pre-finalize (no crash/inf)
    assert np.isnan(out.loc[out["shipment_number"] == "S1", "weight_density_lbs_per_in3"]).all()
    assert not np.isinf(out["weight_density_lbs_per_in3"]).any()


def test_finalize_train_fold_stats_uses_train_only_for_fallback_and_winsorize():
    train = pd.DataFrame(
        {
            "weight_density_lbs_per_in3": [1.0, 2.0, np.nan],
            "fill_ratio_raw": [0.5, 0.6, 0.7],
        }
    )
    test = pd.DataFrame(
        {
            "weight_density_lbs_per_in3": [np.nan],
            "fill_ratio_raw": [50.0],  # extreme test-only outlier
        }
    )
    train_out, test_out = finalize_train_fold_stats(train, test)

    train_median = train["weight_density_lbs_per_in3"].median()
    assert train_out["weight_density_lbs_per_in3"].iloc[2] == pytest.approx(train_median)
    assert test_out["weight_density_lbs_per_in3"].iloc[0] == pytest.approx(train_median)

    # winsorize threshold computed from TRAIN only -- the test-only 50.0
    # outlier must not have influenced the clip threshold.
    train_p99 = train["fill_ratio_raw"].quantile(0.99)
    assert test_out["fill_ratio_winsorized"].iloc[0] == pytest.approx(train_p99)


def test_train_fold_only_encode_ignores_test_data_and_uses_global_fallback():
    train = pd.DataFrame({"box_name": ["A", "A", "B"], "resid": [1.0, 3.0, 10.0]})
    test = pd.DataFrame({"box_name": ["A", "B", "C"], "resid": [999.0, -999.0, -999.0]})

    train_enc, test_enc = train_fold_only_encode(train, test, "box_name", "resid")

    # train encoding is the train-only group mean, unaffected by test values
    assert train_enc.tolist() == pytest.approx([2.0, 2.0, 10.0])
    # test encoding uses the SAME frozen train map -- test's own resid values
    # (999.0, -999.0) must not leak into the encoding
    assert test_enc.iloc[0] == pytest.approx(2.0)  # box "A" -> train mean
    assert test_enc.iloc[1] == pytest.approx(10.0)  # box "B" -> train mean
    # box "C" never seen in train -> global train mean fallback
    global_mean = train["resid"].mean()
    assert test_enc.iloc[2] == pytest.approx(global_mean)
