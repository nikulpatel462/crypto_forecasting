""" (Vastly) simplified submission functions to go in the submission notebook.

Optimised for speed (at the expense of feature quality) to stay in the submission time limit.

Contains no dependencies on the library functions to avoid having to clone + install a private git repo.
"""

import pandas as pd
import numpy as np


def bar_feats_minimal(df):
    """Augment the given dataframe in place with some features for each bar."""
    midpoint = (df["Open"] + df["Close"]) / 2
    feats = {
        "rel_avg": df["VWAP"] / midpoint,
        "avg_t_size": (df["Volume"] / df["Count"])
        ** (1 / 10),  # average # of units per transaction
        "dollar_vol": np.log(df["Volume"] * df["VWAP"]),  # dollar volume traded
        "rel_dev": ((df["High"] - df["Low"]) / midpoint) ** (1 / 3),
        "shadow_diff": (df["High"] + df["Low"]) / (2 * midpoint) - 1,
    }
    for name, feat in feats.items():
        df.loc[:, name] = feat


def ts_feats_minimal(df, window, price_mom_windows, include_target=True):
    """Add rolling z-score features including price momentum features and the target + target scale.
    Assumes index is timestamps + Asset_IDs

    Warning: changes input df in-place to save memory
    """
    to_z_score = [
        "rel_avg",
        "avg_t_size",
        "shadow_diff",
        "dollar_vol",
        "rel_dev",
    ]

    log_close_grp = df[["Close"]].groupby(level="Asset_ID", as_index=False)

    for mom_window in price_mom_windows:
        feat_name = f"price_mom_{mom_window}"
        df.loc[:, feat_name] = log_close_grp.diff(mom_window)["Close"]
        to_z_score.append(feat_name)

    min_periods = max(1, window // 10)
    df_grp = (
        df[to_z_score]
        .groupby(level="Asset_ID", as_index=False)
        .rolling(window, min_periods=min_periods)
    )

    roll_mean = (
        df_grp.mean().drop(columns="Asset_ID").fillna(0)
    )  # FIXME: shouldn't include asset_id column
    roll_std = df_grp.std().drop(columns="Asset_ID").ffill().fillna(1)

    norm_feats = ((df[to_z_score] - roll_mean) / roll_std).rename(
        mapper=lambda x: "roll_" + x, axis="columns"
    )

    norm_feats.loc[:, "target_scale"] = roll_std["price_mom_15"]

    if include_target:  # FIXME: potentially confusing target naming convention
        norm_feats.loc[:, "scaled_target"] = df["Target"] / norm_feats["target_scale"]
        norm_feats.loc[:, "target"] = df["Target"]

    return norm_feats


def all_feats_minimal(df, include_target=True):
    """Minimal version of all_feats"""
    price_mom_windows = (1, 5, 15, 80)
    window = 15

    bar_feats_minimal(df)  # augment in-place with bar features
    df.drop(
        columns=["Count", "Open", "High", "Low", "Volume", "VWAP"], inplace=True
    )  # drop unused columns

    df.set_index(["timestamp", "Asset_ID"], inplace=True)

    return ts_feats_minimal(df, window, price_mom_windows, include_target)


def last_n_ts_df(df, lookback, buffer=100):
    """Returns the last rows of df where the timestamp is in the last n of all
    timestamps. This is to concatenate with new data provided by the API so that
    rolling calculations can be performed.

    Warning: assumes df is ordered by timestamps, and could return more data than
    requested.
    """
    n_assets = 14
    return df.iloc[-(n_assets * lookback + buffer) :]


def concat_old_new(old_data, new_data):
    """Concatenate old and new dfs for feature construction. Ensures
    any overlapping timestamps + assetids in the old df are discarded.
    """
    return pd.concat([old_data, new_data.drop(columns="row_id")], ignore_index=True)


def subset_test_index(data, orig_data):
    """Subset the prepred data df on the original test timestamps + assetids"""
    orig_index = pd.MultiIndex.from_frame(orig_data[["timestamp", "Asset_ID"]])
    return data.loc[orig_index]


def join_rowids(preds, orig_test):
    """Join our predictions df with the rowids in the supplied test data df"""
    orig_join_on = orig_test[["timestamp", "Asset_ID", "row_id"]].set_index(
        ["timestamp", "Asset_ID"]
    )
    return preds.join(orig_join_on).reset_index(drop=True)


def predict_loop(model, weights, prev_data, new_data, sample_pred_df, n_to_keep):
    """Function for looping over in env.iter_test():
    - Concatenate previous + new data
    - Cache last n rows of this df
    - Calculate new features
    - Drop rows to match the original training timestamps + asset ids
    - Calculate predictions on this subset
    - Join with the given row ids in the sample predictions df

    Returns: last n rows from prev + new data, predictions df
    """
    concat_data = concat_old_new(prev_data, new_data)
    last_n = last_n_ts_df(concat_data, n_to_keep)
    feats = all_feats_minimal(
        concat_data, weights, fillna_val=0, include_target=False
    ).stack()
    feats = subset_test_index(feats, new_data)
    preds = model.predict(feats).rename("Target").to_frame()
    return last_n, join_rowids(preds, new_data)
