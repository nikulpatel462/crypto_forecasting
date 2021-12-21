"""All feature calculations"""

import pandas as pd
import numpy as np

from .utils import (
    pivot_data,
    combine_dicts,
    replace_missing_data,
    # dicts_to_long_df,
    # iterative_stack,
)
from .transformations import (
    neut_mkt,
    neut_returns,
    z_score,
    # shift_feats,
    shrink_std,
    is_rth,
    is_weekend,
)


def timestamp_features_(ts: int) -> dict:
    """See timestamp_features"""
    regions = ["US/Eastern", "Europe/London", "Japan"]
    is_rth_ = {r.split("/")[0]: is_rth(ts, r) for r in regions}
    return {"is_weekend": is_weekend(ts), **is_rth_}


def timestamp_features(unix_timestamps: pd.DatetimeIndex) -> pd.DataFrame:
    """A few features based on unix timestamps:
    - is this a weekend?
    - is this when people will be actively trading?
        - ... for US + London + Japan for global coverage

    FIXME: this is unexpectedly slow
    """
    features = [timestamp_features_(ts) for ts in unix_timestamps]
    return pd.DataFrame(features, index=unix_timestamps).reset_index()


def bar_feats(df):
    """Augment the given dataframe in place with some features for each bar."""
    midpoint = (df["Open"] + df["Close"]) / 2
    feats = {
        "rel_avg": df["VWAP"] / midpoint,
        "avg_t_size": df["Volume"] / df["Count"],  # average # of units per transaction
        "dollar_vol": df["Volume"] * df["VWAP"],  # dollar volume traded
        "rel_dev": (df["High"] - df["Low"]) / midpoint,
        "shadow_diff": (df["High"] + df["Low"]) / (2 * midpoint) - 1,
    }
    for name, feat in feats.items():
        df.loc[:, name] = feat


def price_mom_feats(resid_rets):
    """Price momentum features using residual returns as inputs."""
    return {
        f"price_mom_{w}": z_score(v, None, 800) for w, v in resid_rets.items()
    }  # price momentum z-scores


def rolling_feats(pivotted_df, weights, window, shrink_prop):
    """Remaining time series z-score features from the pivotted data"""
    std_window = 20 * window

    minute_returns = np.log(pivotted_df["Close"]).diff()
    observed_vol = minute_returns.ewm(
        span=window, min_periods=max(1, window // 2)
    ).std()

    feats = {
        "rel_avg": np.log(pivotted_df["rel_avg"]),
        "avg_t_size": pivotted_df["avg_t_size"] ** (1 / 10),
        "shadow_diff": pivotted_df["shadow_diff"],
        "dollar_vol": np.log(pivotted_df["dollar_vol"]),
        "obs_vol": observed_vol ** (1 / 5),
        "hi_lo": pivotted_df["rel_dev"] ** (1 / 3),  # vol based on high-low
    }  # apply transformations to reduce skewness

    # let's demean before removing the 'market' beta
    demeaned_feats = {k: z_score(v, window) for k, v in feats.items()}

    beta_neut_feats = {
        k: neut_mkt(v, weights, window=3750, min_window=300)
        for k, v in demeaned_feats.items()
    }  # result should still have mean 0, so only normalise by std dev

    return {
        f"{k}_roll_{window}": z_score(v, std_window=std_window, shrink_prop=shrink_prop)
        for k, v in beta_neut_feats.items()
    }


def target_norm_scale(resid_rets):
    """Return the scaling factor for the target to normalise for volatility.
    This is to make the targets less non-stationary to improve model fitting.

    Args:
    - resid_rets: 15 minute residualised returns (calculated with tradeable data)
    """
    return shrink_std(resid_rets, 3000, 0.2, min_window=300)


def target_preparation(pivotted_df, neut_returns_15, include_target):
    """Prepare target information:
    Args:
    - neut_returns_15: 15 minute neutralised returns for the target scale calculation
    - include_target: if True also include the target from the pivotted df and the
    scaled targets
    """
    default_std = 0.01  # generic std dev of returns

    target_info = {"target_scale": target_norm_scale(neut_returns_15).ffill()}

    replace_missing_data(target_info["target_scale"], default_std)

    if include_target:
        target_info = {
            **target_info,
            "target": pivotted_df["Target"],
            "scaled_target": pivotted_df["Target"] / target_info["target_scale"],
        }
    return target_info


def all_feats(df, weights, fillna_val=0, include_target=True):
    """Take a df of training data and return a dict of all features defined in this
    module for a given set of parameters.

    A few hardcoded parameters below:
        - returns_windows: windows to pass to neut_returns
        - window: characteristic window to pass to rolling_feats
        - shrink_prop: shrink ratio for std deviation calculations

    Args:
    - fillna_val: value to fill in np.nan values in final features dfs
    - include_target: whether to include columns (target, scaled_target). Set
    to True for training data and to False when predicting using the API.

    Warnings:
    - Will delete the input df after finished to reduce memory usage
    """
    returns_windows = (1, 5, 15, 80)
    window = 15
    shrink_prop = 0.2

    bar_feats(df)  # augment in-place with bar features
    pivotted_df = pivot_data(df)

    df.drop(df.index, inplace=True)  # reduce memory usage by dropping all rows

    neut_returns_ = neut_returns(pivotted_df["Close"], weights, returns_windows)
    neut_returns_15 = neut_returns_[15]
    price_mom_feats_ = price_mom_feats(neut_returns_)

    del neut_returns_  # save memory

    all_feats = {
        **price_mom_feats_,
        **rolling_feats(pivotted_df, weights, window, shrink_prop),
    }

    target_info = target_preparation(pivotted_df, neut_returns_15, include_target)

    # general_feats = timestamp_features(pivotted_df.index)

    for v in all_feats.values():
        replace_missing_data(v, fillna_val)

    return pd.concat({**all_feats, **target_info}, axis=1)
