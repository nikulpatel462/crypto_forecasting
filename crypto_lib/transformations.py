"""Funtions for processing raw input data"""

import time
from collections import defaultdict

import numpy as np
from pytz import timezone
from sklearn.cluster import KMeans

from .utils import to_dt


def cluster_returns(n_clusters, returns_corr):
    km = KMeans(n_clusters)
    pred = km.fit_predict((1 - returns_corr) ** 0.5)
    ordering = np.argsort(pred)

    clusters = defaultdict(list)
    for col, label in zip(returns_corr.columns, pred):
        clusters[label].append(col)
    return dict(clusters), ordering


def to_factor(df, weights):
    """Find the 'market' value of a feature

    Reminder: this is arbitrary - there could be other factors present in features for which
    this factor isn't appropriate and eg a PCA would be better, however this is second order
    and inconsistent with how the targets are normalised
    """
    # weight_sum = ((df * 0) + weights).sum(axis=1)
    # return (df * weights).sum(axis=1) / weight_sum
    return (df * weights).sum(axis=1) / weights.sum()


def calc_beta(df, factor, window, min_window, mean_zero):
    if mean_zero:
        beta = (
            df.mul(factor, axis=0)
            .rolling(window, min_window)
            .mean()
            .div((factor ** 2).rolling(window, min_window).mean(), axis=0)
        )  # like in the intro notebook
    else:
        beta = (
            df.rolling(window, min_window)
            .cov(factor)
            .div(factor.rolling(window, min_window).var(), axis=0)
        )  # actual regression beta coefficient
    return beta


def neut_beta(df, factor, window, min_window=None, mean_zero=False):
    """We are predicting beta neutralised returns, so features should also be 'beta neutral'
    (we don't care about the systematic component of any feature since this is removed from the
    targets).
    """
    if not min_window:
        min_window = max(1, window // 2)

    beta = (
        calc_beta(df, factor, window, min_window, mean_zero)
        .fillna(0)
        .replace([np.inf, -np.inf], 0)
    )
    neut_df = df.sub(beta.mul(factor, axis=0), axis=1)

    if not mean_zero:  # remove the intercept term
        neut_df = neut_df.sub(neut_df.rolling(window, min_window).mean())

    return neut_df


def neut_mkt(df, weights, **beta_kwargs):
    """Call neut_beta with factor=to_factor(df) and return the beta neutralised df"""
    factor = to_factor(df, weights)
    return neut_beta(df, factor=factor, **beta_kwargs)


def shrink_std(df, std_window, shrink_prop, min_window=None):
    """Calculate an estimate of rolling std deviation. Prioritise
    new information, but reduce errors by shrinking to a more robust
    estimator, specifically shrink ewm std dev to rolling std dev.

    If shrink_prop==0, return the exponentially weighted std estimate
    """
    assert 0 <= shrink_prop <= 1

    if not min_window:
        min_window = max(std_window // 4, 1)

    std = df.ewm(std_window, min_periods=min_window).std()

    if shrink_prop > 0:
        flat_std = df.rolling(std_window, min_window).std()
        std = (1 - shrink_prop) * std + shrink_prop * flat_std

    return std


def z_score(df, mean_window=None, std_window=None, shrink_prop=0):
    """Calculate a rolling z-score - subtract a rolling mean and normalise
    by exponentially weighted std dev

    If shrink_prop>0, shrink to the rolling (non-exp weighted) rolling std deviation
    with window 2*std_window
    """
    if mean_window:  # normalise by mean first if given
        df = df - df.rolling(mean_window, min_periods=max(1, mean_window // 4)).mean()
    if std_window:  # then by std deviation
        df = df / shrink_std(df, std_window, shrink_prop)
    return df.replace([np.inf, -np.inf], np.nan)


def neut_returns(close_prices, weights, windows):
    """Calculate beta neutralised returns as described in the competition page.
    Also return the calculated betas.
    """
    log_prices = np.log(close_prices)
    log_rets = {w: log_prices.diff(w) for w in windows}
    # beta neutralise log returns (just like for the targets)
    # regress over 3750 minutes for all windows just like for the actual targets
    neut_mkt_ = lambda v: neut_mkt(
        v, weights=weights, window=3750, min_window=3000, mean_zero=True
    )
    return {w: neut_mkt_(v).replace([np.inf, -np.inf], 0) for w, v in log_rets.items()}


def shift_feats(feature_dicts, shifts):
    """Shift the given features, appending f"shift_{shift}" to each feature key"""
    assert all([shift > 0 for shift in shifts])  # ensure no leakage
    return {
        f"{k}_shift_{shift}": v.shift(shift)
        for k, v in feature_dicts.items()
        for shift in shifts
    }


def is_weekend(unix_ts: int) -> bool:
    """For a time.struct_time object return 1 if it is a weekend.

    Warnings:
    - does not work for some edge cases:
        - different time zones
        - adjusting for daylight savings time
    """
    lt = time.localtime(unix_ts)
    wday = lt.tm_wday
    if wday < 5:
        return True
    return False


def is_rth(unix_ts: int, region: str) -> bool:
    """Return True if the given timestamp is between 8am to 5pm in the
    given region. This roughly represents regular trading hours (ignoring
    weekends) times when people are actively trading.

    Args:
    - unix_ts: a UNIX timestamp
    - region: a region recognised by pytz, eg US/Eastern
    """
    rt_hours = (8, 17)
    # dt = datetime.utcfromtimestamp(unix_ts)
    # local_ts = timezone(region).localize(dt)
    dt = to_dt(unix_ts).astimezone(timezone(region))
    local_hr = dt.hour
    return rt_hours[0] < local_hr < rt_hours[1]
