"""Functions for helping explore data"""

import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from statsmodels.tsa.stattools import acf


def calc_expl_vars(df, n_components=10):
    """Calculate the top n_components explained variances from a PCA"""
    return (
        PCA(n_components=n_components)
        .fit(StandardScaler().fit_transform(df))
        .explained_variance_ratio_
    )


def calc_acf(df, **acf_kwargs):
    """Calculate the acf for each column in df.

    Warnings:
    - replaces np.nan with 0, so will fail if all values are missing
    - if a pacf calculation fails we ignore that column
    """
    acfs = {}
    for col in df:
        try:
            acfs[col] = acf(
                df[col].values, missing="drop", **acf_kwargs
            )  # will fail if all 0 (no data for some coins)
        except:
            continue
    return acfs


def plot_moments(df, window, figsize=(16, 4)):
    """Plot rolling 1st, ..., 4th moments of the given df"""
    _, ax = plt.subplots(1, 4, figsize=figsize)
    to_plot = ["std", "skew", "kurt"]
    df.plot(ax=ax[0], legend=False)
    window = 1000
    for i in range(3):
        getattr(df.rolling(window, window // 2), to_plot[i])().plot(
            ax=ax[i + 1], legend=False
        )
