"""
Splitting timestamps into train + test folds for cross validaiton.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold, TimeSeriesSplit


def timestamps_to_indices(timestamps, ts_index: pd.Index, level=0):
    """Returns indices of ts_index where the corresponding timestamp
    is in ts_index.

    Args:
    - timestamps: an iterable of timestamps
    - ts_index: a pd.Index/MultiIndex
    - level: which level of ts_index to search for if this is a multiindex object
    """
    if isinstance(ts_index, pd.MultiIndex):
        arr = ts_index.isin(timestamps, level=level)
    else:
        arr = ts_index.isin(timestamps)
    return np.where(arr)[0]


def timestamp_folds_to_indices(folds, ts_index: pd.Index, level=0):
    """Convert folds of timestamps to folds of indices of ts_index"""
    if isinstance(ts_index, pd.MultiIndex):
        ts_index = ts_index.get_level_values(level)
    ind_func = lambda x: timestamps_to_indices(x, ts_index, level)
    index_folds = [(ind_func(a), ind_func(b)) for a, b in folds]
    return tuple(index_folds)


def _index_timestamps(splits, timestamps):
    """Convert a tuple of pairs of train + test timestamp indices to the
    corresponding splits of actual timestamps.
    """
    return tuple((timestamps[a], timestamps[b]) for a, b in splits)


def ts_split(timestamps, n_splits, split_ratio, overlap=16):
    """Split timestamps into n non-overlapping test and train folds where (roughly)
    len(test)/len(train) = split_ratio using sklearn's TimeSeriesSplit.

    Notes:
    - There are two special cases:
      - n_splits=1: split directly into 2 sets.
      - split_ratio=0: the train window expands instead of having a max size.
    """
    timestamps = np.sort(list(set(timestamps)))
    n_timestamps = len(timestamps)

    if n_splits == 1:
        split_point = int((1 - split_ratio) * n_timestamps)
        return ((timestamps[: split_point - overlap], timestamps[split_point:]),)

    test_size = n_timestamps / (n_splits + 1)
    if split_ratio:
        max_train_size = int(test_size / split_ratio)
    else:
        max_train_size = None
    ts_split = TimeSeriesSplit(
        n_splits=n_splits, max_train_size=max_train_size, gap=overlap
    )
    split_indexes = tuple(ts_split.split(timestamps))
    return _index_timestamps(split_indexes, timestamps)


def cv_split(timestamps, n_splits, overlap, embargo):
    """KFold cv split as in de Prado's `Advances in Financial Machine Learning`"""
    splits = KFold(n_splits, shuffle=False).split(timestamps)
    clean_splits = []
    for train, test in splits:
        min_t = test[0]
        max_t = test[-1]
        train_left = [k for k in train if k < min_t - overlap]
        train_right = [k for k in train if k > max_t + overlap + embargo]
        clean_train = train_left + train_right
        clean_splits.append((clean_train, test))
    return _index_timestamps(clean_splits, timestamps)


def grid_res_to_df(grid_search):
    """Convert results of GridSearchCV or RandomizedSearchCV to a
    df where the index values are the tested hyperparameters.
    """
    df = pd.DataFrame(grid_search.cv_results_)
    ind = pd.DataFrame(list(df["params"].values))
    df.index = pd.MultiIndex.from_frame(ind)
    return df


def plot_cv_scores(grid_res_df, figsize=(10, 6)):
    """Plot the scores for each fold and the mean score for each hyperparameter
    from the output of grid_res_to_df.

    Returns: a df of scores for each fold + hyperparameter
    """
    _, ax = plt.subplots(1, 2)
    scores = grid_res_df[
        [k for k in grid_res_df if "_test_score" in k and "split" in k]
    ].T
    scores.plot(figsize=figsize, ax=ax[0], legend=False)
    scores.mean().plot(figsize=figsize, ax=ax[1])
    return scores
