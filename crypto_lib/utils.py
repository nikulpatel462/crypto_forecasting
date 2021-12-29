"""General utility functions for manipulating data"""

import time
import os
import logging
import pickle
from datetime import datetime
from functools import reduce

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def linear_weighting(timestamps, end_weight=0.1):
    """Linearly decaying weights starting at 1 (at the latest timestamp)
    and decaying to end_weight at the smallest timestamp.
    """
    t_diff = timestamps[-1] - timestamps[0]
    b = (1 - end_weight) / t_diff
    return end_weight + b * (timestamps - timestamps[0])


def pivot_data(df):
    """Pivot a long df to a wide format with asset ids + original columns as
    columns and setting the timestamp as the new index. Also sort on the new
    timestamp index in case this was originally unordered.
    """
    return df.pivot(index="timestamp", columns="Asset_ID").sort_index()


def reindex_missing(pivotted_df):
    """Reindex a dataframe indexed on UTC timestamps padding for missing index values"""
    idx = pivotted_df.index
    return pivotted_df.reindex(range(idx[0], idx[-1] + 60, 60), method="pad")


def totimestamp(s):
    """Convert a datetime string to a UTC timestamp. Copied from the
    intro notebook."""
    return np.int32(time.mktime(datetime.strptime(s, "%d/%m/%Y").timetuple()))


def to_dt(x):
    """Convert a UTC timestamp to a datetime object"""
    return datetime.utcfromtimestamp(x)


def chunk_list(l, chunk_size):
    """Iterate over a list in chunks"""
    for x in range(0, len(l), chunk_size):
        yield l[x : x + chunk_size]


def combine_dicts(dicts):
    """Combine an iterable of dicts into a single dict

    Warnings:
    - Possible undesired behaviour if dictionary keys overlap
    """
    return reduce(lambda x, y: {**x, **y}, dicts)


def replace_missing_data(df, value):
    """Replaces [np.inf, -np.inf, np.nan] with the given value (which can also
    be np.nan) on the givn df in-place"""
    df.replace([np.inf, -np.inf, np.nan], value, inplace=True)


class ResultCacher:
    """Helper class to pickle and load a series of results to a given directory.

    Note: saves as pickles rather than eg parquet for low RAM requirements when reading files
    and for low file sizes (compared to csv).
    """

    def __init__(self, save_path):
        self.save_path = save_path  # directory to cache intermediate results
        self.result_paths = []  # to keep track of cached files

        self.create_save_folder()

    def create_save_folder(self):
        os.makedirs(self.save_path)

    def get_save_path(self, filename):
        return os.path.join(self.save_path, filename)

    def cache_result(self, result, filename):
        """Pickles result to given filename in save_path"""
        file_save_path = self.get_save_path(filename)
        logger.info(f"Saving result to {file_save_path}")
        with open(file_save_path, "wb") as f:
            pickle.dump(result, f)

        self.result_paths.append(file_save_path)

    def load_all_results(self):
        """Loads all cached results and returns as a list"""
        results = []
        for path in self.result_paths:
            with open(path, "rb") as f:
                results.append(pickle.load(f))
        return results  # FIXME: does not clean up cached files


# def dicts_to_long_df(feat_dict):
#     """Take a dict of features, concatenate them and stack to a long formatted
#     dataframe (just like the original training data)
#     """
#     return pd.concat(feat_dict, axis=1).stack().reset_index()


# def iterative_stack(feat_dict, chunk_size):
#     """Call dicts_to_long_df iteratively on chunks of the input feature dictionary
#     to avoid memory issues.
#     """
#     sub_dict = lambda x: {k: v for k, v in feat_dict.items() if k in x}
#     comb_subsets = [
#         sub_dict(chunk) for chunk in chunk_list(list(feat_dict), chunk_size)
#     ]

#     running_stack = dicts_to_long_df(comb_subsets[0])
#     comb_subsets = comb_subsets[1:]

#     while comb_subsets:
#         running_stack = running_stack.merge(
#             dicts_to_long_df(comb_subsets[0]), on=["timestamp", "Asset_ID"]
#         )
#         comb_subsets.pop(0)

#     return running_stack
