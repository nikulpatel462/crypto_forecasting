"""Helper functions for fitting + evaluating models"""
from functools import partial, reduce

import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, clone, RegressorMixin
from sklearn.multioutput import MultiOutputRegressor
from sklearn.ensemble import VotingRegressor
from tensorflow import keras
from tensorflow.keras import layers


from .metrics import weighted_correlation

# FIXME: these columns + others are hardcoded throughout. Should at least be replaced by these
SCALED_TARGET = "scaled_target"
ORIGINAL_TARGET = "target"


def feature_names(data_cols):
    """Take single/multiindex columns from a pandas df of features + targets
    and return only the feature names.
    """
    if isinstance(data_cols, pd.MultiIndex):
        data_cols = data_cols.get_level_values(0).unique()

    return [k for k in data_cols if "target" not in k]


def get_xy_arrays(data_df):
    """Returns a tuple of numpy arrays: features, scaled targets"""
    features = feature_names(data_df.columns)
    try:
        target = data_df["scaled_target"].values
    except:
        target = None
    return data_df[features].values, target


def merge_dfs(df1, df2, join_on):
    """Wrapper around pd.merge to join two dataframes on a given list of columns"""
    return df1.merge(df2, left_on=join_on, right_on=join_on)


def subset_on_index(df, timestamps):
    """Take a long formatted df with an index of timestamps and subset on rows where the
    timestamp is in the given timestamps.
    """
    return df.loc[timestamps, :]


def scale_predictions(model, X, y_scale):
    """Undo the vol normalisation for the targets to get predictions
    for actual returns.
    """
    return model.predict(X) * y_scale


def fit_cluster_panel(model, data, cluster, n_jobs=1):
    """Fit a panel model for a given cluster"""
    data_ = data.loc[:, [slice(None), slice(cluster)]]
    X = data_.drop(columns=["target", "target_scale"])
    y = data_["scaled_target"]
    panel_model = MultiOutputRegressor(model, n_jobs=n_jobs)
    return panel_model.fit(X, y)


def asset_id_from_df(data):
    """Determine the asset_ids either in data.columns or data.index"""
    if isinstance(data.columns, pd.MultiIndex):
        return data.target.columns.values  # same order as targets
    return data.index.get_level_values(1).unique().values


def calc_scaled_preds(model, data):
    """Calls model.predict on the features in data and undo the vol
    scaling to obtain predictions for the original returns.

    Note: If the model predicts a single target, then reshape array
    to (num_preds, 1) for consistency with the multioutput models.
    """
    X, _ = get_xy_arrays(data)
    preds = model.predict(X)
    scale = data.target_scale.values

    if len(preds.shape) == 1:  # only one asset
        preds = preds.reshape(preds.shape[0], 1)
        scale = scale.reshape(preds.shape[0], 1)

    # print(.shape, preds.shape)
    return scale * preds


def evaluate_pool(model, data, metrics_func, asset_labels):
    """Evaluate a fitted pool model. Returns output of
    metrics_func as a dict on the asset_ids in data.index.

    Args:
    - metrics_func: a function expecting single asset predictions
    y_true, y_pred (signature similar to sklearn.metrics functions)
    """
    y_pred = calc_scaled_preds(model, data).reshape(-1)
    y_eval = data.target.values.reshape(-1)

    id_index = data.index.get_level_values(1).values

    evals = {}
    for asset_id in asset_labels:
        id_ind = id_index == asset_id  # predictions only for that asset
        evals[asset_id] = metrics_func(y_eval[id_ind], y_pred[id_ind])
    return evals


def evaluate_panel(model, data, metrics_func, asset_labels):
    """Evaluate a fitted panel model (eg MultiOutputRegressor). Returns output of
    metrics_func as a dict on the asset_ids in data.target.

    Args:
    - metrics_func: a function expecting single asset predictions
    y_true, y_pred (signature similar to sklearn.metrics functions)
    """
    y_pred = calc_scaled_preds(model, data)
    y_eval = data.target.values

    evals = {}
    for i, asset_id in enumerate(asset_labels):
        asset_pred = y_pred[:, i].reshape(-1)
        asset_eval = y_eval[:, i].reshape(-1)
        evals[asset_id] = metrics_func(asset_eval, asset_pred)
    return evals


def fit_model(model, train_ind, test_ind, data, eval_func, **fit_kwargs):
    """Fit a model and pass the predictions to
    metrics_func for evaluation, returning the model and the
    output of metrics_func.

    Args:
    - train_ind: either a list/pd.Series/pd.Index of timestamps to
    select a subset of data for training.
    - test_ind: same as train_ind but for evaluation.
    - eval_func: one of eval_panel, eval_pool. Expects metrics_func
    and asset_labels args to be supplied (eg using functools.partial)
    """
    model.fit(*get_xy_arrays(data.loc[train_ind]), **fit_kwargs)
    evals = eval_func(model, data.loc[test_ind])
    return model, evals


def fold_fit_model(model, data, folds, eval_func, metrics_func):
    """Call fit_model for each (train, test) pair in folds

    Returns: a list of (model, eval) for each fold
    """
    asset_ids = asset_id_from_df(data)
    eval_func_ = partial(eval_func, metrics_func=metrics_func, asset_labels=asset_ids)

    all_results = []
    for train_ind, test_ind in folds:
        all_results.append(
            fit_model(clone(model), train_ind, test_ind, data, eval_func_)
        )
    return all_results


def create_nn_model(model_layers):
    """Create a Keras model with variable layers + dropout

    Args:
    - layers: an iterable of tuples with specifying which layer to add:
      - The first item is the layer type
        - If = "dropout" then add a dropout layer with dropout rate specified by the second item
        - Else, add a dense layer with num_units specified by the second item
        by the next two elements
    """
    model = keras.Sequential()
    model.add(layers.Dense(10, activation="linear"))
    for layer in model_layers:
        if layer[0] == "dropout":
            model.add(layers.Dense(layer[1]))
        else:
            model.add(layers.Dense(layer[1], activation="tanh"))

    # end layers are always the same
    model.add(
        layers.Dense(1, activation="tanh")
    )  # output is roughly normally distributed
    model.add(
        layers.Dense(1, activation="linear")
    )  # so match with tanh + linear scaling

    model.compile(
        optimizer="Adam",
        loss="mse",
    )
    return model


def fit_on_params(params, clusters, train):
    """Take a dict of parameters specifying models + parameters for fitting
    and a dict of clusters to pass to PoolRegressor for fitting models.
    """
    models = {}
    for model_name, config_dict in params.items():
        model = config_dict["model"].set_params(**config_dict["params"])
        pool_model = PoolRegressor(base_model=model, clusters=clusters)
        models[model_name] = pool_model.fit(train)
    return models


def params_to_preds(param_dict, train, test):
    """Take a dict of parameters for pool + single + pool all models, fit the models and return
    their predictions on the test set.
    """
    all_models = {
        k: fit_on_params(pd["params"], pd["clusters"], train)
        for k, pd in param_dict.items()
    }  # fit all models
    all_models_ = {
        f"{setup}_{model_type}": model
        for setup, model_dict in all_models.items()
        for model_type, model in model_dict.items()
    }
    return {k: model.predict(test) for k, model in all_models_.items()}


def score_from_df(pred_df, X):
    """Ensure indices are aligned before calculating the weighted correlation
    on the given predictions df.

    Note: the score will be nan if one df contains index values not found in the other.
    This likely indicates a bug in generating predictions since the predictions should
    have been calculated using X, so the indices should be the same in some order.
    """
    pred_df_reindexed = pred_df.reindex(index=X.index)
    return weighted_correlation(
        X.target.values,
        pred_df_reindexed.values,
        X.target_weight.values,
    )


def score_pool_model(model, X):
    """Convenience function for generating predictions and passing to score_from_df"""
    preds = model.predict(X)
    return score_from_df(preds, X)


class PoolRegressor(BaseEstimator, RegressorMixin):
    """Helper class for fitting pool models.

    Notes:
    - This depends on the input X being a pandas df. sklearn (deliberately) tends not
    to work well with pandas, but we don't use sklearn functionality extensively here
    and what we do use will be okay (for indexing sklearn seems to take care of things,
    see: https://github.com/scikit-learn/scikit-learn/blob/0d378913be6d7e485b792ea36e9268be31ed52d0/sklearn/utils/__init__.py#L307)
    """

    def __init__(self, base_model, clusters: dict):
        self.base_model = base_model
        self.clusters = clusters
        self.asset_ids_ = reduce(lambda x, y: [*x, *y], clusters.values())
        super().__init__()

    def fit(self, X, y=None, **fit_kwargs):
        """Expects a long df using the "targets" column as the targets, and
        any column without "target" in the name is used as a feature. Fit one
        model for each given cluster.

        Note: the case where each cluster has size 1 is the single asset model.
        """
        self.models_ = {}
        for cluster, asset_ids in self.clusters.items():
            X_subset, y_subset = get_xy_arrays(X.loc[(slice(None), list(asset_ids)), :])
            model_clone = clone(self.base_model)
            model_clone.fit(
                X_subset, y_subset, **fit_kwargs
            )  # fit separately for compatibility with Keras
            self.models_[cluster] = model_clone

        return self

    def predict(self, X) -> pd.DataFrame:
        """Take a long df of features and return a wide df of predictions
        with asset_ids as columns.
        """
        preds = []
        for cluster, asset_ids in self.clusters.items():
            X_subset = X.loc[(slice(None), asset_ids), :]
            cluster_preds = scale_predictions(
                self.models_[cluster], get_xy_arrays(X_subset)[0], X_subset.target_scale
            )
            preds.append(pd.Series(cluster_preds, index=X_subset.index))
            # asset_preds = self.models_[asset_id].predict(get_xy_arrays(X_subset)[0])
            # asset_preds = pd.Series(asset_preds, index=X_subset.index)
            # preds[asset_id] = asset_preds * X_subset.target_scale # scale back to returns predictions

        return pd.concat(preds).reindex(index=X.index)  # same order as input df

    def score(self, X, y=None):
        """Return the weighted correlation between all predictions"""
        return score_pool_model(self, X)


def single_asset_regressor(base_model, asset_ids):
    """Factory function for PoolRegressor for single assets"""
    single_clusters = {k: [k] for k in asset_ids}
    return PoolRegressor(base_model, single_clusters)


class PoolVotingRegressor(RegressorMixin):
    """Wrapper around VotingRegressor intended for use with PoolRegressors.
    In particular:
    - change the default scoring function to weighted regression
    - keep the original pandas indices to predictions
    """

    def __init__(self, estimators, weights=None):
        self.estimators = estimators
        self.weights = weights

    def fit(self, X, y=None):
        empty_y = np.empty_like(X.iloc[:, 0])
        self.voting_regressor_ = VotingRegressor(
            estimators=self.estimators, weights=self.weights
        )
        self.voting_regressor_.fit(X, empty_y)
        return self

    def predict(self, X):
        """Adds the original pandas index of X to the output of the wrapped
        VotingRegressor.
        """
        preds = self.voting_regressor_.predict(X)
        return pd.Series(preds, index=X.index)

    def score(self, X, y=None):  # FIXME: duplication from PoolRegressor
        """Return the weighted correlation between all predictions"""
        return score_pool_model(self, X)
