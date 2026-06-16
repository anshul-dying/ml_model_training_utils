import pandas as pd
import numpy as np
from sklearn import metrics
from sklearn import model_selection
from sklearn.base import RegressorMixin, clone
from collections.abc import Sequence, Callable

from sklearn.utils.validation import check_is_fitted
from sklearn.exceptions import NotFittedError

import pickle

def make_kfold(
    data: pd.DataFrame,
    target: str,
    folds: int,
    random_state: int = 42,
    path_to_save: str | None = None,
    filename: str | None = None,  
    save: bool = False
) -> pd.DataFrame:
    """
    Create stratified folds for regression problems.

    Parameters
    ----------
    data : pd.DataFrame
        Input dataframe.

    target : str
        Name of the target column.

    folds : int
        Number of folds.

    path_to_save : str
        Directory where the dataframe will be saved.

    filename : str
        Name of the CSV file without extension.

    save : bool, default=False
        Whether to save the dataframe to disk.

    Returns
    -------
    pd.DataFrame
        DataFrame with an additional 'kfold' column.
    """
    data = data.copy()
    data['kfold'] = -1
    data = data.sample(frac=1, random_state=random_state).reset_index(drop=True)
    n_bins = round(1+np.log2(len(data)))
    data.loc[:, 'bins'] = pd.cut(
        data[target], bins=n_bins, labels=False
    )

    skf = model_selection.StratifiedKFold(n_splits=folds, shuffle=True, random_state=random_state)

    for f, (t_, v_) in enumerate(skf.split(data, data.bins.values)):
        data.loc[v_, 'kfold'] = f

    data = data.drop(columns='bins')
    if save:
        if path_to_save is None or filename is None:
            raise ValueError(
                "path_to_save and filename must be provided when save=True."
            )

        data.to_csv(
            f"{path_to_save}/{filename}.csv",
            index=False
        )

    return data
                   
def train_with_kfolds(
    data: pd.DataFrame,
    models: dict[str, RegressorMixin],
    features: Sequence[str],
    target: str,
    folds: int = 5,
    scoring: Callable = metrics.root_mean_squared_error
) -> tuple[dict[str, list], dict[str, float]]:
    """
    Train multiple regression models using K-fold cross-validation.

    Parameters
    ----------
    data : pd.DataFrame
        DataFrame containing the features, target column, and a
        'kfold' column specifying fold assignments.

    models : dict
        Dictionary mapping model names to scikit-learn compatible
        regression estimators.

    features : array-like
        Names of the feature columns used for training.

    target : str
        Name of the target column.

    folds : int
        Number of folds used for cross-validation.

    scoring : Callable, default=metrics.root_mean_squared_error
        Metric function used to evaluate predictions. Must accept
        (y_true, y_pred) and return a scalar score.

    Returns
    -------
    tuple[dict[str, list], dict[str, float]]
        - First dictionary contains fold-wise scores.
        - Second dictionary contains mean scores.

    Notes
    -----
    This function assumes that the input DataFrame contains a
    'kfold' column with values ranging from 0 to folds - 1.

    Examples
    --------
    >>> models = {
    ...     "LinearRegression": LinearRegression(),
    ...     "RandomForest": RandomForestRegressor()
    ... }
    >>> train_kfold_with_models(
    ...     data=df,
    ...     models=models,
    ...     features=feature_cols,
    ...     target="price",
    ...     folds=5,
    ...     scoring=metrics.mean_squared_error
    ... )
    {'LinearRegression': 4.82, 'RandomForest': 3.15}
    """

    scores = {model_name: [] for model_name in models}
    for fold in range(folds):
        train = data[data.kfold != fold]
        valid = data[data.kfold == fold]

        X_train, y_train = train[features], train[target]
        X_valid, y_valid = valid[features], valid[target]

        for model_name, model in models.items():
            curr_model = clone(model)
            curr_model.fit(X_train, y_train)
            preds = curr_model.predict(X_valid)
            score = scoring(y_valid, preds)
            scores[model_name].append(score)
            print(
                f"Fold {fold+1} | "
                f"{model_name}: {score:.4f}"
            )
        print("-"*50)

    mean_scores = {
        model_name: np.mean(model_scores)
        for model_name, model_scores in scores.items()
    }

    return scores, mean_scores

def train_single_model(
    data: pd.DataFrame,
    features: Sequence[str],
    target: str,
    model: RegressorMixin,
    folds: int = 5,
    scoring: Callable = metrics.root_mean_squared_error
) -> tuple[list[float], float]:
    """
    Train a single regression model using K-fold cross-validation.

    Parameters
    ----------
    data : pd.DataFrame
        DataFrame containing the feature columns, target column,
        and a 'kfold' column specifying fold assignments.

    features : Sequence[str]
        Names of the feature columns used for training.

    target : str
        Name of the target column.

    model : RegressorMixin
        Scikit-learn compatible regression estimator.

    folds : int, default=5
        Number of folds used for cross-validation.

    scoring : Callable, default=metrics.root_mean_squared_error
        Metric function used to evaluate predictions. Must accept
        (y_true, y_pred) and return a scalar score.

    Returns
    -------
    tuple[list[float], float]
        A tuple containing:

        - list[float]: Score obtained on each validation fold.
        - float: Mean score across all folds.

    Notes
    -----
    A fresh clone of the provided model is trained on each fold
    using sklearn.base.clone to prevent state leakage between
    folds.

    This function assumes that the input DataFrame contains a
    'kfold' column with values ranging from 0 to folds - 1.

    Examples
    --------
    >>> from sklearn.ensemble import RandomForestRegressor
    >>> scores, mean_score = train_single_model(
    ...     data=df,
    ...     features=feature_cols,
    ...     target="price",
    ...     model=RandomForestRegressor(),
    ...     folds=5
    ... )
    >>> print(scores)
    [2.81, 2.67, 2.74, 2.88, 2.79]
    >>> print(mean_score)
    2.778
    """

    scores = []
    for fold in range(folds):
        train = data[data.kfold != fold]
        valid = data[data.kfold == fold]

        X_train, y_train = train[features], train[target]
        X_valid, y_valid = valid[features], valid[target]

        curr_model = clone(model)
        curr_model.fit(X_train, y_train)
        preds = curr_model.predict(X_valid)
        
        score = scoring(y_valid, preds)
        print(f"Fold {fold+1} | Score: {score}")
        print("-"*50)
        scores.append(score)

    return scores, np.mean(scores)  
    
def make_oof_predictions(
    data: pd.DataFrame,
    features: Sequence[str],
    target: str,
    models: dict[str,RegressorMixin],
    folds: int = 5,
    scoring: Callable = metrics.root_mean_squared_error
) -> tuple[
    dict[str, np.ndarray],
    dict[str, float]
]:
    """
    Generate Out-Of-Fold (OOF) predictions for multiple regression models
    using K-fold cross-validation.

    Parameters
    ----------
    data : pd.DataFrame
        DataFrame containing the feature columns, target column,
        and a 'kfold' column specifying fold assignments.

    features : Sequence[str]
        Names of the feature columns used for training.

    target : str
        Name of the target column.

    models : dict[str, RegressorMixin]
        Dictionary mapping model names to scikit-learn compatible
        regression estimators.

    folds : int, default=5
        Number of folds used for cross-validation.

    scoring : Callable, default=metrics.root_mean_squared_error
        Metric function used to evaluate predictions. Must accept
        (y_true, y_pred) and return a scalar score.

    Returns
    -------
    tuple[dict[str, np.ndarray], dict[str, float]]

        A tuple containing:

        - dict[str, np.ndarray]
            Dictionary mapping each model name to its Out-Of-Fold
            predictions. Each prediction array has the same length
            as the input dataset.

        - dict[str, float]
            Dictionary mapping each model name to its overall OOF
            score computed using the provided scoring metric.

    Notes
    -----
    For each fold, a fresh clone of every estimator is trained on
    all folds except the current validation fold. Predictions for
    the validation fold are stored at their original row indices,
    ensuring that every sample is predicted by a model that was
    not trained on that sample.

    Out-Of-Fold predictions provide an unbiased estimate of model
    performance and are commonly used for:

    - Cross-validation evaluation.
    - Error analysis.
    - Model comparison.
    - Stacking and blending ensembles.

    This function assumes that the input DataFrame contains a
    'kfold' column with values ranging from 0 to folds - 1.

    Examples
    --------
    >>> from sklearn.linear_model import LinearRegression
    >>> from sklearn.ensemble import RandomForestRegressor

    >>> models = {
    ...     "LinearRegression": LinearRegression(),
    ...     "RandomForest": RandomForestRegressor()
    ... }

    >>> oof_preds, oof_scores = make_oof_predictions(
    ...     data=df,
    ...     features=feature_cols,
    ...     target="price",
    ...     models=models,
    ...     folds=5
    ... )

    >>> oof_scores
    {
        'LinearRegression': 2.81,
        'RandomForest': 2.47
    }
    """
    oof_preds = {
        model_name: np.zeros(len(data)) for model_name in models
    }

    for fold in range(folds):
        train = data[data.kfold != fold]
        valid = data[data.kfold == fold]

        X_train, y_train = train[features], train[target]
        X_valid = valid[features]

        for model_name, model in models.items():
            curr_model = clone(model)
            curr_model.fit(X_train, y_train)
            preds = curr_model.predict(X_valid)

            score = scoring(
                valid[target],
                preds
            )

            oof_preds[model_name][valid.index] = preds
            print(
                f"Fold {fold+1} | "
                f"{model_name}: {score:.4f}"
            )
        print("-"*50)

    oof_scores = {
        model_name: scoring(data[target], oof_preds[model_name])
        for model_name in models
    }

    return oof_preds, oof_scores

def feature_importance(
    model: RegressorMixin,
    features: Sequence[str],
    method: str = 'auto'
) -> pd.DataFrame:
    """
    Extract and rank feature importances from a fitted regression model.

    Parameters
    ----------
    model : RegressorMixin
        A fitted scikit-learn compatible regression estimator.

        The estimator must expose either:

        - ``feature_importances_`` (e.g. tree-based models), or
        - ``coef_`` (e.g. linear models)

        depending on the selected method.

    features : Sequence[str]
        Names of the feature columns corresponding to the model
        inputs.

    method : {"auto", "tree", "coefficients"}, default="auto"
        Method used to extract feature importance values.

        - ``"auto"``:
            Automatically uses ``feature_importances_`` when
            available, otherwise falls back to ``coef_``.
        - ``"tree"``:
            Uses ``feature_importances_``.
        - ``"coefficients"``:
            Uses ``coef_``.

    Returns
    -------
    pd.DataFrame
        DataFrame containing:

        - ``feature`` : Feature name.
        - ``importance`` : Importance value associated with the
          feature.

        The DataFrame is sorted in descending order of importance.

    Raises
    ------
    ValueError
        If:

        - ``method`` is not one of
          ``{"auto", "tree", "coefficients"}``.
        - The estimator does not expose the required attribute for
          the selected method.
        - Feature importances cannot be extracted from the model.

    Notes
    -----
    The model must be fitted before calling this function.

    For tree-based estimators such as RandomForestRegressor and
    XGBRegressor, importances are obtained from
    ``feature_importances_``.

    For linear estimators such as LinearRegression, Ridge, and
    Lasso, importances are obtained from ``coef_``.

    Examples
    --------
    >>> model.fit(X_train, y_train)
    >>> feature_importance(
    ...     model=model,
    ...     features=X_train.columns
    ... )
         feature  importance
    0      area    0.4215
    1  bedrooms    0.2318
    2       age    0.1182
    """

    try:
        check_is_fitted(model)
    except NotFittedError:
        raise ValueError(
            f"{type(model).__name__} must fitted before "
            "calling feature_importance()"
        )

    if method not in ['auto', 'tree', 'coefficients']:
        raise ValueError("method parameter is wrong. it must be one of ['auto', 'tree', 'coefficients']")
    if method == 'auto':
        if hasattr(model, "feature_importances_"):
            imp = np.array(model.feature_importances_)
        elif hasattr(model, "coef_"):
            imp = np.array(model.coef_)
        else:
            raise ValueError(f"Cannot extract importances from {type(model).__name__}.")
    elif method == 'tree':
        if not hasattr(model, "feature_importances_"):
            raise ValueError(
                f"{type(model).__name__} does not support tree importances."
            )
        imp = np.array(model.feature_importances_)
    elif method == 'coefficients':
        if not hasattr(model, "coef_"):
            raise ValueError(
                f"{type(model).__name__} does not expose coefficients."
            )
        imp = np.array(model.coef_)

    imp = np.atleast_1d(imp)
    df = pd.DataFrame({
        "feature": features,
        "importance": imp
    }).sort_values("importance", ascending=False).reset_index(drop=True)

    return df

def save_model(
    model: RegressorMixin,
    path: str
):
    """
    Save a fitted regression model to disk using pickle.

    Parameters
    ----------
    model : RegressorMixin
        The fitted scikit-learn compatible regression estimator to save.
    path : str
        The file path where the model should be saved.

    Raises
    ------
    ValueError
        If the model is not fitted before saving.
    """
    try:
        check_is_fitted(model)
    except NotFittedError:
        raise ValueError(
            f"{type(model).__name__} must be fitted before saving."
        )
    
    with open(path, 'wb') as file:
        pickle.dump(model, file)
    
def load_model(path: str):
    """
    Load a saved regression model from disk using pickle.

    Parameters
    ----------
    path : str
        The file path from which to load the model.

    Returns
    -------
    RegressorMixin or Any
        The loaded estimator or object.
    """
    with open(path, 'rb') as f:
        model = pickle.load(f)
    return model