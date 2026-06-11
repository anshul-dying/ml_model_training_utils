# utils_regression.py
# ├── make_kfold()
# ├── train_with_kfolds()
# ├── train_single_model()
# ├── make_oof_predictions()
# ├── feature_importance()
# ├── save_model()
# └── load_model()

import pandas as pd
import numpy as np
from sklearn import metrics
from sklearn import model_selection
from sklearn.base import RegressorMixin
from collections.abc import Sequence, Callable

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
    data = data.sample(frac=1).reset_index(drop=True)
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
) -> dict[str, float]:
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
    dict[str, float]
        Dictionary mapping each model name to its mean score across
        all folds.

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
            model.fit(X_train, y_train)
            preds = model.predict(X_valid)
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

    return mean_scores

def train_single_model():
    pass

def make_oof_predictions():
    pass