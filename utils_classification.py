import pandas as pd
import numpy as np
from sklearn import model_selection
from sklearn.base import ClassifierMixin
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
    some function that does stuff to your data and saves to the specified path
    """
    data = data.copy()
    data['kfold'] = -1
    data = data.sample(frac=1, random_state=random_state).reset_index(drop=True)
    skf = model_selection.StratifiedKFold(n_splits=folds, shuffle=True, random_state=random_state)
    for f, (t_, v_) in enumerate(skf.split(data, data[target])):
        data.loc[v_, 'kfold'] = f
    
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

def save_model(
    model: ClassifierMixin,
    path: str
):
    try:
        check_is_fitted(model)
    except NotFittedError:
        raise ValueError(f"{type(model).__name__} must be fitted before saving.")
    
    with open(path, 'wb') as f:
        pickle.dump(model, f)

def load_model(
    path: str
):
    with open(path, 'rb') as f:
        model = pickle.load(f)
    return model