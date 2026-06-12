# ML Model Training Utils

Utility functions for training and evaluating machine learning models with K-Fold Cross Validation.

## Installation
```bash
git clone https://github.com/anshul-dying/ml_model_training_utils.git
```
```bash
cd ml_model_training_utils
```

### Install dependencies:
```bash
pip install pandas numpy scikit-learn
```

### Example
```python
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor

from utils_regression import (
    make_kfold,
    train_with_kfolds
)

# Load your dataset
df = pd.read_csv("train.csv")

# Create folds
df = make_kfold(
    data=df,
    target="target",
    folds=5
)

# Define features
features = [
    col for col in df.columns
    if col not in ["target", "kfold"]
]

# Define models
models = {
    "LinearRegression": LinearRegression(),
    "RandomForest": RandomForestRegressor()
}

# Train models
scores, mean_scores = train_with_kfolds(
    data=df,
    models=models,
    features=features,
    target="target"
)

print(mean_scores)
```

## Acknowledgements
Special thanks to Abhishek Thakur Sir and his book Approaching (Almost) Any Machine Learning Problem for helping me learn many of the concepts used in this project.

**Note:** docstrings and documentation were generated with AI and reviewed before being added to the project.
