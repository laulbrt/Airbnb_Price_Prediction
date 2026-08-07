# Airbnb price prediction

Predicting the log price of an Airbnb listing from 27 features that mix numbers, categories,
dates and free text. Data analysis project, ESILV. Othman Kharroubi and Laura Labarthe.

22,234 listings for training, 51,877 to predict. The target is `log_price`, so the submission
has to be back in log space whatever the model outputs.

## Submitted model

XGBoost inside a scikit-learn pipeline, so preprocessing and model are fitted together and the
same transformations apply to the test set without leaking anything.

```
ColumnTransformer
  numeric      SimpleImputer(median)
  categorical  OneHotEncoder
  boolean      passed through
XGBRegressor(n_estimators=100, learning_rate=0.1, max_depth=6)
```

Features used: `accommodates`, `bathrooms`, `bedrooms`, `beds`, `number_of_reviews`,
`review_scores_rating` for the numbers; `property_type`, `room_type`, `bed_type`,
`cancellation_policy`, `city` one-hot encoded; `cleaning_fee`, `host_has_profile_pic`,
`host_identity_verified` and `instant_bookable` converted from the `t` and `f` strings the
dataset ships them as.

| Metric | Validation |
|---|---|
| R² | 0.6552 |
| MAE | 0.3171 |
| RMSE | 0.4203 |

## Exploration notebook

A second notebook covers the parts that did not make it into the final pipeline.

**Derived features.** `host_age_days` and `review_age_days` from the date columns, `name_len` and
`desc_len` from the text, a boolean for the cleaning fee, and `dist_center_km`, the geodesic
distance from the listing coordinates to the city centre.

**Text.** TF-IDF on the description, plugged into the ColumnTransformer alongside the numeric and
categorical branches.

**Model comparison** on a held-out split:

| Model | RMSE |
|---|---|
| Mean baseline | 0.7159 |
| Linear regression | 0.4677 |
| Random forest | 0.4189 |

A note on those numbers: the conclusion cell of the exploration notebook reports cross-validated
RMSE values around 0.0015, which is not credible for this task and points to a column leaking the
target into the features. The figures above are the ones measured on a proper held-out split, and
they are the ones worth reading. The submitted XGBoost model lands in the same range at 0.4203,
which is roughly a 40% improvement over predicting the mean.

## Running it

```bash
pip install pandas numpy scikit-learn xgboost matplotlib seaborn geopy nltk
jupyter notebook airbnb_xgboost_submission.ipynb
```

`airbnb_train.csv` and `airbnb_test.csv` are 94 MB together and are not versioned here. Their
columns are listed in `dataset_description.txt`.

## Files

- `airbnb_xgboost_submission.ipynb`: the submitted pipeline, step by step
- `airbnb_exploration_and_models.ipynb`: exploration, feature engineering, model comparison
- `prediction_final.csv`: predictions on the test set
- `dataset_description.txt`: column list and the log price instruction
- `projet_description.pdf`: assignment brief

## Stack

Python, pandas, scikit-learn, XGBoost, seaborn, geopy
