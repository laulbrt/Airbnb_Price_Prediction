# Airbnb price prediction

Predicting the log price of an Airbnb listing from its characteristics. The dataset mixes
numerical fields, categorical fields and free text descriptions, so the interesting part is
getting three very different feature types into the same model.

Course project, ESILV.

## Approach

**Preparation.** Load the train and test sets, inspect missing values and inconsistent entries,
encode categorical variables, normalise the numerical ones.

**Text.** Listing descriptions are cleaned, tokenised and vectorised with TF-IDF, then
concatenated with the tabular features.

**Exploration.** Distributions, correlations, outlier detection, and a look at which variables
actually carry signal on price.

**Feature engineering.** Aggregated review indicators, location encoding, and derived numerical
features.

**Modelling.** Linear regression and regularised variants as a baseline, then random forest and
gradient boosting, compared by cross-validation with hyperparameter tuning.

**Output.** The best model is refit and used to generate predictions on the test set, exported
as a submission CSV.

## Stack

Python, pandas, NumPy, scikit-learn, Matplotlib
