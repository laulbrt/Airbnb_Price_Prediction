# Airbnb Price Prediction – Machine Learning Project

This project focuses on predicting Airbnb listing prices using **machine learning techniques** applied to a structured dataset.  
It includes data preprocessing, exploratory data analysis, feature engineering, text processing, model training, and final prediction generation.

The project was developed as part of an academic assignment in data science.

---

## Project Overview

The goal of this project is to predict the logarithm of the price (`log_price`) of an Airbnb listing from its characteristics.  
The dataset contains numerical features, categorical features, and textual descriptions, making it suitable for mixed-feature machine learning.

The workflow includes:

- Data exploration and cleaning  
- Handling missing values and inconsistent entries  
- Categorical encoding and normalization  
- Text preprocessing using TF-IDF  
- Model selection and evaluation  
- Generating final predictions for the test dataset  

The objective is to design a robust modeling pipeline that balances interpretability and predictive performance.

---

## Main Features

### Data Preparation
- Load and inspect the training and test datasets  
- Clean missing values  
- Encode categorical variables  
- Normalize numerical features  
- Preprocess text fields (cleaning, tokenization, TF-IDF)  

### Exploratory Data Analysis
- Statistical summaries  
- Distribution visualizations  
- Correlation analysis  
- Detection of outliers and influential variables  

### Feature Engineering
- Construction of numerical indicators  
- Aggregation of review information  
- Encoding of location-based attributes  
- Vectorization of textual descriptions  

### Model Training
- Comparison of baseline models  
- Linear Regression and Regularized Models  
- Random Forest / Gradient Boosting  
- Cross-validation and hyperparameter tuning  

### Prediction Generation
- Evaluate model performance  
- Select best-performing model  
- Generate predictions for the test dataset  
- Export the final CSV file  

---

## How It Works

1. Load the training and test CSV files.  
2. Explore and clean the data.  
3. Transform numerical, categorical, and textual features.  
4. Train multiple models and compare their performance.  
5. Select the best model based on validation metrics.  
6. Use the selected model to predict Airbnb prices.  
7. Export final predictions into a submission file.

---

## Technologies Used

| Purpose | Technology |
|---------|------------|
| Data Manipulation | pandas, numpy |
| Visualization | matplotlib, seaborn |
| Machine Learning | scikit-learn |
| Text Processing | TF-IDF (scikit-learn) |
| Environment | Jupyter Notebook |
| Python Version | 3.11 |

---

## Files Included

- `airbnb_train.csv` – Training dataset  
- `airbnb_test.csv` – Test dataset  
- `prediction_example.csv` – Example submission  
- `prediction_final.csv` – Final predictions  
- `projet_airbnb_notebook_final.ipynb` – Main Jupyter notebook  
- `Explication.txt` – Additional notes  
- `projet description.pdf` – Assignment instructions  

---

## Installation and Execution

### Clone the repository

```bash
git clone https://github.com/<your-username>/airbnb-project.git
cd airbnb-project

Install dependencies
pip install -r requirements.txt

Launch the notebook
jupyter notebook

Open:

projet_airbnb_notebook_final.ipynb

License

This project was created for academic purposes.
