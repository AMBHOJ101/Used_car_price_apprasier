# 🚗 Used Car Price Appraiser

### Machine Learning Regression Project for Used Vehicle Price Estimation

A machine learning regression system that estimates the market price of used vehicles based on their specifications, mileage, age, condition, and other categorical attributes.

The project uses **Pandas, Category Encoders, and Scikit-Learn** to build an end-to-end machine learning pipeline covering data preprocessing, exploratory data analysis, feature engineering, target encoding, model training, evaluation, and price prediction.

---

## 📌 Project Overview

Buying or selling a used vehicle can be challenging because vehicle prices depend on multiple factors such as:

* Brand
* Model
* Manufacturing year
* Mileage
* Fuel type
* Engine
* Transmission
* Exterior and interior condition
* Accident history
* Title condition

This project develops a regression-based **Used Car Price Appraiser** that learns relationships between these vehicle characteristics and historical selling prices.

Given the specifications of a used car, the model estimates its expected market price.

### Problem Type

**Supervised Machine Learning → Regression**

### Target Variable

`price`

### Primary Evaluation Metrics

* **Mean Absolute Error (MAE)**
* **R² Score**

---

# 🎯 Objectives

The main objectives of this project are:

1. Analyze and understand used-car market data.
2. Clean numerical and categorical vehicle information.
3. Convert unstructured price and mileage values into numerical features.
4. Handle missing values appropriately.
5. Engineer meaningful features such as vehicle age and mileage per year.
6. Apply **Target Encoding** to high-cardinality categorical variables.
7. Train and compare multiple regression models.
8. Evaluate model performance using MAE and R².
9. Identify the most influential factors affecting vehicle prices.
10. Develop a reusable function for estimating the price of a new vehicle.

---

# 📊 Dataset

The project uses a used-car dataset containing approximately **4,000 vehicle records** and the following attributes:

| Feature        | Description              |
| -------------- | ------------------------ |
| `brand`        | Vehicle manufacturer     |
| `model`        | Vehicle model            |
| `model_year`   | Manufacturing/model year |
| `milage`       | Vehicle mileage          |
| `fuel_type`    | Type of fuel used        |
| `engine`       | Engine specification     |
| `transmission` | Transmission type        |
| `ext_col`      | Exterior color           |
| `int_col`      | Interior color           |
| `accident`     | Accident history         |
| `clean_title`  | Vehicle title condition  |
| `price`        | Target vehicle price     |

---

# 🛠️ Technology Stack

### Programming Language

* Python

### Data Processing

* Pandas
* NumPy

### Data Visualization

* Matplotlib

### Machine Learning

* Scikit-Learn

### Categorical Encoding

* Category Encoders
* Target Encoding

### Model Persistence

* Joblib

### Development Environment

* Google Colab / Jupyter Notebook

---

# 🧠 Machine Learning Workflow

The complete workflow follows a standard data science lifecycle:

```text
Raw Dataset
     │
     ▼
Data Understanding
     │
     ▼
Data Cleaning
     │
     ▼
Exploratory Data Analysis
     │
     ▼
Feature Engineering
     │
     ▼
Train/Test Split
     │
     ▼
Target Encoding
     │
     ▼
Model Training
     │
     ├───────────────┐
     ▼               ▼
Random Forest   Gradient Boosting
     │               │
     └───────┬───────┘
             ▼
       Model Evaluation
             │
       ┌─────┴─────┐
       ▼           ▼
      MAE          R²
       │           │
       └─────┬─────┘
             ▼
      Best Model Selection
             │
             ▼
      Used Car Price Appraiser
```

---

# 🧹 Data Preprocessing

Several preprocessing steps were required because some variables were stored as text rather than numerical values.

## Price Conversion

Original values:

```text
$10,300
$38,005
$54,598
```

were converted into numerical values:

```text
10300
38005
54598
```

## Mileage Conversion

Original values:

```text
51,000 mi.
34,742 mi.
22,372 mi.
```

were converted into numerical mileage values.

The feature was also renamed from:

```text
milage
```

to:

```text
mileage
```

for consistency.

---

# 🔧 Feature Engineering

Additional features were created to improve the model.

## Vehicle Age

Vehicle age was calculated using:

```text
Vehicle Age = Current Year − Model Year
```

This provides a more intuitive representation of depreciation than the manufacturing year alone.

## Mileage Per Year

A mileage utilization feature was created:

```text
Mileage Per Year = Mileage / Vehicle Age
```

This helps distinguish between vehicles that have accumulated similar total mileage over different periods.

## Accident Indicator

Accident history was converted into a binary feature:

```text
0 → No reported accident
1 → Accident reported
```

These engineered features provide the model with additional information about vehicle usage and condition.

---

# 🎯 Target Encoding

Categorical variables such as:

* Brand
* Model
* Engine
* Transmission
* Fuel type
* Exterior color
* Interior color
* Accident history
* Clean title

were transformed using **Target Encoding**.

Instead of creating a large number of one-hot encoded columns, Target Encoding represents categories using information derived from the target variable.

For example:

```text
Brand        Encoded Value
Toyota       Average Toyota Price
BMW          Average BMW Price
Audi         Average Audi Price
```

### Preventing Data Leakage

The encoder was fitted **only on the training dataset**:

```python
encoder.fit_transform(X_train, y_train)
```

and subsequently applied to the test dataset:

```python
encoder.transform(X_test)
```

This prevents information from the test set from influencing the training process.

---

# 🤖 Models

Two regression algorithms were evaluated.

## 1. Random Forest Regressor

Random Forest combines multiple decision trees to produce a robust prediction.

Advantages:

* Handles nonlinear relationships.
* Captures feature interactions.
* Less sensitive to feature scaling.
* Works well with mixed numerical and encoded categorical data.

---

## 2. Gradient Boosting Regressor

Gradient Boosting builds models sequentially, with each new model attempting to reduce errors made by previous models.

Advantages:

* Strong predictive performance.
* Captures nonlinear patterns.
* Effective for structured/tabular datasets.

---

# 📈 Model Evaluation

Two primary metrics were used.

## Mean Absolute Error — MAE

MAE measures the average absolute difference between actual and predicted prices.

```text
Lower MAE = Better Model
```

For example, an MAE of `$4,000` means that the model's predictions differ from actual prices by approximately $4,000 on average.

---

## R² Score

R² measures how much of the variation in vehicle prices is explained by the model.

```text
Higher R² = Better Model
```

A value closer to 1 indicates stronger predictive performance.

---

# 🏆 Model Performance

Replace the following values with the actual results from your notebook:

| Model             |      MAE |     R² |
| ----------------- | -------: | -----: |
| Random Forest     | `$X,XXX` | `0.XX` |
| Gradient Boosting | `$X,XXX` | `0.XX` |

### Final Model

The model with the best combination of **low MAE and high R²** was selected as the final price prediction model.

> **Final Model:** `[Random Forest / Gradient Boosting]`

> **MAE:** `$[YOUR VALUE]`

> **R²:** `[YOUR VALUE]`

---

# 📊 Exploratory Data Analysis

The project includes several visual analyses, including:

### Price Distribution

Analyzes the distribution and spread of used vehicle prices.

### Mileage vs Price

Examines the relationship between vehicle mileage and selling price.

### Vehicle Age vs Price

Analyzes how vehicle age influences market value.

### Brand vs Average Price

Compares average prices across different vehicle manufacturers.

### Actual vs Predicted Prices

Evaluates how closely model predictions match actual prices.

### Feature Importance

Identifies the variables contributing most strongly to price predictions.

---

# 🔍 Feature Importance

Feature importance analysis provides interpretability by identifying which vehicle attributes have the greatest influence on the model's predictions.

Potentially influential factors include:

* Vehicle model
* Mileage
* Vehicle age
* Brand
* Engine
* Transmission
* Accident history

The final ranking should be based on the trained model's feature importance results.

---

# 💡 Example Prediction

The trained system can receive information about a vehicle such as:

```python
example_car = {
    "brand": "Toyota",
    "model": "Camry",
    "model_year": 2020,
    "mileage": 35000,
    "fuel_type": "Gasoline",
    "engine": "2.5L I4",
    "transmission": "Automatic",
    "ext_col": "White",
    "int_col": "Black",
    "accident": "None reported",
    "clean_title": "Yes"
}
```

The model then generates an estimated market price:

```text
Estimated Vehicle Price: $XX,XXX
```

---

# 📁 Project Structure

```text
used-car-price-appraiser/
│
├── data/
│   └── used_cars.csv
│
├── notebooks/
│   └── used_car_price_prediction.ipynb
│
├── models/
│   ├── used_car_price_model.pkl
│   └── target_encoder.pkl
│
├── src/
│   ├── preprocessing.py
│   ├── train.py
│   └── predict.py
│
├── app/
│   └── app.py
│
├── requirements.txt
├── README.md
└── .gitignore
```

---

# 🚀 Future Improvements

The current project can be extended into a production-ready application by:

* Hyperparameter optimization using GridSearchCV or RandomizedSearchCV.
* Cross-validation for more robust model evaluation.
* Adding additional vehicle datasets.
* Using logarithmic transformation for highly skewed prices.
* Implementing advanced boosting algorithms.
* Adding explainable AI using SHAP.
* Building an interactive Streamlit dashboard.
* Deploying the model as a web application.
* Adding confidence intervals or prediction ranges.
* Integrating real-time vehicle market data.

---

# 📌 Key Learning Outcomes

Through this project, the following practical machine learning skills were developed:

* Data cleaning with Pandas
* Exploratory Data Analysis
* Feature engineering
* Handling missing values
* Categorical variable encoding
* Target Encoding
* Regression modeling
* Train/test splitting
* Model evaluation
* MAE and R² interpretation
* Feature importance analysis
* Model comparison
* Model serialization with Joblib
* End-to-end machine learning workflow

---

# 👨‍💻 Author

**Ambhoj Verma**

---

# ⭐ Project Highlights

> **End-to-end regression system for estimating used vehicle prices using vehicle specifications, condition, mileage, and historical pricing data.**

**Core Technologies:** Python · Pandas · NumPy · Scikit-Learn · Category Encoders · Matplotlib · Joblib

**Machine Learning:** Random Forest Regression · Gradient Boosting Regression · Target Encoding

**Evaluation:** MAE · R² Score
