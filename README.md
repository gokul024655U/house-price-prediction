# House Price Prediction using Linear Regression

## 📌 Project Overview

This project focuses on predicting house prices using Machine Learning. A Linear Regression model is trained using housing-related features such as the number of bedrooms, bathrooms, living area, lot area, floors, waterfront information, view, condition, and other property details.

The project demonstrates a basic end-to-end Machine Learning workflow using Python and Scikit-learn.

## 🎯 Objective

The main objective of this project is to build a Machine Learning model that can learn the relationship between house features and their prices and use that model to predict the price of a new house.

## 🛠️ Technologies Used

* Python
* Pandas
* Scikit-learn
* Linear Regression

## 📊 Dataset

The dataset contains **4,600 house records** with **18 columns**.

The selected features used for prediction include:

* Bedrooms
* Bathrooms
* Living Area (`sqft_living`)
* Lot Area (`sqft_lot`)
* Floors
* Waterfront
* View
* Condition
* Above Ground Area (`sqft_above`)
* Basement Area (`sqft_basement`)
* Year Built (`yr_built`)
* Year Renovated (`yr_renovated`)

The target variable is:

* `price`

## 🔄 Project Workflow

1. Load the dataset using Pandas
2. Explore the dataset
3. Check for missing values
4. Select the required features
5. Remove invalid price values
6. Separate features and target variable
7. Split the data into training and testing sets
8. Train a Linear Regression model
9. Generate predictions
10. Evaluate the model
11. Compare actual and predicted prices
12. Predict the price of a new house using user-provided details

## 🤖 Machine Learning Model

The project uses:

**Linear Regression**

The dataset was divided into:

* **80% Training Data**
* **20% Testing Data**

A `random_state` of `42` was used to make the train-test split reproducible.

## 📈 Model Performance

The model achieved the following results on the test dataset:

| Metric             |            Result |
| ------------------ | ----------------: |
| R² Score           |            0.6006 |
| Mean Squared Error | 59,413,057,706.82 |

The R² score indicates that the model explains approximately 60% of the variation in the target prices for this test split.

## 🏠 Sample Prediction

The trained model was also tested with new house details.

Example prediction:

**Predicted House Price: 1,085,276.46**

The displayed price follows the original unit/currency of the dataset.

## 💻 How to Run

### 1. Clone the repository

```bash
git clone https://github.com/YOUR-USERNAME/house-price-prediction.git
```

### 2. Open the project folder

```bash
cd house-price-prediction
```

### 3. Install the required libraries

```bash
pip install pandas scikit-learn
```

### 4. Run the Python program

```bash
python house_price.py
```

### 5. Enter the house details

The program will ask for details such as:

* Bedrooms
* Bathrooms
* Living area
* Lot area
* Floors
* Waterfront
* View
* Condition
* Above-ground area
* Basement area
* Year built
* Year renovated

The trained model will then generate a predicted house price.

## 📁 Project Structure

```text
house-price-prediction/
│
├── house_price.py
├── data.csv
├── README.md
└── screenshots/
```

## 📚 Learning Outcomes

Through this project, I gained practical experience in:

* Data loading and exploration
* Data preprocessing
* Feature selection
* Train-test splitting
* Linear Regression
* Model evaluation
* Actual vs. predicted value comparison
* Making predictions using new input data
* Building a basic end-to-end Machine Learning workflow

## 🚀 Future Improvements

Possible improvements for this project include:

* Testing additional regression algorithms
* Comparing multiple model performances
* Adding MAE and RMSE metrics
* Performing more detailed exploratory data analysis
* Feature engineering
* Hyperparameter tuning
* Creating a user-friendly Streamlit interface
