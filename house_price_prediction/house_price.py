# ==========================================
# HOUSE PRICE PREDICTION USING LINEAR REGRESSION
# ==========================================

# Step 1 : Import Libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error

# ==========================================
# Step 2 : Load Dataset
# ==========================================

df = pd.read_csv("data.csv")
print("========== FIRST 5 ROWS ==========")
print(df.head())
print("\n========== DATASET SHAPE ==========")
print(df.shape)
print("\n========== DATASET INFO ==========")
print(df.info()) 

# ==========================================
# Step 3 : Check Missing Values
# ==========================================

print("\n========== MISSING VALUES ==========")
print(df.isnull().sum())

# ==========================================
# Step 4 : Keep Only Required Columns
# ==========================================

df = df[
    [
        "price",
        "bedrooms",
        "bathrooms",
        "sqft_living",
        "sqft_lot",
        "floors",
        "waterfront",
        "view",
        "condition",
        "sqft_above",
        "sqft_basement",
        "yr_built",
        "yr_renovated"
    ]
]

# Remove invalid prices
df = df[df["price"] > 0]
print("\n========== CLEANED DATA ==========")
print(df.head())

# ==========================================
# Step 5 : Split Features and Target
# ==========================================

X = df.drop("price", axis=1)
y = df["price"]
print("\nFeature Shape :", X.shape)
print("Target Shape :", y.shape)

# ==========================================
# Step 6 : Split Dataset
# ==========================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42
)
print("\nTraining Shape :", X_train.shape)
print("Testing Shape :", X_test.shape)

# ==========================================
# Step 7 : Create Model
# ==========================================

model = LinearRegression()

# ==========================================
# Step 8 : Train Model
# ==========================================

model.fit(X_train, y_train)
print("\n Linear Regression Model Trained Successfully")

# ==========================================
# Step 9 : Prediction on Test Data
# ==========================================

y_pred = model.predict(X_test)

# ==========================================
# Step 10 : Evaluate Model
# ==========================================

print("\n========== MODEL PERFORMANCE ==========")
print("R2 Score :", r2_score(y_test, y_pred))
print("Mean Squared Error :", mean_squared_error(y_test, y_pred))

# ==========================================
# Step 11 : Compare Actual vs Predicted
# ==========================================

result = pd.DataFrame({
    "Actual Price": y_test.values,
    "Predicted Price": y_pred
})
print("\n========== ACTUAL vs PREDICTED ==========")
print(result.head(10))
print("\n========== ENTER NEW HOUSE DETAILS ==========")
bedrooms = float(input("Enter Bedrooms : "))
bathrooms = float(input("Enter Bathrooms : "))
sqft_living = float(input("Enter Living Area (sqft) : "))
sqft_lot = float(input("Enter Lot Area (sqft) : "))
floors = float(input("Enter Floors : "))
waterfront = float(input("Enter Waterfront (0 or 1) : "))
view = float(input("Enter View (0-4) : "))
condition = float(input("Enter Condition (1-5) : "))
sqft_above = float(input("Enter Above Ground Area (sqft) : "))
sqft_basement = float(input("Enter Basement Area (sqft) : "))
yr_built = float(input("Enter Year Built : "))
yr_renovated = float(input("Enter Year Renovated (0 if never renovated) : "))
new_house = pd.DataFrame({
    "bedrooms": [bedrooms],
    "bathrooms": [bathrooms],
    "sqft_living": [sqft_living],
    "sqft_lot": [sqft_lot],
    "floors": [floors],
    "waterfront": [waterfront],
    "view": [view],
    "condition": [condition],
    "sqft_above": [sqft_above],
    "sqft_basement": [sqft_basement],
    "yr_built": [yr_built],
    "yr_renovated": [yr_renovated]
})
predicted_price = model.predict(new_house)
print("\n================================")
print("Predicted House Price :", round(predicted_price[0], 2))
print("================================")