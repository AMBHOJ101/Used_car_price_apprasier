
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
from category_encoders import TargetEncoder

from preprocessing import load_data, clean_data


# Load data
df = load_data("../data/used_cars.csv")

# Clean data
df = clean_data(df)


# Features and target
X = df.drop("price", axis=1)
y = df["price"]


categorical_features = ["brand", "model", "fuel_type", "engine", "transmission", "ext_col", "int_col", "clean_title"]

numerical_features = ["model_year", "mileage", "vehicle_age", "mileage_per_year", "accident"]


# Train-test split
X_train, X_test, y_train, y_test = train_test_split( X, y, test_size=0.20, random_state=42)


# Target Encoding
encoder = TargetEncoder( cols=categorical_features, smoothing=10)

X_train_encoded = encoder.fit_transform(X_train,y_train)

X_test_encoded = encoder.transform(X_test)


# Model
model = RandomForestRegressor( n_estimators=300, random_state=42, n_jobs=-1)


# Train
model.fit(X_train_encoded,y_train)

# Prediction
predictions = model.predict(X_test_encoded)


# Evaluation
mae = mean_absolute_error( y_test, predictions)

r2 = r2_score( y_test, predictions)


print(f"MAE: ${mae:,.2f}")
print(f"R²: {r2:.4f}")