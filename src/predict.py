import joblib
import pandas as pd

from preprocessing import CURRENT_YEAR


model = joblib.load(
    "../models/used_car_price_model.pkl"
)

encoder = joblib.load(
    "../models/target_encoder.pkl"
)


categorical_features = ["brand", "model", "fuel_type", "engine", "transmission", "ext_col", "int_col", "clean_title"]

numerical_features = ["model_year", "mileage", "vehicle_age", "mileage_per_year", "accident"]

def predict_car_price(car_data):

    car_df = pd.DataFrame([car_data])

    car_df["vehicle_age"] = (
        CURRENT_YEAR - car_df["model_year"]
    )

    car_df["mileage_per_year"] = (
        car_df["mileage"] /
        car_df["vehicle_age"].replace(0, 1)
    )

    for col in categorical_features:
        car_df[col] = car_df[col].fillna("Unknown")

    encoded = encoder.transform(car_df)

    prediction = model.predict(encoded)

    return prediction[0]