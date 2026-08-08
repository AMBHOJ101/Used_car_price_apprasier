import pandas as pd

CURRENT_YEAR = 2026


def load_data(path):
    return pd.read_csv(path)


def clean_data(df):

    df = df.copy()

    # Clean price
    df["price"] = (
        df["price"]
        .str.replace("$", "", regex=False)
        .str.replace(",", "", regex=False)
        .astype(int)
    )

    # Clean mileage
    df["milage"] = (
        df["milage"]
        .str.replace(",", "", regex=False)
        .str.replace(" mi.", "", regex=False)
        .astype(int)
    )

    df.rename(columns={"milage": "mileage"}, inplace=True)

    df["clean_title"] = df["clean_title"].fillna("No")

    # Missing values
    categorical_columns = [
        "fuel_type",
        "accident",
    ]

    for col in categorical_columns:
        df[col] = df[col].fillna("Unknown")

    # Feature engineering
    df["vehicle_age"] = (
        CURRENT_YEAR - df["model_year"]
    )

    df["mileage_per_year"] = (
        df["mileage"] /
        df["vehicle_age"].replace(0, 1)
    )

    df["accident"] = (
        df["accident"]
        .str.contains(
            "accident",
            case=False,
            na=False
        )
        .astype(int)
    )

    ext_col_stats = df.ext_col.value_counts(ascending=False)
    ext_col_less_than_10 = ext_col_stats[ext_col_stats <=10]
    df.ext_col=df.ext_col.apply(lambda x: 'other' if x in ext_col_less_than_10 else x)
    df.ext_col.nunique()

    int_col_stats = df.int_col.value_counts(ascending=False)
    int_col_less_than_10 = int_col_stats[int_col_stats <=10]
    df.int_col=df.int_col.apply(lambda x: 'other' if x in int_col_less_than_10 else x)
    df.int_col.nunique()

    brand_stats = df.brand.value_counts(ascending=False)
    brand_less_than_10 = brand_stats[brand_stats <=10]
    df.brand=df.brand.apply(lambda x: 'other' if x in brand_less_than_10 else x)
    df.brand.nunique()

    model_stats = df.model.value_counts(ascending=False)
    model_less_than_10 = model_stats[model_stats <=10]
    df.model=df.model.apply(lambda x: 'other' if x in model_less_than_10 else x)
    df.model.nunique()

    return df