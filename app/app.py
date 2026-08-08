import streamlit as st
import pandas as pd
import joblib


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Used Car Price Appraiser",
    page_icon="🚗",
    layout="wide"
)


# ============================================================
# LOAD MODEL AND ENCODER
# ============================================================

@st.cache_resource
def load_artifacts():

    model = joblib.load(
        "models/used_car_price_model.pkl"
    )

    encoder = joblib.load(
        "models/target_encoder.pkl"
    )

    return model, encoder


model, encoder = load_artifacts()


# ============================================================
# CONSTANTS
# ============================================================

categorical_features = [
    "brand",
    "model",
    "fuel_type",
    "engine",
    "transmission",
    "ext_col",
    "int_col",
    "clean_title"
]

numerical_features = [
    "model_year",
    "mileage",
    "accident",
    "vehicle_age",
    "mileage_per_year"
]


# ============================================================
# PAGE HEADER
# ============================================================

st.title("🚗 Used Car Price Appraiser")

st.markdown(
    """
    Enter the vehicle specifications below to estimate
    its expected used-car market price.
    """
)

# ============================================================
# LOADING DATASET
# ============================================================

st.divider()

@st.cache_data
def load_dataset():

    return pd.read_csv(
        "data/used_cars.csv"
    )


df = load_dataset()

# ============================================================
# VEHICLE INFORMATION
# ============================================================

st.subheader("🚘 Vehicle Information")

col1, col2, col3 = st.columns(3)


with col1:

    brand = st.selectbox(
    "Brand",
    sorted(df["brand"].dropna().unique())
    )

    model_name = st.selectbox(
        "Model",
        sorted(df["model_name"].dropna().unique())
    )

    model_year = st.number_input(
        "Model Year",
        min_value=1990,
        max_value=2026,
        value=2020,
        step=1
    )


with col2:

    mileage = st.number_input(
        "Mileage",
        min_value=0,
        max_value=500000,
        value=35000,
        step=1000
    )

    fuel_type = st.selectbox(
    "Fuel Type",
    sorted(df["fuel_type"].dropna().unique())
    )

    engine = st.selectbox(
        "Engine",
        sorted(df["engine"].dropna().unique())
    )


with col3:

    transmission = st.selectbox(
    "Transmission",
    sorted(df["transmission"].dropna().unique())
    )

    ext_col = st.selectbox(
    "Exterior Color",
    sorted(df["ext_col"].dropna().unique())
    )

    int_col = st.selectbox(
    "Interior Color",
    sorted(df["int_col"].dropna().unique())
    )


# ============================================================
# VEHICLE CONDITION
# ============================================================

st.subheader("🔧 Vehicle Condition")

col4, col5 = st.columns(2)


with col4:

    accident = st.selectbox(
        "Accident",
        options=[0, 1],
        index=0,
        help="0 = No accident, 1 = Accident reported"
    )


with col5:

    clean_title = st.selectbox(
        "Clean Title",
        options=["Yes", "No", "Unknown"],
        index=0
    )


# ============================================================
# ENGINEERED FEATURES
# ============================================================

st.subheader("📊 Derived Vehicle Features")

col6, col7 = st.columns(2)


with col6:

    vehicle_age = st.number_input(
        "Vehicle Age",
        min_value=0,
        max_value=100,
        value=5,
        step=1
    )


with col7:

    mileage_per_year = st.number_input(
        "Mileage Per Year",
        min_value=0.0,
        max_value=500000.0,
        value=20000.0,
        step=1000.0
    )


st.divider()


# ============================================================
# CREATE INPUT DATAFRAME
# ============================================================

def create_input_dataframe():

    input_data = {
        "brand": brand,
        "model": model_name,
        "model_year": model_year,
        "mileage": mileage,
        "fuel_type": fuel_type,
        "engine": engine,
        "transmission": transmission,
        "ext_col": ext_col,
        "int_col": int_col,
        "accident": accident,
        "clean_title": clean_title,
        "vehicle_age": vehicle_age,
        "mileage_per_year": mileage_per_year
    }

    return pd.DataFrame([input_data])


# ============================================================
# PREDICTION FUNCTION
# ============================================================

def predict_price(input_df):

    # Copy dataframe
    data = input_df.copy()


    # --------------------------------------------------------
    # Handle categorical missing values
    # --------------------------------------------------------

    for column in categorical_features:

        data[column] = data[column].fillna(
            "Unknown"
        )


    # --------------------------------------------------------
    # Target Encoding
    # --------------------------------------------------------

    encoded_data = encoder.transform(data)


    # --------------------------------------------------------
    # Model Prediction
    # --------------------------------------------------------

    prediction = model.predict(
        encoded_data
    )

    return prediction[0]


# ============================================================
# PREDICTION BUTTON
# ============================================================

if st.button(
    "💰 Estimate Car Price",
    type="primary",
    use_container_width=True
):

    try:

        # Create dataframe
        input_df = create_input_dataframe()


        # Display input
        st.subheader("📋 Vehicle Information")

        st.dataframe(
            input_df,
            use_container_width=True,
            hide_index=True
        )


        # Predict
        predicted_price = predict_price(
            input_df
        )


        # Display result
        st.success(
            "Price prediction generated successfully!"
        )


        st.metric(
            label="Estimated Vehicle Price",
            value=f"${predicted_price:,.2f}"
        )


        st.info(
            """
            The estimated price is generated by the trained
            machine learning model. It should be considered
            an approximate valuation rather than a guaranteed
            market price.
            """
        )


    except Exception as e:

        st.error(
            f"Prediction failed: {e}"
        )


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.header("🤖 Model Information")

    st.write(
        """
        This application uses a trained regression model
        to estimate used-car prices.
        """
    )

    st.divider()

    st.write("### Input Features")

    st.write(
        """
        • Brand  
        • Model  
        • Model Year  
        • Mileage  
        • Fuel Type  
        • Engine  
        • Transmission  
        • Exterior Color  
        • Interior Color  
        • Accident  
        • Clean Title  
        • Vehicle Age  
        • Mileage Per Year
        """
    )

    st.divider()

    st.write("### ML Pipeline")

    st.write(
        """
        Input Data  
        ↓  
        Target Encoding  
        ↓  
        Regression Model  
        ↓  
        Estimated Price
        """
    )

