import streamlit as st
import joblib
import numpy as np

# Load the trained model
model_path = "Logistic_Norm_Cancer_predictor"
try:
    model = joblib.load(model_path)
except Exception as e:
    st.error(f"Error loading the model: {e}")
    st.stop()

# Streamlit app
def main():
    st.title("Breast Cancer Prediction")

    st.write("Enter the following attributes:")

    # Define attribute labels
    attributes = [
        "radius_mean", "texture_mean", "perimeter_mean", "area_mean", "smoothness_mean",
        "compactness_mean", "concavity_mean", "concave points_mean", "symmetry_mean", "fractal_dimension_mean",
        "radius_se", "texture_se", "perimeter_se", "area_se", "smoothness_se",
        "compactness_se", "concavity_se", "concave points_se", "symmetry_se", "fractal_dimension_se",
        "radius_worst", "texture_worst", "perimeter_worst", "area_worst", "smoothness_worst",
        "compactness_worst", "concavity_worst", "concave points_worst", "symmetry_worst", "fractal_dimension_worst"
    ]

    # Create input fields for attributes
    inputs = []
    for attribute in attributes:
        value = st.text_input(attribute, value="", key=attribute)
        inputs.append(value)

    # Predict button
    if st.button("Predict"):
        try:
            # Convert inputs to float and reshape for model
            inputs_array = np.array([float(value) for value in inputs]).reshape(1, -1)

            # Get prediction from the model
            prediction = model.predict(inputs_array)

            # Map prediction to meaningful labels
            diagnosis = "Malignant (Cancer Detected)" if prediction[0] == 1 else "Benign (No Cancer Detected)"

            # Display result
            # st.success(prediction[0])
            st.success(diagnosis)
        except ValueError:
            st.error("Please enter valid numeric values for all fields.")

if __name__ == "__main__":
    main()
