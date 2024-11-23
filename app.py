"""
@author: Atithi Shrestha 
"""

import numpy as np
import pickle
import streamlit as st

# Load the pre-trained model. Make ABSOLUTELY sure this path is correct!
try:
    loaded_model = pickle.load(open('model/trained_model.pkl', 'rb'))
except FileNotFoundError:
    st.error("Error: Could not find the model file. Please make sure 'model/trained_model.pkl' exists in your project directory.")
    st.stop()  # Stop execution if model not found.

# Preprocessing function
def preprocess_input(input_data):
    sex_mapping = {"Male": 1, "Female": 0, "Other": 2}  # Ensure this matches training
    processed_data = list(input_data)
    processed_data[1] = sex_mapping[processed_data[1]]
    # Convert all other string inputs to float
    for i in [0, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]:
        processed_data[i] = float(processed_data[i])
    return np.array(processed_data, dtype=np.float64)

# Prediction function
def Heart_disease_Prediction(input_data):
    input_data_array = preprocess_input(input_data)
    input_data_reshaped = input_data_array.reshape(1, -1)
    try:
        result = loaded_model.predict(input_data_reshaped)[0]  # added [0] to get the actual result
        return result  # Return 0 or 1 instead of string
    except Exception as e:
        return f"Prediction error: {e}"

def main():
    st.markdown("<h1 style='text-align: center; color: red;'>Heart Disease Prediction Application</h1>", unsafe_allow_html=True)
    st.markdown("***")

    # Input fields
    age = st.text_input("Age of person: ")
    sex = st.selectbox("Sex:", options=["Male", "Female", "Other"])
    cp = st.text_input("Chest pain type (0-3): ")
    restbps = st.text_input("Resting BP: ")
    chol = st.text_input("Serum Cholesterol (mg/dl): ")
    fbs = st.text_input("Fasting blood sugar > 120 mg/dl (0 or 1): ")
    restecg = st.text_input("Resting electrocardiographic results (0-2): ")
    thalach = st.text_input("Maximum heart rate achieved: ")
    exang = st.text_input("Exercise induced angina (0 or 1): ")
    oldpeak = st.text_input("Oldpeak: ")
    slope = st.text_input("Slope of the peak exercise ST segment (1-3): ")
    ca = st.text_input("Number of major vessels (0-3): ")
    thal = st.text_input("Thal (0=normal, 1=fixed, 2=reversible): ")

    predict = ""

    if st.button('Diagnosis Test Result', key='diagnosis_button'):
        try:
            input_list = [age, sex, cp, restbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal]
            if any(x == "" for x in input_list):
                st.error("Please fill in all fields.")
            else:
                prediction_result = Heart_disease_Prediction(input_list)
                if isinstance(prediction_result, str):
                    st.error(prediction_result)
                elif prediction_result == 1:
                    st.success("Person has a Heart Disease.")
                else:
                    st.success("Person doesn't have Heart Disease.")
        except ValueError as e:
            st.error(f"Invalid input: Please enter numeric values where applicable. {e}")
        except Exception as e:
            st.exception(e)

    st.markdown("***")
    st.markdown("""
    Sample data to fill: 

        52 1 2 172 199 1 1 162 0 0.5 2 0 3	  => Person has Heart Disease 
    """)
    st.markdown("***")
    st.markdown("""
    About the data to be filled (all data is in numeric form without units): 

        1. Age (in numbers)
        2. Sex 
        3. Chest pain type (4 values: 0-3)
        4. Resting blood pressure (numeric only)
        5. Serum Cholesterol in mg/dl
        6. Fasting blood sugar > 120 mg/dl (0 or 1)
        7. Resting electrocardiographic results (values: 0, 1, 2)
        8. Maximum heart rate achieved
        9. Exercise-induced angina (0 or 1)
        10. Oldpeak = ST depression induced by exercise relative to rest
        11. The slope of the peak exercise ST segment
        12. Number of major vessels (0-3) colored by fluoroscopy
        13. Thal: 0 = normal; 1 = fixed defect; 2 = reversible defect
        
        Output: Either Heart Disease is present (1) or not (0)
    """)
    st.text("\n\n")

    st.write(" \n\n\n\n")
    st.markdown("******")

    st.write("Contributor : [Atithi Shrestha](https://github.com/atithishrestha123) \n [LinkedIn](https://www.linkedin.com/in/atithi-shrestha-835a70257/)")

if __name__ == '__main__':
    main()
