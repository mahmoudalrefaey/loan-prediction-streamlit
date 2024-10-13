import streamlit as st
from sklearn.preprocessing import LabelEncoder
import pickle
import pandas as pd
import numpy as np


def get_data():
    """
    Read and clean the data.
    """
    data = pd.read_csv("datasets/Training Data.csv")
    conversion_rate = 0.011909906
    data['Income'] = data['Income'] * conversion_rate
    data = data.drop(['Id', 'CITY', 'STATE', 'CURRENT_HOUSE_YRS'], axis=1)
    return data


def add_sidebar():
    """
    Create the sidebar with input fields.
    """
    st.sidebar.header("Loan holder information")
    data = get_data()
    label_encoder = LabelEncoder()

    slider_labels = [
        ("Income ($)", "Income"),
        ("Age", "Age"),
        ("Experience (Years)", "Experience"),
        ("Current Job (Years)", "CURRENT_JOB_YRS"),
    ]
    dropdown_labels = [
        ("Marital Status", "Married/Single"),
        ("House Ownership", "House_Ownership"),
        ("Car Ownership", "Car_Ownership"),
        ("Profession", "Profession"),
    ]

    input_dict = {}
    encoders = {}

    for label, key in slider_labels:
        input_dict[key] = st.sidebar.slider(
            label,
            min_value=int(data[key].min()),
            max_value=int(data[key].max()),
            value=int(data[key].mean())
        )

    for label, key in dropdown_labels:
        unique_values = data[key].unique()
        label_encoder.fit(unique_values)
        encoders[key] = label_encoder
        selected_option = st.sidebar.selectbox(
            label,
            options=unique_values
        )
        encoded_option = label_encoder.transform([selected_option])[0]
        input_dict[key] = encoded_option

    return input_dict


def add_predictions(input_data):
    """
    Make predictions using the model.
    """
    model = pickle.load(open("model/model.pkl", 'rb'))
    scaler = pickle.load(open("model/scaler.pkl", 'rb'))

    input_array = np.array(list(input_data.values())).reshape(1, -1)
    input_array_scaled = scaler.transform(input_array)
    prediction = model.predict(input_array_scaled)

    st.subheader('Risk Flag Prediction')

    if prediction[0] == 0:
        st.markdown('<p style="color:#38B000;">Eligable </p>', unsafe_allow_html=True)
    elif prediction[0] == 1:
        st.markdown('<p style="color:#ff4b4b;">Risk </p>', unsafe_allow_html=True)

    st.write("Probability of being Eligable: ", model.predict_proba(input_array_scaled)[0][0])
    st.write("Probability of being Risky: ", model.predict_proba(input_array_scaled)[0][1])


def main():
    """
    Main function.
    """
    st.set_page_config(
        page_title="LoanWise",
        page_icon='🏦',
        layout="wide",
        initial_sidebar_state="expanded"
    )

    get_data()
    input_data = add_sidebar()

    with st.container():
        st.title("Smarter Loan Risk Prediction for Confident Banking Decisions")
        st.write("Unlock the power of smart lending with LoanWise! Our cutting-edge tool Powered by AI  & uses advanced analytics to predict loan eligibility at a glance."
                 "\nGet insights into your financial future and make confident borrowing decisions with ease!")

    col1, col2 = st.columns([4, 1])

    with col1:
        add_predictions(input_data)
        if input_data['House_Ownership'] == np.int32(0):
            st.markdown('<p style="color:#ff4b4b;">Ask about their living situation. </p>', unsafe_allow_html=True)
        if input_data["CURRENT_JOB_YRS"] > input_data["Experience"]:
            st.markdown('<p style="color:#ff4b4b;">Ask about their work experience in details. </p>', unsafe_allow_html=True)

    st.write("⚠️ This app can aid in predicting load risks, but it should not replace the expertise of a qualified professional.")


if __name__ == '__main__':
    main()