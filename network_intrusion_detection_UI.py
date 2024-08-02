import streamlit as st
import tensorflow as tf
import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder
import joblib
from config import _path_to_scaler, _save_model_dir, _ohe_categories, _path_to_audio


def load_model(path):
    '''Load the saved TensorFlow model.'''
    model = tf.keras.models.load_model(path)
    return model


def preprocess_data(test_features):
    '''Preprocess input data to match the model’s expected input format.'''
    columns = ['duration', 'protocol_type', 'service', 'flag', 'src_bytes',
               'dst_bytes', 'land', 'wrong_fragment', 'urgent', 'hot',
               'num_failed_logins', 'logged_in', 'num_compromised', 'root_shell',
               'su_attempted', 'num_root', 'num_file_creations', 'num_shells',
               'num_access_files', 'num_outbound_cmds', 'is_host_login',
               'is_guest_login', 'count', 'srv_count', 'serror_rate',
               'srv_serror_rate', 'rerror_rate', 'srv_rerror_rate', 'same_srv_rate',
               'diff_srv_rate', 'srv_diff_host_rate', 'dst_host_count',
               'dst_host_srv_count', 'dst_host_same_srv_rate',
               'dst_host_diff_srv_rate', 'dst_host_same_src_port_rate',
               'dst_host_srv_diff_host_rate', 'dst_host_serror_rate',
               'dst_host_srv_serror_rate', 'dst_host_rerror_rate',
               'dst_host_srv_rerror_rate']

    # Create DataFrame with string column names
    test_features_df = pd.DataFrame([test_features], columns=columns)
    categorical_columns = ['protocol_type', 'service', 'flag']
    ohe2 = OneHotEncoder(categories=_ohe_categories, sparse=False)
    ohe_features = ohe2.fit_transform(test_features_df[categorical_columns])
    dummies = pd.DataFrame(
        ohe_features, index=test_features_df.index, dtype=int)
    encoded_columns = ohe2.get_feature_names_out(categorical_columns)
    dummies.columns = encoded_columns
    test_dataframe = pd.concat([test_features_df.drop(
        categorical_columns, axis=1), dummies], axis=1)
    # Convert column names to strings
    test_dataframe.columns = test_dataframe.columns.astype(str)

    scaler = joblib.load(_path_to_scaler)
    scaled_features = scaler.transform(test_dataframe)
    test_dataframe_tensor = tf.convert_to_tensor(
        scaled_features, dtype='float32')
    return test_dataframe_tensor


def make_prediction(model, input_data):
    '''Make prediction and return the label with confidence.'''
    pred = model.predict(input_data)
    pred = np.round(pred[0])[0].astype('int32')
    class_names = ['attack', 'normal']
    pred_label = class_names[pred]
    return pred_label


def main(model_path):
    '''Streamlit main function for user interface.'''
    st.title('Network Intrusion Detection System')
    st.write('Enter the network packet features as a comma-separated list.')

    st.markdown("""
        ### Instructions:
        1. Enter the features in the text box below, separated by commas.
        2. Click the **Predict** button to classify the network packet.
    """)

    feature_list = st.text_area(
        'Features List', help="Enter features like 'duration, protocol_type, service, ...'")

    if st.button('Predict'):
        if feature_list:
            parts = feature_list.split(',')
            converted_parts = []
            for part in parts:
                part = part.strip()
                try:
                    converted_part = float(part)
                except ValueError:
                    converted_part = part
                converted_parts.append(converted_part)
            model = load_model(model_path)
            input_data = preprocess_data(converted_parts)
            prediction = make_prediction(model, input_data)
            st.write(f'Prediction: **{prediction}**')
            # st.audio()
            if prediction == 'attack':
                st.markdown("""
                    <p style="color: red; font-size: 20px;">
                        ⚠️ **Warning:** The system detected an attack.
                    </p>
                    <ul>
                        <li>Check the system logs for unusual activity.</li>
                        <li>Update your firewall and antivirus software.</li>
                        <li>Conduct a security audit to ensure no vulnerabilities exist.</li>
                    </ul>
                """, unsafe_allow_html=True)

                # Provide the path to your local audio file in Colab
                st.audio(
                    _path_to_audio, format='audio/wav', autoplay=True, end_time=9)

            else:
                st.markdown("""
                    <p style="color: green; font-size: 20px;">
                        ✅ <b>Safe:</b> The network packet appears normal.
                    </p>
                    <ul>
                        <li>Regularly monitor network traffic.</li>
                        <li>Ensure that all security patches are up to date.</li>
                        <li>Follow best practices for network security.</li>
                    </ul>
                """, unsafe_allow_html=True)

        else:
            st.warning('Please enter a list of features to predict.')

    st.sidebar.title("About This App")
    st.sidebar.info("""
        This app uses a Deep Neural Network (DNN) to classify network packets as either 'attack' or 'normal'.
        Ensure to input the correct feature list for accurate predictions.
        
        Developed as part of a final year project from the department of Computer Engineering
        Federal University Oye-Ekiti.
    """)


if __name__ == '__main__':
    main(_save_model_dir + '/DNN_Model')
