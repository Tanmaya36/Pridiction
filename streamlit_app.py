import streamlit as st
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import load_model

# Load trained model (ensure it's saved as 'model.h5')
model = load_model('model.h5')

# Dummy scaler for demonstration (replace with your actual scaler)
scaler = MinMaxScaler(feature_range=(0, 1))
scaler.fit(np.array(range(1000)).reshape(-1, 1))

# Function to predict the next number
def predict_next_sequence(custom_sequence):
    custom_sequence = np.array(custom_sequence).reshape(-1, 1)
        custom_sequence = scaler.transform(custom_sequence)
            custom_sequence = np.reshape(custom_sequence, (1, custom_sequence.shape[0], 1))
                predicted = model.predict(custom_sequence)
                    predicted = scaler.inverse_transform(predicted)
                        return int(predicted[0][0])

                        # Streamlit App Interface
                        st.title("AI-based PRNG (LCG) Sequence Predictor")
                        st.write("Enter 10 numbers to predict the next one.")

                        # Input field for sequence
                        sequence_input = st.text_input("Enter 10 comma-separated numbers:", "123,456,789,101,112,131,415,161,718,192")

                        # Predict Button
                        if st.button("Predict"):
                            try:
                                    custom_sequence = list(map(int, sequence_input.split(",")))
                                            if len(custom_sequence) != 10:
                                                        st.error("Please enter exactly 10 numbers.")
                                                                else:
                                                                            predicted_next = predict_next_sequence(custom_sequence)
                                                                                        st.success(f"Predicted Next Number: {predicted_next}")
                                                                                            except ValueError:
                                                                                                    st.error("Invalid input. Please enter numbers only.")
