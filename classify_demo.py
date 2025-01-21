import streamlit as st
import streamlit.components.v1 as components
import numpy as np
import joblib

model = joblib.load('model.pkl')
scaler = joblib.load('normalization.pkl')

#Interface
st.markdown('## Iris Species Prediction')
sepal_length = st.slider('sepal length (cm)',min_value=scaler.data_min_[0], max_value=scaler.data_max_[0])
sepal_width = st.slider('sepal width (cm)',min_value=scaler.data_min_[1], max_value=scaler.data_max_[1])
petal_length = st.slider('petal length (cm)',min_value=scaler.data_min_[2], max_value=scaler.data_max_[2])
petal_width = st.slider('petal width (cm)',min_value=scaler.data_min_[3], max_value=scaler.data_max_[3])

#Predict button
if st.button('Predict'):
    
    X = np.array([sepal_length, sepal_width, petal_length, petal_width]).reshape(1, -1)
    X_norm = scaler.transform(X)
    # if any(X <= 0):
    #     st.markdown('### Inputs must be greater than 0')
    # else:
    prediction = model.predict(X_norm)
    st.markdown(f'### Prediction is {prediction}')
    components.iframe(f"https://www.google.com/search?igu=1&ei=&q={prediction}", height=1000)
