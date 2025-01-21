import streamlit as st
import streamlit.components.v1 as components
import numpy as np
import joblib
import warnings
warnings.filterwarnings("ignore")

model = joblib.load('regressor.pkl')
scaler = joblib.load('regress_norm.pkl')
encoder = joblib.load('onehot.pkl')
min_max_cat = joblib.load('min_max_and_categories.pkl')

#Interface
st.markdown('## Sale Price Prediction')
min_values = min_max_cat['Min values']
max_values = min_max_cat['Max values']
categories = min_max_cat['Categories']

# Initialize a list to store user inputs
user_inputs_num = []
user_inputs_year = []

for col, min_val in min_values.items():
    max_val = max_values[col]
    if min_val.dtype == 'int64':
        s=1  # Step size
    else:
        s=1.0
    if col == 'SalePrice':
        continue
    elif col == 'YearBuilt' or col == 'YearRemodAdd':
        user_input = st.slider(
            f'Select a value for {col}',  # Label for the slider
            min_value=min_val,  # Min value from saved data
            max_value=max_val,  # Max value from saved data
            value=(min_val + max_val) // 2,  # Default value (mean of min and max)
            step = s
        )
        user_inputs_year.append(user_input)  # Append the input to the listnum
        st.write(f'Selected {col}: {user_input}')   
    else:
        user_input = st.slider(
            f'Select a value for {col}',  # Label for the slider
            min_value=min_val,  # Min value from saved data
            max_value=max_val,  # Max value from saved data
            value=(min_val + max_val) // 2,  # Default value (mean of min and max)
            step = s
        )
        user_inputs_num.append(user_input)  # Append the input to the list
        st.write(f'Selected {col}: {user_input}')
    

    user_inputs_cat = []
# Loop through the categorical columns and create selectboxes using saved categories
for col, options in categories.items():
    user_input = st.selectbox(
        f'Select a {col}',  # Label for the dropdown
        options  # Categories from saved data
    )
    user_inputs_cat.append(user_input)  # Append the input to the list
    st.write(f'Selected {col}: {user_input}')


#Predict button
if st.button('Predict'):
    user_inputs_num_norm = scaler.transform(np.array(user_inputs_num).reshape(1, -1))
    user_inputs_cat_onehot = encoder.transform(np.array(user_inputs_cat).reshape(1, -1))
    user_inputs_year = np.array(user_inputs_year).reshape(1,-1)
    X = np.concatenate((user_inputs_num_norm, user_inputs_year, user_inputs_cat_onehot), axis=1)
    # if any(X <= 0):
    #     st.markdown('### Inputs must be greater than 0')
    # else:
    prediction = model.predict(X)
    st.markdown(f'### Prediction is {prediction}')
