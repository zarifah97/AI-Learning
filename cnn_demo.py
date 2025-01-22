import streamlit as st
from tensorflow.keras.models import load_model, model_from_json
from tensorflow.keras.preprocessing import image
from tensorflow.keras.initializers import glorot_uniform
import numpy as np
from PIL import Image
import io

# Load the pre-trained model (replace 'model.h5' with your model path)
#Reading the model from JSON file
with open('fruits-model.json', 'r') as json_file:
    loaded_model_json = json_file.read()

loaded_model = model_from_json(loaded_model_json)
loaded_model.summary()

loaded_model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

loaded_model.load_weights("fruits-model.weights.h5")
print("Loaded model from disk")

loaded_model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

# Class names corresponding to your model output (example for a binary classification model)
class_names = ['Apple Braeburn', 'Apple Crimson Snow', 'Apple Golden 1', 'Apple Golden 2',
 'Apple Golden 3', 'Apple Granny Smith', 'Apple Pink Lady', 'Apple Red 1',
 'Apple Red 2', 'Apple Red 3', 'Apple Red Delicious', 'Apple Red Yellow 1',
 'Apple Red Yellow 2', 'Apricot', 'Avocado', 'Avocado ripe', 'Banana',
 'Banana Lady Finger', 'Banana Red', 'Beetroot', 'Blueberry' ,'Cactus fruit',
 'Cantaloupe 1' ,'Cantaloupe 2' ,'Carambula', 'Cauliflower', 'Cherry 1',
 'Cherry 2' ,'Cherry Rainier' ,'Cherry Wax Black' ,'Cherry Wax Red',
 'Cherry Wax Yellow' ,'Chestnut' ,'Clementine' ,'Cocos', 'Corn' ,'Corn Husk',
 'Cucumber Ripe', 'Cucumber Ripe 2', 'Dates', 'Eggplant', 'Fig' ,'Ginger Root',
 'Granadilla', 'Grape Blue' ,'Grape Pink' ,'Grape White' ,'Grape White 2',
 'Grape White 3' ,'Grape White 4', 'Grapefruit Pink', 'Grapefruit White',
 'Guava', 'Hazelnut', 'Huckleberry' ,'Kaki','Kiwi' ,'Kohlrabi' ,'Kumquats',
 'Lemon' ,'Lemon Meyer' ,'Limes' ,'Lychee' ,'Mandarine' ,'Mango', 'Mango Red',
 'Mangostan' ,'Maracuja' ,'Melon Piel de Sapo', 'Mulberry', 'Nectarine',
 'Nectarine Flat', 'Nut Forest' ,'Nut Pecan', 'Onion Red', 'Onion Red Peeled',
 'Onion White' ,'Orange' ,'Papaya' ,'Passion Fruit' ,'Peach' ,'Peach 2',
 'Peach Flat' ,'Pear', 'Pear 2', 'Pear Abate', 'Pear Forelle' ,'Pear Kaiser',
 'Pear Monster' ,'Pear Red', 'Pear Stone' ,'Pear Williams', 'Pepino',
 'Pepper Green', 'Pepper Orange' ,'Pepper Red' ,'Pepper Yellow' ,'Physalis',
 'Physalis with Husk', 'Pineapple', 'Pineapple Mini' ,'Pitahaya Red', 'Plum',
 'Plum 2' ,'Plum 3' ,'Pomegranate' ,'Pomelo Sweetie' ,'Potato Red',
 'Potato Red Washed' ,'Potato Sweet', 'Potato White' ,'Quince' ,'Rambutan',
 'Raspberry' ,'Redcurrant' ,'Salak' ,'Strawberry' ,'Strawberry Wedge',
 'Tamarillo', 'Tangelo' ,'Tomato 1' ,'Tomato 2', 'Tomato 3', 'Tomato 4',
 'Tomato Cherry Red' ,'Tomato Heart', 'Tomato Maroon', 'Tomato Yellow',
 'Tomato not Ripened', 'Walnut', 'Watermelon']  # Replace with your model's classes

# Function to make a prediction
def predict_image(img):
    # Resize image to match the input shape expected by the model (e.g., 224x224x3)
    img = img.resize((32, 32))
    
    # Convert image to array
    img_array = np.array(img)
    
    # If the model expects a different input shape, add necessary processing steps
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    img_array = img_array / 255.0  # Normalize if required (e.g., if model was trained on normalized data)
    
    # Predict the class
    prediction = loaded_model.predict(img_array)
    
    # Get the predicted class
    predicted_class = class_names[np.argmax(prediction)]
    return predicted_class, prediction

# Streamlit app
st.title("Image Classification with Pre-trained TensorFlow Model")

# Upload image
uploaded_image = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_image is not None:
    # Open image with PIL
    img = Image.open(uploaded_image)
    
    # Display the image
    st.image(img, caption="Uploaded Image.", use_container_width =True)
    
    # Predict the image
    if st.button('Predict'):
        label, prediction = predict_image(img)
        st.write(f"Predicted Class: {label}")
        st.write(f"Prediction Probability: {np.max(prediction):.2f}")
