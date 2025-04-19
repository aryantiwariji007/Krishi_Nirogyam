import streamlit as st
import tensorflow as tf 
import numpy as np
import gdown
import os


MODEL_PATH = "trained_plant_disease_model.keras"
FILE_ID = "https://drive.google.com/file/d/1oJLzUAjcQXvB4byMLHHpnlAWrIUfIE3d/view?usp=sharing"  # Replace this with your real file ID

@st.cache_resource
def load_model():
    if not os.path.exists(MODEL_PATH):
        url = f"https://drive.google.com/uc?id={https://drive.google.com/file/d/1oJLzUAjcQXvB4byMLHHpnlAWrIUfIE3d/view?usp=sharing}"
        gdown.download(url, MODEL_PATH, quiet=False)
    model = tf.keras.models.load_model(MODEL_PATH)
    return model

def model_prediction(test_image):
    cnn = load_model()
    image = tf.keras.preprocessing.image.load_img(test_image, target_size=(128, 128))
    input_arr = tf.keras.preprocessing.image.img_to_array(image)
    input_arr = np.array([input_arr])  # Convert single image to a batch
    prediction = cnn.predict(input_arr)
    result_index = np.argmax(prediction)
    return result_index

st.sidebar.title('Dashboard')
app_mode = st.sidebar.selectbox("Select Page", ["Home", "About", "Disease Recognition"])
#HOME PAGE
if app_mode == "Home":
    st.header("Plant Disease Recognition System")
    st.image(r"Home.jpeg")  # Replace local path with a public image URL or host your image on GitHub

    st.markdown("""  
    Welcome to the Plant Disease Recognition System! 🌿🔍

    Our mission is to help in identifying plant diseases efficiently. Upload an image of a plant, and our system will analyze it to detect any signs of diseases.

    ### How It Works
    1. **Upload Image:** Go to the **Disease Recognition** page.
    2. **Analysis:** We process the image using advanced deep learning models.
    3. **Results:** View predictions instantly.

    ### Why Choose Us?
    - **Accuracy**
    - **User-Friendly**
    - **Fast and Efficient**

    ### Get Started
    Click on the **Disease Recognition** page in the sidebar to begin!
    """)

# -------------------- About Page --------------------
elif app_mode == "About":
    st.header("About")
    st.markdown("""
    #### About Dataset
    
    This dataset consists of about 87K RGB images of healthy and diseased crop leaves categorized into 38 classes.
    
    #### Content
    - Train Set: 70K+
    - Validation Set: 17K+
    - Test Set: 33 images
    """)

# -------------------- Disease Recognition Page --------------------
elif app_mode == "Disease Recognition":
    st.header("Disease Recognition")
    test_image = st.file_uploader("Choose an Image:", type=["jpg", "jpeg", "png"])

    if st.button("Show Image") and test_image is not None:
        st.image(test_image, width=300)

    if st.button("Predict") and test_image is not None:
        st.balloons()
        st.write("🔍 Our Prediction:")
        result_index = model_prediction(test_image)

        class_name = [
            'Apple___Apple_scab', 'Apple___Black_rot', 'Apple___Cedar_apple_rust', 'Apple___healthy',
            'Blueberry___healthy', 'Cherry_(including_sour)___Powdery_mildew', 'Cherry_(including_sour)___healthy',
            'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot', 'Corn_(maize)___Common_rust_',
            'Corn_(maize)___Northern_Leaf_Blight', 'Corn_(maize)___healthy', 'Grape___Black_rot',
            'Grape___Esca_(Black_Measles)', 'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)', 'Grape___healthy',
            'Orange___Haunglongbing_(Citrus_greening)', 'Peach___Bacterial_spot', 'Peach___healthy',
            'Pepper,_bell___Bacterial_spot', 'Pepper,_bell___healthy', 'Potato___Early_blight',
            'Potato___Late_blight', 'Potato___healthy', 'Raspberry___healthy', 'Soybean___healthy',
            'Squash___Powdery_mildew', 'Strawberry___Leaf_scorch', 'Strawberry___healthy',
            'Tomato___Bacterial_spot', 'Tomato___Early_blight', 'Tomato___Late_blight', 'Tomato___Leaf_Mold',
            'Tomato___Septoria_leaf_spot', 'Tomato___Spider_mites Two-spotted_spider_mite', 'Tomato___Target_Spot',
            'Tomato___Tomato_Yellow_Leaf_Curl_Virus', 'Tomato___Tomato_mosaic_virus', 'Tomato___healthy'
        ]
        st.success(f"Model is predicting: **{class_name[result_index]}**")
