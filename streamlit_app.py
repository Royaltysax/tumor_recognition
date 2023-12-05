# importing the libraries and dependencies needed for creating the UI and supporting the deep learning models used in the project
import streamlit as st  
import tensorflow as tf
import random
from PIL import Image, ImageOps
import numpy as np
from tensorflow.keras.models import load_model
import numpy as np

from tensorflow.keras.preprocessing.image import load_img, img_to_array

from PIL import Image
# hide deprecation warnings which directly don't affect the working of the application
import warnings
warnings.filterwarnings("ignore")

# set some pre-defined configurations for the page, such as the page title, logo-icon, page loading state (whether the page is loaded automatically 
# or you need to perform some action for loading)
st.set_page_config(
    page_title="Brain Tumor Recognition and Classification",
    initial_sidebar_state = 'auto'
)

st.caption("Upload an image. ")

st.caption("The application will infer the one label out of 4 labels: 'Glioma', 'Meningioma', 'Healthy', 'Pituitary'.")

st.caption("Warning: Do not click Recognize button before uploading image. It will result in error.")


model = load_model("/content/tumor_efficientnet(99.22).h5")

# Define the class names

class_names = ['Glioma', 'Meningioma', 'Healthy', 'Pituitary']

# Fxn

@st.cache_data

def load_image(image_file):

        img = Image.open(image_file)

        return img


imgpath = st.file_uploader("Choose a file", type =['png', 'jpeg', 'jpg'])

if imgpath is not None:

    img = load_image(imgpath )

    st.image(img, width=224)



def predict_label(image2):
    imgLoaded = load_img(image2, target_size=(224, 224))

    # Convert the image to an array
    img_array = img_to_array(imgLoaded)    #print(img_array)

    #print(img_array.shape)
    img_array = np.reshape(img_array, (1, 224, 224, 3))

    # Get the model predictions
    predictions = model.predict(img_array)

    #print("predictions:", predictions)
    # Get the class index with the highest predicted probability
    class_index = np.argmax(predictions[0])

    # Get the predicted class label
    predicted_label = class_names[class_index]

    return predicted_label

if st.button('Recognise'):
    predicted_label = predict_label(imgpath)

    if predicted_label == 'Healthy':
      st.write("The MRI image shows sign of an '{}' brain.".format(predicted_label))
    else:
      st.write("The MRI image shows sign of a '{}' brain tumor.".format(predicted_label))
