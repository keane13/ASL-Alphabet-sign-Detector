#Library imports
import numpy as np
import streamlit as st
from PIL import Image
from keras.models import load_model
import numpy as np
from keras.preprocessing import image
import string
import io
import os
#from utils import image_prep


#Loading the Model
model = load_model("C:/Users/USER/Desktop/Desktop/macth2bangalore/ASL/models.h5")

#Name of Classes
#classes= ['A', 'B','C','D','E','F','G','H','I','J','K','L','M','N','O','P','Q','R','S','T','U','V','W','X','Y','Z']


label_dic = {i:string.ascii_uppercase[i] for i in range(26)}
label_dic.pop(9)
label_dic.pop(25)
#Setting Title of App
st.title("Alfabet Hand Sign Detection")
st.markdown("Upload an image")
TARGET_SIZE=225

#Uploading the dog image
img = st.file_uploader("Choose an image...", type="jpg")
submit = st.button('Predict')
#On predict button click
if submit:


    if img is not None:

        # Convert the file to an opencv image.
        #img = image_file.read()
        #ximg = Image.open(img)
        images=img.read()
        images1 = image.load_img(images)
        #resized_im = images.resize((150, 150))
        im_array = np.asarray(images1)
        im_array = im_array*(1/225)
      
        #x = image.img_to_array(img)
        #x /= 255
        #x = np.expand_dims(x, axis=0)
        #prepped_img = image_prep.imageprepare(img)
         #classes = model.predict(x, batch_size=1)
        prediction = np.argmax(model.predict(im_array))
        alphabet = label_dic[prediction]
        st.write(f'The sign was  {alphabet}')
          
       

        # Displaying the image
     
        #Make Prediction
       # Y_pred = model.predict(opencv_image)
       # result = CLASS_NAMES[np.argmax(Y_pred)]
        #st.title(str("This is "+result.split('-')[0]+ " leaf with " + result.split('-')[1]))
