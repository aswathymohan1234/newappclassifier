

import streamlit as st
import tensorflow as tf
import cv2
from PIL import Image,ImageOps
import numpy as np



model =  tf.keras.models.load_model('08_classifier_model.hdf5')

st.write("""Custom Vision""")
file =[]
file = st.file_uploader('Please Upload the Image',type=['jpg','png'],accept_multiple_files=True)

def import_and_predict(image,model):
    size = (224,224)
    image = ImageOps.fit(image,size,Image.ANTIALIAS)
    im = np.asarray(image)
    im = im[np.newaxis,...]
    prediction = model.predict(im)  
    return prediction

prediction=[]
if file is None:
    st.text('Please Upload an Image')
else:
    for im in file:
        image = Image.open(im)
        st.image(image,use_column_width=True)
        prediction = import_and_predict(image,model)
        
        # string = 'Prediction = '+ str(prediction) 
        # st.success(string)
        # #class_nm = ['Electric_Meter','Electrical_Panel','Electrical_Bill','Home_Across_Street','Main_Breaker']
        # class_nm = ['Attic','Rafter_Size','Rafter_Space']
        # string = 'This image is likely to be '+class_nm[np.argmax(prediction)]            
        # st.success(string)
        # string ='Class'+ str(np.argmax(prediction))
        # st.success(string)
        
        if np.argmax(prediction) == 0:
               st.write("Electric Meter:", prediction[0][0]*100)
        elif np.argmax(prediction) == 1:
            st.write("Electrical Panel:", prediction[0][1]*100)
        elif np.argmax(prediction) == 2:
            st.write("Electric Bill:", prediction[0][2]*100)
        elif np.argmax(prediction) == 3:
            st.write("Home:",prediction[0][3]*100)  
        elif np.argmax(prediction) == 4:
            st.write("Main Breaker:",prediction[0][4]*100)  
        elif np.argmax(prediction) == 5:
            st.write("Attic:",prediction[0][5]*100)  
        elif np.argmax(prediction) == 6:
            st.write("Rafter Size:", prediction[0][6]*100)    
        else:
            st.write("Rafter Space:", prediction[0][7]*100)
