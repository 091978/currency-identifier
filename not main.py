import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import cv2
import numpy as np
import pyttsx3
from keras.models import Model, load_model
from gtts import gTTS
from keras.applications.xception import Xception, preprocess_input
import playsound
from IPython.display import Audio

text_speech = pyttsx3.init()
model_final = tf.keras.models.load_model('finalmodel.h5')
cam = cv2.VideoCapture(1)
cv2.namedWindow("Currency Identification")
img_counter = 0
while True:
    ret, frame = cam.read()
    if not ret:
        print("failed to grab frame")
        break
    cv2.imshow("Currency Identification", frame)
    k = cv2.waitKey(1)
    if k%256 == 27:
        # ESC pressed
        print("Escape hit, closing...")
        break
    if k%256 == 32:
        # SPACE pressed
        img_name = "currency.png".format(img_counter)
        cv2.imwrite(img_name, frame)
        print("{} written!".format(img_name))
        img_counter += 1
        test_image = tf.keras.preprocessing.image.load_img(img_name,target_size = (255,255))
        test_image = tf.keras.preprocessing.image.img_to_array(test_image)
        test_image = np.expand_dims(test_image, axis = 0)
        prediction = model_final.predict(test_image)
        idx = np.argmax(prediction, axis = 1)
        if idx==0:
            digit="1 peso coin"
        elif idx==1:
            digit="10 pesos coin"
        elif idx==2:
            digit="100 pesos bill"
        elif idx==3:
            digit="1000 pesos bill"
        elif idx==4:
            digit="20 pesos bill"
        elif idx==5:
            digit="20 pesos coin"
        elif idx==6:
            digit="200 pesos bill"
        elif idx==7:
            digit="5 pesos coin"
        elif idx==8:
            digit="50 pesos bill"
        elif idx==9:
            digit="500 pesos bill"
        confidence = prediction[0, idx] * 100
        predicted_name=f'{digit} with score {confidence[0]}'
        position = (100-20,50-10)
        print(predicted_name)
        if 96 <= confidence <= 100:
            cv2.putText(
                frame, #numpy array on which text is written
                predicted_name, #text
                position, #position at which writing has to start
                cv2.FONT_HERSHEY_SIMPLEX, #font family
                0.8, #font size
                (0, 0, 255), #font color
                3)
        text_speech.say(digit)
        text_speech.runAndWait()
                
   

 

# Release the cam and close the window
cam.release()
cv2.destroyAllWindows()
