import os
from tensorflow.keras.saving import load_model
import cv2
import numpy as np
from flask import Flask, render_template, request

app = Flask(__name__)

model= load_model("./trained_model/captcha_solver.h5")
character = "0123456789"

#to predict captcha
def predict_model(filepath):
    img = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)

    if img is not None:
        img = np.reshape(img,(80,190,1)) #incase wrong resolution images inputted
        img = img / 255.0 #Scale image
        
    else:
        print("Not detected");

    res = np.array(model.predict(img[np.newaxis, :, :, np.newaxis])) #np.newaxis=1
    #returns array of size 1*6*10
    result = np.reshape(res, (6, 10)) 
    k_ind = []
    probs = []
    for i in result:
        k_ind.append(np.argmax(i)) #adds the index of the char found in captcha

    capt = '' #string to store predicted captcha
    for k in k_ind:
        capt += character[k] #finds the char corresponding to the index
    return capt

@app.route("/", methods=['GET'])
@app.route("/home", methods=['GET'])
def home():
    return render_template('home.html')

@app.route("/predict", methods = ['GET','POST'])
def predict():
    
    if request.method == 'POST':
        
        file = request.files['file']
        filename = file.filename
        file_path = os.path.join(r'./static/img_captcha', filename)          
        file.save(file_path)
        print(filename)
        result = predict_model(file_path)
        print(result)
        
    return render_template('predict.html', result = result, user_image = filename)   
if __name__ == "__main__":
    app.run(debug=True)