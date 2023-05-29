from keras.applications.vgg16 import preprocess_input
from flask import Flask, request, render_template
from werkzeug.utils import secure_filename
from tensorflow import keras
import keras.utils as image
import numpy as np
import cv2 as cv
import os

app = Flask(__name__)
model = keras.models.load_model('model/chest_xray2.h5')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['GET', 'POST'])
def predict():
    if request.method == 'POST':
        f = request.files['file']
        basepath = os.path.dirname(__file__)
        absolutePath = os.path.join(basepath,'static', secure_filename(f.filename))
        f.save(absolutePath)

        image = cv.imread(absolutePath)
        grayImage = cv.cvtColor(image,cv.COLOR_BGR2GRAY)
        resized = cv.resize(grayImage,(128,128))
        normalized = resized/255
        reshaped = np.reshape(normalized,(1,128,128,1))
        result = model.predict(reshaped)


        normalResult =  result[0][0] * 100
        pneumoniaResult = result[0][1] * 100
        
        color = ""
        finalResult = ""

        if normalResult > 80:
            color = "danger"
            finalResult = "Negativo" 

        elif pneumoniaResult > 80:
            color = "success"
            finalResult = "Positivo" 
        
        else:
            color = "secondary"
            finalResult = "Ningun porcentaje sobrepasa el 80%"


        return render_template('prediccion.html', prediction=[normalResult, pneumoniaResult, secure_filename(f.filename), finalResult, color])
    else: 
        return render_template('prediccion.html')

if __name__ == '__main__':
    app.run(debug=True)