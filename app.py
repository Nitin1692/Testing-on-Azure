from flask import Flask, render_template, request
import tensorflow as tf
import numpy as np
app = Flask(__name__)

@app.route('/',methods=['GET'])
def hello_world():
    return render_template('index.html')

@app.route('/',methods=['POST'])
def predict():
    imagefile = request.files['imagefile']
    image_path = "./images/" + imagefile.filename
    imagefile.save(image_path)
    model = tf.keras.models.load_model('model.h5')
    img = tf.keras.utils.load_img(image_path, target_size = (224,224))
    imagee=tf.keras.utils.img_to_array(img)
    imagee=np.expand_dims(imagee, axis=0)
    img_data=tf.keras.applications.densenet.preprocess_input(imagee)
    prediction=model.predict(img_data)
    if prediction[0][0]>prediction[0][1]:  
        pred = "Person is Safe."
    else:
        pred = "Person is affected with Pneumonia."

    return render_template('index.html', prediction=pred)
    

if __name__ == '__main__':
    app.run(host='0.0.0.0',port=5000)