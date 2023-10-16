
import joblib
from flask import Flask, url_for, request, render_template
import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
# Define a flask app
app = Flask(__name__)

# Model saved with Keras model.save()
MODEL_PATH ='Model.h5'


# Load your trained model
# model = load_model(MODEL_PATH)
model = joblib.load(MODEL_PATH)


@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')

@app.route('/test', methods=['GET'])
def test():
    return render_template('test.html')


@app.route('/predict', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        data = request.form
        features = [prop for prop in data]
        values = list(data.values())
        values = list(map(float, values))

        sc_X = StandardScaler()
        Scaled_testset =  pd.DataFrame(sc_X.fit_transform(np.array(values).reshape(1,len(features))), columns=features)

        prediction = model.predict(Scaled_testset)
        print(prediction)
        output = prediction[0]
    
    return render_template('result.html', output=output)


if __name__ == '__main__':
    app.run(port=5001,debug=True)
