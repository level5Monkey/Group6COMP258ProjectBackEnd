# -*- coding: utf-8 -*-
"""
Created on Sun Dec 12 09:19:59 2021

@author: Jiaxing Li
"""

import sys
import pickle
import json
import numpy as np
import pandas as pd
from flask import Flask
from flask import request
from tensorflow import keras
from flask_cors import CORS

app = Flask(__name__)
CORS(app)



with open("encoder.pkl", "rb") as f: 
    encoder = pickle.load(f) 

with open("scaler.pkl", "rb") as f: 
    scaler = pickle.load(f)     
    
model = keras.models.load_model("model_Jiaxing.h5")

@app.route("/")
def hello_world():
    return "<p>test!</p>"
    

@app.route("/predict", methods=['GET','POST'])
def predict():
    
    data = request.get_json(force=True)
    df = pd.DataFrame([data.values()], columns=data.keys())
    
    df['FIRST YEAR PERSISTENCE COUNT'] = df['FIRST YEAR PERSISTENCE COUNT'].astype(int)
    df['PROGRAM LENGTH'] = df['PROGRAM LENGTH'].astype(int)
    df['REST SEMESTERS'] = df['REST SEMESTERS'].astype(int)
    
    NUMERICAL_FEATURE_KEYS = ['PROGRAM LENGTH', 'REST SEMESTERS']
    
    data_num = df[NUMERICAL_FEATURE_KEYS]
    data_cat = df.drop(NUMERICAL_FEATURE_KEYS, axis=1)
    
    data_cat = encoder.transform(data_cat).toarray()
    
    data_num = scaler.transform(data_num)
    data_num = pd.DataFrame(data = data_num, columns = scaler.feature_names_in_)
    
    
    X = pd.DataFrame(data = data_cat, columns = encoder.get_feature_names()).join(data_num)
    
    result = str(model.predict(X)[0][0])
    return {
        "result": result,
    }    

if __name__ == '__main__':
    try:
        port = int(sys.argv[1]) # This is for a command-line input
    except:
        port = 5000 # If you don't provide any port the port will be set to 12345
    app.run(port=port, debug=True)