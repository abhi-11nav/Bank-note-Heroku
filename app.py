
# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

from flask import Flask, render_template, request
import numpy as np
import pandas as pd
import sklearn  
import json
import pickle


app = Flask(__name__,template_folder="template")

model = pickle.load(open("model.pkl","rb"))

@app.route("/",methods=["GET"])
def home():
    return render_template("/index.html")

@app.route("/",methods=["POST"])
def predict():
    if request.method =="POST":
        
        variance = float(request.form['variance'])
        
        skewness = float(request.form['skewness'])
        
        curtosis = float(request.form['curtosis'])
        
        entropy = float(request.form['entropy'])
        
        
        input_feat = np.array([variance,skewness,curtosis,entropy])
        
        input_feat = input_feat.reshape(1,-1)
        
        prediction = model.predict(input_feat)
        
     
        
        if (prediction == 1):
            return render_template("/index.html",prediction_text ="VALID")
        else:
            return render_template("/index.html",prediction_text ="INVALID")

if __name__ == "__main__":
    app.run(debug=True)