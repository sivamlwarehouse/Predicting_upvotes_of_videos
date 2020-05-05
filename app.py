# -*- coding: utf-8 -*-
"""
Created on Fri Jan 24 13:08:27 2020

@author: admin
"""
import numpy as np
import pandas as pd
from flask import Flask,request,jsonify,render_template
import pickle



app=Flask(__name__)
model=pickle.load(open('D:/siva/Deployment/upvotes/upvotes.pkl','rb'))



@app.route('/')
def home():
    return render_template('index.html')


@app.route('/predict',methods=['post'])
def predict():
    '''
    For rendering results on HTML GUI
    '''
    int_features=[int(x) for x in request.form.values()]
    final_features=[np.array(int_features)]
    prediction=model.predict(final_features)
    
    output=round(prediction[0],2)
    return render_template('index.html',prediction_text='Video gets upvotes approximatesly $ {}'.format(output))

if __name__=="__main__":
    app.run(debug=True)

