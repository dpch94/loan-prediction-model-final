#import libraries
import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle
import os

#app name
app = Flask(__name__)

#load the saved model
def load_model():
    return pickle.load(open('xgboost_loanpred_model.pkl', 'rb'))

#home page
@app.route('/')
def home():
    return render_template('index.html')

#predict the result and return it
@app.route('/predict', methods=(['POST']))
def predict():
    '''
    For rendering results on HTML
    '''
    labels = ['not granted','granted']
   
    feature1 = request.form['ApplicantIncome']
    feature2 = request.form['CoapplicantIncome']
    feature3 = request.form['LoanAmount']
    feature4 = request.form['Loan_Amount_Term']
    feature5 = request.form['Credit_History']
    feature6 = request.form['Gender']
    feature7 = request.form['Married']
    feature8 = request.form['Dependents']
    feature9= request.form['Education']
    feature10= request.form['Self_Employed']
    
    model = load_model()

    prediction = model.predict([[feature1, feature2, feature3, feature4, feature5, feature6, feature7, feature8, feature9, feature10]])  

    result = labels[prediction[0]]
    
    return render_template('index.html', output='The Loan is {}'.format(result)) 
    


if __name__ == "__main__":
    port=int(os.environ.get('PORT',5000))
    app.run(port=port,debug=True,use_reloader=False)