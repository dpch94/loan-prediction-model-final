#import libraries
import numpy as np
import pandas as pd
from flask import Flask, request, jsonify, render_template
import pickle
import os

from xgboostclassifier_model import X_train11

#app name
app = Flask(__name__)

#load the saved model
def load_model():
    return pickle.load(open('lightgbm_loanpred_model.pkl', 'rb'))

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
    data_in = None
    if request.method == 'POST':

        data_in = request.form.to_dict()

        labels = ['not granted','granted']

        pred_df = pd.DataFrame([data_in.values()], columns=data_in.keys())
        print(pred_df)

        pred_df['Gender'] = pred_df['Gender'].astype('category')
        pred_df['Married'] = pred_df['Married'].astype('category')
        pred_df['Dependents'] = pred_df['Dependents'].astype('category')
        pred_df['Education'] = pred_df['Education'].astype('category')
        pred_df['Self_Employed'] = pred_df['Self_Employed'].astype('category')
        pred_df['Property_Area'] = pred_df['Property_Area'].astype('category')
        
        pred_df['ApplicantIncome'] = pred_df['ApplicantIncome'].astype('float64')
        pred_df['CoapplicantIncome'] = pred_df['CoapplicantIncome'].astype('float64')
        pred_df['LoanAmount'] = pred_df['LoanAmount'].astype('float64')
        pred_df['Loan_Amount_Term'] = pred_df['Loan_Amount_Term'].astype('float64')
        pred_df['Credit_History'] = pred_df['Credit_History'].astype('float64')

        model = load_model()
        prediction = model.predict(pred_df)

        result = labels[prediction[0]]

        return render_template('index.html', output='The Loan is {}'.format(result)) 


    else:
        return "Error. Please try again later."

if __name__ == "__main__":
    port=int(os.environ.get('PORT',5000))
    app.run(port=port,debug=True,use_reloader=False)