from flask import Flask,request,render_template
import pickle
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score,confusion_matrix, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import classification_report
from sklearn.model_selection import KFold, cross_val_score
import statistics as stat


app=Flask(__name__)
with open('model.pkl','rb') as model_file:
   model=pickle.load(model_file)

@app.route('/')
def home():
   return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():
   Credit_Mix=request.form['Credit_Mix']
   if Credit_Mix=="Bad":
     Credit_Mix=1
   elif Credit_Mix=="Standard":
     Credit_Mix=2
   elif Credit_Mix=="Good":
     Credit_Mix=3
   Annual_Income=float(request.form['Annual_Income'])
   Num_Bank_Accounts=float(request.form['Num_Bank_Accounts'])
   Num_Credit_Card=float(request.form['Num_Credit_Card'])
   Interest_Rate=float(request.form['Interest_Rate'])
   Num_of_Loan=float(request.form['Num_of_Loan'])
   Delay_from_due_date=float(request.form['Delay_from_due_date'])
   Num_of_Delayed_Payment=float(request.form['um_of_Delayed_Payment'])
   Changed_Credit_Limit=float(request.form['Changed_Credit_Limit'])
   Num_Credit_Inquiries=float(request.form['Num_Credit_Inquiries'])
   Outstanding_Debt=float(request.form['Outstanding_Debt'])
   Credit_Utilization_Ratio=float(request.form['Credit_Utilization_Ratio'])
   Credit_History_Age=float(request.form['Credit_History_Age'])
   Total_EMI_per_month=float(request.form['Total_EMI_per_month'])
   Amount_invested_monthly=float(request.form['Amount_invested_monthly'])
   Monthly_Balance=float(request.form['Monthly_Balance'])
   feature=np.array([[Annual_Income,Num_Bank_Accounts,Num_Credit_Card,Interest_Rate,Num_of_Loan,Delay_from_due_date, Num_of_Delayed_Payment,Changed_Credit_Limit,Num_Credit_Inquiries, Credit_Mix, Outstanding_Debt,Credit_Utilization_Ratio,Credit_History_Age, Total_EMI_per_month,Amount_invested_monthly,Monthly_Balance]])
   prediction=model.predict(feature)
   return render_template('index.html',pred_res=prediction[0])

   
if __name__=='__main__':
  app.run(debug=True) 