from flask import Flask, render_template, request
import joblib
import pandas as pd
from datetime import datetime

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/disease_predictor')
def disease_predictor():
    return render_template('disease_selection.html')

# Diabetes Prediction
@app.route('/diabetes_prediction', methods=['GET', 'POST'])
def diabetes_prediction():
    if request.method == 'POST':
        try:
            model = joblib.load('models/diabetes_prediction_model.joblib')
            gender_encoder = joblib.load('models/label_encoder_GENDER.joblib')
            tobacco_encoder = joblib.load('models/label_encoder_Tobacco smoking status Value.joblib')
            
            data = request.form
            
            # Calculate BMI automatically from height and weight
            height = float(data['height'])
            weight = float(data['weight'])
            bmi = weight / ((height/100) ** 2)
            
            # Prepare input data with EXACT feature names expected by the model
            input_data = {
                'AGE': float(data['age']),
                'GENDER': gender_encoder.transform([data['gender']])[0],
                'Body Height Value': height,
                'Body Weight Value': weight,
                'BMI': bmi,  # Using calculated BMI
                'Diastolic Blood Pressure Value': float(data['diastolic_bp']),
                'Systolic Blood Pressure Value': float(data['systolic_bp']),
                'Erythrocyte distribution width [Entitic volume] by Automated count Value': float(data['erythrocyte_width']),
                'Erythrocytes [#/volume] in Blood by Automated count Value': float(data['erythrocytes']),
                'Heart rate Value': float(data['heart_rate']),
                'Hematocrit [Volume Fraction] of Blood by Automated count Value': float(data['hematocrit']),
                'Hemoglobin [Mass/volume] in Blood Value': float(data['hemoglobin']),
                'Leukocytes [#/volume] in Blood by Automated count Value': float(data['leukocytes']),
                'MCH [Entitic mass] by Automated count Value': float(data['mch']),
                'MCHC [Mass/volume] by Automated count Value': float(data['mchc']),
                'MCV [Entitic volume] by Automated count Value': float(data['mcv']),
                'Platelet distribution width [Entitic volume] in Blood by Automated count Value': float(data['platelet_width']),
                'Platelet mean volume [Entitic volume] in Blood by Automated count Value': float(data['platelet_volume']),
                'Platelets [#/volume] in Blood by Automated count Value': float(data['platelets']),
                'Respiratory rate Value': float(data['respiratory_rate']),
                'Tobacco smoking status Value': tobacco_encoder.transform([data['smoking_status']])[0]
            }
            
            # Debug print to verify all features
            print("Final input data with all features:", input_data)
            
            probability = model.predict_proba(pd.DataFrame([input_data]))[0][1] * 100
            return render_template('diabetes_result.html', 
                                probability=round(probability, 2),
                                risk_level="High" if probability > 50 else "Low")
            
        except Exception as e:
            print("Error during prediction:", str(e))
            return render_template('error.html', error=str(e))
    
    return render_template('diabetes_form.html')

# CKD Prediction
@app.route('/ckd_prediction', methods=['GET', 'POST'])
def ckd_prediction():
    if request.method == 'POST':
        try:
            model = joblib.load('models/ckd_prediction_model.joblib')
            gender_encoder = joblib.load('models/ckd_label_encoder_GENDER.joblib')
            tobacco_encoder = joblib.load('models/ckd_label_encoder_Tobacco smoking status Value.joblib')
            
            data = request.form
            input_data = {
                'GENDER': gender_encoder.transform([data['gender']])[0],
                'Body Height Value': float(data['height']),
                'Body Weight Value': float(data['weight']),
                'BMI': float(data['weight']) / ((float(data['height'])/100) ** 2),
                'AGE': float(data['age']),
                'Diastolic Blood Pressure Value': float(data['diastolic_bp']),
                'Systolic Blood Pressure Value': float(data['systolic_bp']),
                'Tobacco smoking status Value': tobacco_encoder.transform([data['smoking_status']])[0]
            }
            
            probability = model.predict_proba(pd.DataFrame([input_data]))[0][1] * 100
            return render_template('ckd_result.html', 
                                probability=round(probability, 2),
                                risk_level="High" if probability > 50 else "Low")
        except Exception as e:
            return render_template('error.html', error=str(e))
    return render_template('ckd_form.html')

@app.route('/ischemic_prediction', methods=['GET', 'POST'])
def ischemic_prediction():
    if request.method == 'POST':
        try:
            model = joblib.load('models/ischemic_prediction_model.joblib')
            gender_encoder = joblib.load('models/ischemic_label_encoder_GENDER.joblib')
            tobacco_encoder = joblib.load('models/ischemic_label_encoder_Tobacco smoking status Value.joblib')
            
            data = request.form
            
            # Calculate BMI automatically
            height = float(data['height'])
            weight = float(data['weight'])
            bmi = weight / ((height/100) ** 2)
            
            # Prepare input data with ALL required features
            input_data = {
                'GENDER': gender_encoder.transform([data['gender']])[0],
                'Body Height Value': height,
                'Body Weight Value': weight,
                'BMI': bmi,  # Changed to match what model expects
                'AGE': float(data['age']),
                'Diastolic Blood Pressure Value': float(data['diastolic_bp']),
                'Systolic Blood Pressure Value': float(data['systolic_bp']),
                'Heart rate Value': float(data['heart_rate']),
                'Respiratory rate Value': float(data['respiratory_rate']),
                'Erythrocyte distribution width [Entitic volume] by Automated count Value': float(data['erythrocyte_width']),
                'Erythrocytes [#/volume] in Blood by Automated count Value': float(data['erythrocytes']),
                'Hematocrit [Volume Fraction] of Blood by Automated count Value': float(data['hematocrit']),
                'Hemoglobin [Mass/volume] in Blood Value': float(data['hemoglobin']),
                'Leukocytes [#/volume] in Blood by Automated count Value': float(data['leukocytes']),
                'Platelets [#/volume] in Blood by Automated count Value': float(data['platelets']),
                'Tobacco smoking status Value': tobacco_encoder.transform([data['smoking_status']])[0]
            }
            
            # Debug print to verify all features
            print("Final input data for ischemic prediction:", input_data)
            
            probability = model.predict_proba(pd.DataFrame([input_data]))[0][1] * 100
            return render_template('ischemic_result.html', 
                                probability=round(probability, 2),
                                risk_level="High" if probability > 50 else "Low")
            
        except Exception as e:
            print("Error during ischemic prediction:", str(e))
            return render_template('error.html', error=str(e))
    
    return render_template('ischemic_form.html')

# ... (keep all other routes the same)

@app.route('/hospital_finder', methods=['GET', 'POST'])
def hospital_finder():
    df = pd.read_csv('data/organizations.csv')
    cities = df['CITY'].unique().tolist()
    
    selected_city = None
    hospitals = []
    
    if request.method == 'POST':
        try:
            selected_city = request.form['city']
            hospitals = df[df['CITY'].str.lower() == selected_city.lower()]['NAME'].tolist()
        except Exception as e:
            return render_template('error.html', error=str(e))
    
    return render_template('hospital_finder.html', 
                         cities=cities,
                         selected_city=selected_city,
                         hospitals=hospitals)

# ... (rest of the file remains the same)

if __name__ == '__main__':
    app.run(debug=True)
