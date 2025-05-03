import joblib
import numpy as np

# Load the trained model and encoders
clf = joblib.load('models/disease_prediction_model.joblib')
le_tobacco = joblib.load('models/label_encoder_tobacco.joblib')
le_cond = joblib.load('models/label_encoder.joblib')

def make_prediction(user_input):
    try:
        user_input['Tobacco smoking status'] = le_tobacco.transform([user_input['Tobacco smoking status']])[0]

        user_features = np.array([
            user_input['AGE'], user_input['Body Height'], user_input['Body Weight'],
            user_input['Body mass index (BMI) [Ratio]'], user_input['Tobacco smoking status']
        ]).reshape(1, -1)

        probabilities = clf.predict_proba(user_features)[0]
        disease_probabilities = sorted(zip(le_cond.classes_, probabilities), key=lambda x: x[1], reverse=True)

        return {disease: round(probability * 100, 2) for disease, probability in disease_probabilities[:15]}

    except Exception as e:
        return {"error": str(e)}