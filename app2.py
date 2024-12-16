from flask import Flask, render_template, request
import joblib
import numpy as np
import pandas as pd

app = Flask(__name__)

# Load the saved model and label encoders
model_filename = "arts_model.pkl"
label_encoder_filename = "arts_label_encoders.pkl"

model = joblib.load(model_filename)
label_encoders = joblib.load(label_encoder_filename)

skills_data = pd.read_excel('datasets\\ArtsOccupationskills.xlsx', sheet_name='Skills')

skills_data['Field of Interest'] = skills_data['Field of Interest'].str.strip()
# Home route
@app.route('/')
def home():
    return render_template('index2.html')

# Prediction route
@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Collect numerical inputs
        numerical_features = [
            'Social Awareness',
            'Communication Skills',
            'Empathy and Counseling Skills',
            'Critical Thinking',
            'Cultural Literacy',
            'Research Skills'
        ]
        
        numerical_values = [int(request.form[feature]) for feature in numerical_features]

        # Collect categorical inputs
        categorical_features = [
            'Public Speaking',
            'Writing and Editing',
            'Interpersonal Skills',
            'Ethical Judgment',
            'Problem-Solving',
            'Legal Knowledge',
            'Analytical Skills',
            'Negotiation Skills',
            'Advocacy',
            'Strategic Thinking',
            'Language Proficiency',
            'Emotional Intelligence'
        ]
        
        categorical_values = [request.form[feature] for feature in categorical_features]

        # Encode categorical values
        encoded_categorical_values = []
        for feature, value in zip(categorical_features, categorical_values):
            if value in label_encoders[feature].classes_:
                encoded_value = label_encoders[feature].transform([value])[0]
            else:
                encoded_value = -1  # Default for unknown values
            encoded_categorical_values.append(encoded_value)

        # Combine numerical and categorical values
        input_features = np.array(numerical_values + encoded_categorical_values).reshape(1, -1)

        # Make a prediction using the trained model
        prediction = model.predict(input_features)[0]

        # Decode the prediction (Occupation)
        occupation = list(label_encoders['Occupation'].inverse_transform([prediction]))[0]

        career_details = skills_data[skills_data['Field of Interest'] == occupation]
        if career_details.empty:
            return render_template(
                'error.html',
                message=f"No career details found for the predicted occupation: {occupation}."
            )


        career_details = career_details.iloc[0]
        foundational_skills = career_details['Foundational Skills']
        intermediate_skills = career_details['Intermediate-Level Skills']
        professional_skills = career_details['Professional-Level Skills']

        # Render the result template with the prediction and skills details
        return render_template(
            'result.html',
            prediction=occupation,
            foundational_skills=foundational_skills,
            intermediate_skills=intermediate_skills,
            professional_skills=professional_skills
        )

    except Exception as e:
        return f"An error occurred: {str(e)}"

if __name__ == '__main__':
    app.run(debug=True,host='0.0.0.0',port=6003)
