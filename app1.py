from flask import Flask, render_template, request
import joblib
import numpy as np
import pandas as pd

app = Flask(__name__)

# Load the trained model and label encoders
model = joblib.load('Science_Model_Categorical.pkl')
label_encoders = joblib.load('label_encoders_categorical.pkl')

# Load the dataset containing career paths
skills_data = pd.read_excel('datasets\\scienceskillsnew.xlsx')  # Adjust the path if necessary

# Home route
@app.route('/')
def home():
    return render_template('index1.html')

# Prediction route
@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Define the categorical features used for prediction
        categorical_features = [
            'Experiment Comfort', 'Problem Solving', 'Math Comfort', 'Tech Interest',
            'Field vs. Lab', 'Long-Term Projects', 'Attention to Detail', 'Real-World Applications',
            'Work Style', 'Adaptability', 'Interest in Reading', 'Creativity',
            'Patient Interaction', 'Design Interest', 'Technical Comfort', 'Bio/Chem Interest',
            'Earth/Space Interest', 'Environmental Interest', 'Human Behavior'
        ]

        # Gather the values from the form
        categorical_values = [request.form[feature] for feature in categorical_features]

        # Encode the categorical values
        encoded_categorical_values = [
            label_encoders[feature].transform([value])[0]
            for feature, value in zip(categorical_features, categorical_values)
        ]

        # Combine encoded values into a single input array
        input_features = np.array(encoded_categorical_values).reshape(1, -1)

        # Make a prediction
        prediction_encoded = model.predict(input_features)[0]

        # Decode the prediction to get the 'Field of Interest'
        field_of_interest = label_encoders['Field of Interest'].inverse_transform([prediction_encoded])[0]

        # Retrieve the career path details
        career_details = skills_data[skills_data['Field of Interest'] == field_of_interest].iloc[0]
        foundational_skills = career_details['Foundational Skills']
        intermediate_skills = career_details['Intermediate-Level Skills']
        professional_skills = career_details['Professional-Level Skills']

        return render_template(
            'result.html',
            prediction=field_of_interest,
            foundational_skills=foundational_skills,
            intermediate_skills=intermediate_skills,
            professional_skills=professional_skills
        )

    except Exception as e:
        return f"An error occurred: {str(e)}"

if __name__ == '__main__':
    app.run(debug=True,host='0.0.0.0',port=6002)
