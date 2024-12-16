from flask import Flask, render_template, request
import joblib
import pickle
import numpy as np
import pandas as pd

app = Flask(__name__)

# Load all models and label encoders
# Science model and label encoders
science_model = joblib.load('Science_Model_Categorical.pkl')
science_label_encoders = joblib.load('label_encoders_categorical.pkl')
science_skills_data = pd.read_excel('datasets\\scienceskillsnew.xlsx')

# Commerce model and label encoders
commerce_model = pickle.load(open('career_prediction_model.pkl', 'rb'))
commerce_label_encoders = pickle.load(open('label_encoders.pkl', 'rb'))
commerce_skills_data = pd.read_excel('datasets\\CommerceOccupationsnew.xlsx', sheet_name='skills')

# Arts model and label encoders
arts_model = joblib.load('arts_model.pkl')
arts_label_encoders = joblib.load('arts_label_encoders.pkl')
arts_skills_data = pd.read_excel('datasets\\ArtsOccupationskills.xlsx', sheet_name='Skills')

# Strip extra whitespace in the fields
science_skills_data['Field of Interest'] = science_skills_data['Field of Interest'].str.strip()
commerce_skills_data['Field of Interest'] = commerce_skills_data['Field of Interest'].str.strip()
arts_skills_data['Field of Interest'] = arts_skills_data['Field of Interest'].str.strip()

# Home route (for selecting category)
@app.route('/')
def home():
    return render_template('index.html')

# Route for Science prediction
@app.route('/predict_science', methods=['POST'])
def predict_science():
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
            science_label_encoders[feature].transform([value])[0]
            for feature, value in zip(categorical_features, categorical_values)
        ]

        # Combine encoded values into a single input array
        input_features = np.array(encoded_categorical_values).reshape(1, -1)

        # Make a prediction
        prediction_encoded = science_model.predict(input_features)[0]

        # Decode the prediction to get the 'Field of Interest'
        field_of_interest = science_label_encoders['Field of Interest'].inverse_transform([prediction_encoded])[0]

        # Retrieve the career path details
        career_details = science_skills_data[science_skills_data['Field of Interest'] == field_of_interest].iloc[0]
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

# Route for Commerce prediction
@app.route('/predict_commerce', methods=['POST'])
def predict_commerce():
    try:
        # Define numerical and categorical features for commerce prediction
        numerical_features = [
            'Business Communication Skills', 'Decision-Making', 'Marketing Knowledge',
            'Risk Management', 'Taxation Knowledge'
        ]
        numerical_values = [int(request.form[feature]) for feature in numerical_features]

        categorical_features = [
            'Financial Analysis Skills', 'Accounting Knowledge', 'Negotiation Skills', 'Team Management',
            'Financial Regulations', 'Customer Service Skills', 'Sales Acumen', 'Technological Adaptability',
            'Market Research Skills', 'Strategic Planning', 'Budgeting & Forecasting', 'Data Analysis',
            'Investment Knowledge', 'Product Development Insight', 'Supply Chain Knowledge'
        ]
        categorical_values = [request.form[feature] for feature in categorical_features]

        # Encode categorical values
        encoded_categorical_values = [
            commerce_label_encoders[feature].transform([value])[0]
            for feature, value in zip(categorical_features, categorical_values)
        ]

        # Combine numerical and categorical features
        input_features = np.array(numerical_values + encoded_categorical_values).reshape(1, -1)

        # Predict using the trained model
        prediction = commerce_model.predict(input_features)[0]

        # Decode the prediction to get the occupation
        occupation = list(commerce_label_encoders['Occupation'].inverse_transform([prediction]))[0]

        # Filter the skills data based on the predicted occupation
        career_details = commerce_skills_data[commerce_skills_data['Field of Interest'] == occupation]
        
        if career_details.empty:
            return render_template(
                'error.html',
                message=f"No career details found for the predicted occupation: {occupation}."
            )

        career_details = career_details.iloc[0]
        foundational_skills = career_details['Foundational Skills']
        intermediate_skills = career_details['Intermediate-Level Skills']
        professional_skills = career_details['Professional-Level Skills']

        return render_template(
            'result.html',
            prediction=occupation,
            foundational_skills=foundational_skills,
            intermediate_skills=intermediate_skills,
            professional_skills=professional_skills
        )

    except Exception as e:
        return render_template('error.html', message=f"An error occurred: {str(e)}")

# Route for Arts prediction
@app.route('/predict_arts', methods=['POST'])
def predict_arts():
    try:
        # Collect numerical and categorical inputs for arts prediction
        numerical_features = [
            'Social Awareness', 'Communication Skills', 'Empathy and Counseling Skills', 'Critical Thinking',
            'Cultural Literacy', 'Research Skills'
        ]
        numerical_values = [int(request.form[feature]) for feature in numerical_features]

        categorical_features = [
            'Public Speaking', 'Writing and Editing', 'Interpersonal Skills', 'Ethical Judgment', 'Problem-Solving',
            'Legal Knowledge', 'Analytical Skills', 'Negotiation Skills', 'Advocacy', 'Strategic Thinking',
            'Language Proficiency', 'Emotional Intelligence'
        ]
        categorical_values = [request.form[feature] for feature in categorical_features]

        # Encode categorical values
        encoded_categorical_values = [
            arts_label_encoders[feature].transform([value])[0] if value in arts_label_encoders[feature].classes_ else -1
            for feature, value in zip(categorical_features, categorical_values)
        ]

        # Combine numerical and categorical values
        input_features = np.array(numerical_values + encoded_categorical_values).reshape(1, -1)

        # Make a prediction using the trained model
        prediction = arts_model.predict(input_features)[0]

        # Decode the prediction (Occupation)
        occupation = list(arts_label_encoders['Occupation'].inverse_transform([prediction]))[0]

        career_details = arts_skills_data[arts_skills_data['Field of Interest'] == occupation]
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
    app.run(debug=True, port=5000)
