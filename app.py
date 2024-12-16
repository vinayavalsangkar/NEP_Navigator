from flask import Flask, render_template, request
import pickle
import numpy as np
import pandas as pd

app = Flask(__name__)

# Load the trained model and label encoders
with open('career_prediction_model.pkl', 'rb') as model_file:
    model = pickle.load(model_file)

with open('label_encoders.pkl', 'rb') as encoder_file:
    label_encoders = pickle.load(encoder_file)

# Load the skills data from Excel
skills_data = pd.read_excel('datasets\\CommerceOccupationsnew.xlsx', sheet_name='skills')

# Strip any extra whitespace from the 'Field of Interest' column for consistency
skills_data['Field of Interest'] = skills_data['Field of Interest'].str.strip()

# Home route
@app.route('/')
def home():
    return render_template('index.html')

# Prediction route
@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Define numerical features
        numerical_features = [
            'Business Communication Skills',
            'Decision-Making',
            'Marketing Knowledge',
            'Risk Management',
            'Taxation Knowledge'
        ]
        
        # Collect numerical input values from the form
        numerical_values = [int(request.form[feature]) for feature in numerical_features]

        # Define categorical features
        categorical_features = [
            'Financial Analysis Skills',
            'Accounting Knowledge',
            'Negotiation Skills',
            'Team Management',
            'Financial Regulations',
            'Customer Service Skills',
            'Sales Acumen',
            'Technological Adaptability',
            'Market Research Skills',
            'Strategic Planning',
            'Budgeting & Forecasting',
            'Data Analysis',
            'Investment Knowledge',
            'Product Development Insight',
            'Supply Chain Knowledge'
        ]
        
        # Collect categorical input values from the form
        categorical_values = [request.form[feature] for feature in categorical_features]

        # Encode categorical values
        encoded_categorical_values = [
            label_encoders[feature].transform([value])[0]
            for feature, value in zip(categorical_features, categorical_values)
        ]

        # Combine numerical and categorical features into a single array
        input_features = np.array(numerical_values + encoded_categorical_values).reshape(1, -1)

        # Predict using the trained model
        prediction = model.predict(input_features)[0]

        # Decode the prediction to get the occupation name
        occupation = list(label_encoders['Occupation'].inverse_transform([prediction]))[0]

        # Filter the skills data based on the predicted occupation
        career_details = skills_data[skills_data['Field of Interest'] == occupation]

        # Check if the filtered DataFrame is empty
        if career_details.empty:
            return render_template(
                'error.html',
                message=f"No career details found for the predicted occupation: {occupation}."
            )

        # Safely access the first row of the filtered data
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
        # Render an error template if an exception occurs
        return render_template('error.html', message=f"An error occurred: {str(e)}")

if __name__ == '__main__':
    app.run(debug=True,port=5000)