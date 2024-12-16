from flask import Flask, render_template, request, redirect, url_for
import joblib
import pickle
import numpy as np
import pandas as pd

app = Flask(__name__)

# Load models and label encoders for each field
model_commerce = pickle.load(open('career_prediction_model.pkl', 'rb'))
label_encoders_commerce = pickle.load(open('label_encoders.pkl', 'rb'))

model_science = joblib.load('Science_Model_Categorical.pkl')
label_encoders_science = joblib.load('label_encoders_categorical.pkl')

model_arts = joblib.load('arts_model.pkl')
label_encoders_arts = joblib.load('arts_label_encoders.pkl')

# Load datasets
skills_data_commerce = pd.read_excel('datasets\\CommerceOccupationsnew.xlsx', sheet_name='Sheet4')
skills_data_science = pd.read_excel('datasets\\scienceskillsnew.xlsx')
skills_data_arts = pd.read_excel('datasets\\ArtsOccupationskills.xlsx', sheet_name='Skills')

# Survey questions for all fields
questions = [
    {"question": "Creative Expression:", "options": ["Writing", "Art", "Music", "Craft", "Other"]},
    {"question": "Preferred Project Type:", "options": ["Research", "Experiments", "Creative", "Business", "Community"]},
    {"question": "Learning Style:", "options": ["Hands-on", "Reading", "Discussion", "Visual", "Group"]},
    {"question": "Motivation:", "options": ["Interest", "Grades", "Career", "Impact", "Skills"]},
    {"question": "Outside Activities:", "options": ["Sports", "Arts", "Volunteering", "Business", "Tech"]},
    {"question": "Discussion Topics:", "options": ["Society", "Technology", "Literature", "Economics", "Science"]},
    {"question": "Problem Solving:", "options": ["Analyze", "Collaborate", "Create", "Follow", "Seek Help"]},
    {"question": "Role in Groups:", "options": ["Leader", "Researcher", "Creative", "Planner", "Supporter"]},
    {"question": "Ideal Environment:", "options": ["Studio", "Office", "Lab", "Classroom", "Outdoor"]},
    {"question": "Favorite Subjects:", "options": ["Behavior", "Environment", "Business", "History", "Technology"]},
]

# Initialize scores
scores = {
    "Science": 0,
    "Commerce": 0,
    "Arts": 0,
    "Other": 0,
}

def calculate_scores(answer):
    """Update scores based on the selected answer."""
    global scores
    if answer in ["Writing", "Art", "Music", "Craft"]:
        scores["Arts"] += 1
    elif answer in ["Research", "Experiments", "Tech"]:
        scores["Science"] += 1
    elif answer == "Business":
        scores["Commerce"] += 1
    elif answer == "Other":
        scores["Other"] += 1
@app.route('/vision')
def vision():
    """Render the Vision page."""
    return render_template('vision.html')

@app.route('/', methods=['GET'])
def landing():
    """Render the landing page."""
    return render_template('landing.html')


@app.route('/survey', methods=['GET', 'POST'])
def survey():
    """Render the survey and handle responses."""
    if request.method == 'POST':
        current_index = int(request.form['current_index'])
        answer = request.form['answer']

        # Update scores based on answer
        calculate_scores(answer)

        # Proceed to the next question or display results
        if current_index + 1 < len(questions):
            return render_template('survey.html', question=questions[current_index + 1], current_index=current_index + 1)
        else:
            # Determine the recommended field
            recommended_field = max(scores, key=scores.get)
            return redirect(url_for('redirect_field', field=recommended_field))

    # Initial question
    return render_template('survey.html', question=questions[0], current_index=0)


@app.route('/redirect/<field>')
def redirect_field(field):
    """Display prediction and allow user to proceed to the next field."""
    # Display predicted field
    if field == "Arts":
        return render_template('prediction_result.html', field=field, next_page=url_for('arts_predict'))
    elif field == "Commerce":
        return render_template('prediction_result.html', field=field, next_page=url_for('commerce_predict'))
    elif field == "Science":
        return render_template('prediction_result.html', field=field, next_page=url_for('science_predict'))
    else:
        return "Invalid field"


# ========================== Commerce Prediction ==========================

@app.route('/commerce_predict', methods=['GET', 'POST'])
def commerce_predict():
    """Render the commerce-specific prediction page."""
    if request.method == 'GET':
        return render_template('index.html')

    # Prediction logic as per the provided code
    try:
        numerical_features = [
            'Business Communication Skills', 'Decision-Making', 'Marketing Knowledge',
            'Risk Management', 'Taxation Knowledge'
        ]
        numerical_values = [int(request.form[feature]) for feature in numerical_features]
        categorical_features = [
            'Financial Analysis Skills', 'Accounting Knowledge', 'Negotiation Skills',
            'Team Management', 'Financial Regulations', 'Customer Service Skills',
            'Sales Acumen', 'Technological Adaptability', 'Market Research Skills',
            'Strategic Planning', 'Budgeting & Forecasting', 'Data Analysis',
            'Investment Knowledge', 'Product Development Insight', 'Supply Chain Knowledge'
        ]
        categorical_values = [request.form[feature] for feature in categorical_features]
        encoded_categorical_values = [
            label_encoders_commerce[feature].transform([value])[0]
            for feature, value in zip(categorical_features, categorical_values)
        ]
        input_features = np.array(numerical_values + encoded_categorical_values).reshape(1, -1)
        prediction = model_commerce.predict(input_features)[0]
        occupation = label_encoders_commerce['Occupation'].inverse_transform([prediction])[0]
        career_details = skills_data_commerce[skills_data_commerce['Field of Interest'] == occupation]
        if career_details.empty:
            return render_template(
                'error.html',
                message=f"No career details found for the predicted occupation: {occupation}."
            )
        career_details = career_details.iloc[0]
        return render_template(
            'result.html',
            prediction=occupation,
            foundational_skills=career_details['Foundational Skills'],
            intermediate_skills=career_details['Intermediate-Level Skills'],
            professional_skills=career_details['Professional-Level Skills']
        )
    except Exception as e:
        return render_template('error.html', message=f"An error occurred: {str(e)}")


# ========================== Science Prediction ==========================

@app.route('/science_predict', methods=['GET', 'POST'])
def science_predict():
    """Render the science-specific prediction page."""
    if request.method == 'GET':
        return render_template('index1.html')
    try:
        categorical_features = [
            'Experiment Comfort', 'Problem Solving', 'Math Comfort', 'Tech Interest',
            'Field vs. Lab', 'Long-Term Projects', 'Attention to Detail', 'Real-World Applications',
            'Work Style', 'Adaptability', 'Interest in Reading', 'Creativity',
            'Patient Interaction', 'Design Interest', 'Technical Comfort', 'Bio/Chem Interest',
            'Earth/Space Interest', 'Environmental Interest', 'Human Behavior'
        ]
        # Gather and encode user inputs
        categorical_values = [request.form[feature] for feature in categorical_features]
        encoded_values = [label_encoders_science[feature].transform([value])[0] 
                          for feature, value in zip(categorical_features, categorical_values)]
        input_features = np.array(categorical_values).reshape(1, -1)

        
        prediction = model_science.predict([encoded_values])[0]
        field_of_interest = label_encoders_science['Field of Interest'].inverse_transform([prediction])[0]
        career_details = skills_data_science[skills_data_science['Field of Interest'] == field_of_interest]
        
        if career_details.empty:
            return render_template('error.html', 
                                   message=f"No career details found for: {field_of_interest}.")
        
        career_details = career_details.iloc[0]
        return render_template(
            'result.html',
            prediction=field_of_interest,
            foundational_skills=career_details['Foundational Skills'],
            intermediate_skills=career_details['Intermediate-Level Skills'],
            professional_skills=career_details['Professional-Level Skills']
        )
    except Exception as e:
        return render_template('error.html',message=f"An error occurred during prediction: {str(e)}")
    # Prediction logic continues here...

# ========================== Arts Prediction ==========================

@app.route('/arts_predict', methods=['GET', 'POST'])
def arts_predict():
    """Render the arts-specific prediction page."""
    if request.method == 'GET':
        return render_template('index2.html')
    # Prediction logic continues here...
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
            if value in label_encoders_arts[feature].classes_:
                encoded_value = label_encoders_arts[feature].transform([value])[0]
            else:
                encoded_value = -1  # Default for unknown values
            encoded_categorical_values.append(encoded_value)

        # Combine numerical and categorical values
        input_features = np.array(numerical_values + encoded_categorical_values).reshape(1, -1)

        # Make a prediction using the trained model
        prediction = model_arts.predict(input_features)[0]

        # Decode the prediction (Occupation)
        occupation = list(label_encoders_arts['Occupation'].inverse_transform([prediction]))[0]
        career_details = skills_data_arts[skills_data_arts['Field of Interest'] == occupation]
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
    app.run(debug=True)
