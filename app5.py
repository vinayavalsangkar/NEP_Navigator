from flask import Flask, render_template, request, redirect, url_for

app = Flask(__name__)

# Questions and options
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


@app.route('/', methods=['GET', 'POST'])
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
            return render_template('result1.html', recommended_field=recommended_field)

    # Initial question
    return render_template('survey.html', question=questions[0], current_index=0)


@app.route('/redirect/<field>')
def redirect_field(field):
    """Redirect to appropriate file based on the field."""
    if field == "Arts":
        return redirect("http://192.168.1.2:6003")
    elif field == "Commerce":
        return redirect("http://192.168.1.2:6000")
    elif field == "Science":
        return redirect("http://192.168.1.2:6002")
    else:
        return "Invalid field"


if __name__ == '__main__':
    app.run(debug=True)
