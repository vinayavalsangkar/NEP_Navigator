const questions = [
    {
        question: "Creative Expression:",
        options: ["Writing", "Art", "Music", "Craft", "Other"]
    },
    {
        question: "Preferred Project Type:",
        options: ["Research", "Experiments", "Creative", "Business", "Community"]
    },
    {
        question: "Learning Style:",
        options: ["Hands-on", "Reading", "Discussion", "Visual", "Group"]
    },
    {
        question: "Motivation:",
        options: ["Interest", "Grades", "Career", "Impact", "Skills"]
    },
    {
        question: "Outside Activities:",
        options: ["Sports", "Arts", "Volunteering", "Business", "Tech"]
    },
    {
        question: "Discussion Topics:",
        options: ["Society", "Technology", "Literature", "Economics", "Science"]
    },
    {
        question: "Problem Solving:",
        options: ["Analyze", "Collaborate", "Create", "Follow", "Seek Help"]
    },
    {
        question: "Role in Groups:",
        options: ["Leader", "Researcher", "Creative", "Planner", "Supporter"]
    },
    {
        question: "Ideal Environment:",
        options: ["Studio", "Office", "Lab", "Classroom", "Outdoor"]
    },
    {
        question: "Favorite Subjects:",
        options: ["Behavior", "Environment", "Business", "History", "Technology"]
    },
];

const scores = {
    Science: 0,
    Commerce: 0,
    Arts: 0,
    Other: 0,
};

let currentQuestionIndex = 0;

function displayQuestion() {
    const questionText = document.getElementById('question-text');
    const answerSelect = document.getElementById('answer-select');
    const nextButton = document.getElementById('next-button');

    // Clear previous options
    answerSelect.innerHTML = '<option value="" disabled selected>Select your answer</option>';

    // Get the current question
    const currentQuestion = questions[currentQuestionIndex];
    
    // Display the question
    questionText.innerText = currentQuestion.question;

    // Populate the dropdown with options
    currentQuestion.options.forEach(option => {
        const optionElement = document.createElement('option');
        optionElement.value = option;
        optionElement.textContent = option;
        answerSelect.appendChild(optionElement);
    });

    // Hide the button until an option is selected
    nextButton.disabled = true;
    answerSelect.onchange = () => {
        nextButton.disabled = false;
    };
}

function calculateScores(answer) {
    switch (answer) {
        case "Writing":
        case "Art":
        case "Music":
        case "Craft":
            scores.Arts += 1;
            break;
        case "Research":
        case "Experiments":
        case "Tech":
            scores.Science += 1;
            break;
        case "Business":
            scores.Commerce += 1;
            break;
        case "Other":
            scores.Other += 1;
            break;
        // Add more cases as needed for other answers
    }
}

document.getElementById('next-button').addEventListener('click', () => {
    const answer = document.getElementById('answer-select').value;
    calculateScores(answer);
    
    currentQuestionIndex++;
    if (currentQuestionIndex < questions.length) {
        displayQuestion();
    } else {
        // Display completion message and recommended field
        const recommendedField = Object.keys(scores).reduce((a, b) => scores[a] > scores[b] ? a : b);
        document.getElementById('question-container').innerHTML = `
            <h1>Thank you for completing the survey!</h1>
            <p>We recommend pursuing: <strong>${recommendedField}</strong></p>
        `;
    }
});

// Initialize the first question
displayQuestion();