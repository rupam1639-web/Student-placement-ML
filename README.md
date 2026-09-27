# Student Placement Prediction

A beginner-level Machine Learning project that predicts whether a student is likely to be placed based on academic and skill-related features.

## Project Overview

This project uses **Logistic Regression** to predict student placement.

The model uses the following features:

* CGPA
* Number of Internships
* Number of Projects
* Coding Score
* Communication Score

The target variable is:

* `1` → Placed
* `0` → Not Placed

## ML Workflow

The project follows this workflow:

1. Load the dataset
2. Explore the data
3. Handle missing values
4. Remove duplicate records
5. Separate features and target
6. Encode the target variable
7. Split the data into training and testing sets
8. Train a Logistic Regression model
9. Make predictions
10. Evaluate the model
11. Perform cross-validation
12. Save and load the trained model
13. Predict placement for a new student

## Technologies Used

* Python
* Pandas
* Scikit-learn
* Joblib

## Model

**Logistic Regression**

The model was trained using the cleaned student placement dataset.

## Evaluation

The model was evaluated using:

* Accuracy
* Confusion Matrix
* Precision
* Recall
* F1-score
* 5-fold Cross-validation

> Note: The dataset used in this project is very small, so the evaluation results should not be considered representative of real-world placement prediction performance.

## Example Prediction

Example student:

```text
CGPA: 8.2
Internships: 2
Projects: 3
Coding Score: 82
Communication Score: 78
```

The model predicts:

```text
Placed
```

## Project Structure

```text
student-placement-ml/
│
├── student_placement.csv
├── your_python_file.py
├── placement_model.pkl
└── README.md
```

## Future Improvements

* Build an API using FastAPI
* Create a simple web interface
* Improve the dataset with more real-world data
* Experiment with other ML algorithms
* Deploy the application
* Add model monitoring

## Learning Goal

This project was built as a practical introduction to the **Machine Learning Engineering workflow**, from data preparation and model training to model saving and prediction.
