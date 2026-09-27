import pandas as pd

df = pd.read_csv("student_placement.csv")

print(df.head())
print(df.shape)
print(df.info())
print(df.isnull().sum())
print(df["CGPA"].median())
print(df["Coding_Score"].median())
df["CGPA"] = df["CGPA"].fillna(df["CGPA"].median())
df["Coding_Score"] = df["Coding_Score"].fillna(df["Coding_Score"].median())
print(df.isnull().sum())
print(df.duplicated().sum())
df = df.drop_duplicates()
print(df.shape)
print(df)
X = df.drop("Placed", axis=1)
y = df["Placed"]
print(X)
print(y)
y = y.map({"Yes": 1, "No": 0})
print(y)


from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print(X_train.shape)
print(X_test.shape)

from sklearn.linear_model import LogisticRegression

model = LogisticRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
print(y_pred)

print(y_test.values)

from sklearn.metrics import accuracy_score

accuracy = accuracy_score(y_test, y_pred)
print(accuracy)

from sklearn.metrics import confusion_matrix

cm = confusion_matrix(y_test, y_pred)
print(cm)

from sklearn.metrics import classification_report

print(classification_report(y_test, y_pred))

from sklearn.model_selection import cross_val_score

scores = cross_val_score(model, X, y, cv=5)
print(scores)
print(scores.mean())

new_student = pd.DataFrame([{
    "CGPA": 8.2,
    "Internships": 2,
    "Projects": 3,
    "Coding_Score": 82,
    "Communication_Score": 78
}])
prediction = model.predict(new_student)
print(prediction)

import joblib
joblib.dump(model, "placement_model.pkl")

import joblib
loaded_model = joblib.load("placement_model.pkl")
prediction = loaded_model.predict(new_student)
print(prediction)