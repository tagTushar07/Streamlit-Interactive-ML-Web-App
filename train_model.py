"""
Train & Export Machine Learning Model
Task: Streamlit Interactive ML Web App
Due Date: 17 October 2026
"""

import pandas as pd
from sklearn.tree import DecisionTreeClassifier
import joblib

def main():
    print("Ingesting data...")
    url = "https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv"
    df = pd.read_csv(url)
    
    df['Age'] = df['Age'].fillna(df['Age'].median())
    df['Fare'] = df['Fare'].fillna(df['Fare'].median())
    df['Sex'] = df['Sex'].map({'male': 0, 'female': 1})
    
    features = ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare']
    X = df[features]
    y = df['Survived']
    
    print("Training Decision Tree Model...")
    model = DecisionTreeClassifier(max_depth=4, random_state=42)
    model.fit(X, y)
    
    joblib.dump(model, "titanic_model.joblib")
    print("Model serialized and saved as 'titanic_model.joblib' successfully!")

if __name__ == "__main__":
    main()
