import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import joblib

# Load dataset (dummy dataset for demonstration)
data = {
    'Company Size': [10, 200, 50, 500, 1000, 20, 300, 40, 700, 150],
    'Revenue': [100000, 2000000, 500000, 10000000, 50000000, 200000, 3000000, 400000, 7000000, 1500000],
    'Engagement Score': [5, 8, 6, 9, 10, 4, 7, 5, 9, 6],
    'Industry': [1, 2, 1, 3, 3, 1, 2, 1, 3, 2],  # Encoded industry types
    'Lead Score': [0, 1, 0, 1, 1, 0, 1, 0, 1, 1]  # 1 = High potential, 0 = Low potential
}

df = pd.DataFrame(data)

# Splitting data into training and testing sets
X = df.drop(columns=['Lead Score'])
y = df['Lead Score']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)

# Evaluate model
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)
print("Classification Report:\n", classification_report(y_test, y_pred))

# Save model
joblib.dump(model, 'b2b_lead_scoring_model.pkl')

# Function for lead scoring prediction
def predict_lead_score(company_size, revenue, engagement_score, industry):
    model = joblib.load('b2b_lead_scoring_model.pkl')
    input_data = np.array([[company_size, revenue, engagement_score, industry]])
    prediction = model.predict(input_data)
    return 'High Potential' if prediction[0] == 1 else 'Low Potential'

# Example usage
print(predict_lead_score(300, 5000000, 8, 2))
