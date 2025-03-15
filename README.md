# AI-Driven B2B Lead Scoring

## Overview
This project implements an AI-driven B2B lead scoring system using machine learning. It utilizes a Random Forest classifier to assess and categorize leads based on company size, revenue, engagement score, and industry type.

## Features
- Data preprocessing and feature engineering
- Machine learning model training (Random Forest Classifier)
- Lead score prediction
- Model evaluation and performance reporting
- Model persistence using Joblib

## Technologies Used
- Python
- Pandas
- NumPy
- Scikit-learn
- Joblib

## Installation
1. Clone the repository:
   ```sh
   git clone https://github.com/your-repo/ai-b2b-lead-scoring.git
   ```
2. Navigate to the project directory:
   ```sh
   cd ai-b2b-lead-scoring
   ```
3. Install dependencies:
   ```sh
   pip install -r requirements.txt
   ```

## Usage
### Training the Model
Run the script to train the lead scoring model:
```sh
python ai_b2b_lead_scoring.py
```

### Predicting Lead Scores
Use the following function to predict a lead score:
```python
from ai_b2b_lead_scoring import predict_lead_score

result = predict_lead_score(company_size=300, revenue=5000000, engagement_score=8, industry=2)
print(result)  # Output: High Potential or Low Potential
```

## Model Evaluation
After training, the model evaluates accuracy and generates a classification report.

## Saving & Loading the Model
The trained model is saved as `b2b_lead_scoring_model.pkl` using Joblib. You can load and reuse it for predictions.

## License
This project is licensed under the MIT License.

## Author
Your Name

For any inquiries, feel free to reach out!

