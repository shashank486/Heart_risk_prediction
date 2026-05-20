# ❤️ Heart Disease Prediction System Using Machine Learning

## 📌 Introduction

The Heart Disease Prediction System is a Machine Learning-based healthcare application developed to predict the risk of heart disease using patient medical data. 

The system analyzes important health parameters such as:
- Age
- Blood Pressure
- Cholesterol
- ECG Results
- Heart Rate
- Chest Pain Type
- Exercise Angina
- Oldpeak Value

The project uses Machine Learning algorithms to classify whether a patient is at:
- ✅ Low Risk
- 🚨 High Risk

The application is integrated with a Streamlit web interface that provides real-time prediction results along with a risk score and visualization.

---

# 🛠️ Technologies Used

- Python
- Pandas
- NumPy
- Scikit-learn
- Streamlit
- Matplotlib
- Seaborn
- Pickle

---

# 📂 Project Structure

heart_prediction/
│
├── app.py
├── gboost_model.pkl
├── requirements.txt
└── README.md

---

# ▶️ Steps to Run the Project

## ✅ Step 1: Clone or Download the Project

Move to project directory:

```bash
cd heart_prediction
```

## Step 2: Create Virtual Environment
For Windows
```bash
python -m venv venv
```
For macOS/Linux
```bash
python3 -m venv venv
```

## Step 3: Activate Virtual Environment

Windows
```bash
venv\Scripts\activate
```

macOS/Linux
```bash
source venv/bin/activate
```

## Step 4: Install Required Libraries
```bash
pip install streamlit pandas numpy scikit-learn matplotlib seaborn
```

# Step 5: Run the Streamlit Application

```bash
streamlit run app.py
```

# Test Cases

## High Risk Test Case

| Feature        | Value |
| -------------- | ----- |
| Age            | 65    |
| Gender         | M     |
| RestingBP      | 180   |
| Cholesterol    | 400   |
| FastingBS      | 1     |
| MaxHR          | 85    |
| Oldpeak        | 5     |
| ChestPainType  | ASY   |
| RestingECG     | LVH   |
| ExerciseAngina | Y     |
| ST_Slope       | flat  |


## Low Risk Test Case

| Feature        | Value  |
| -------------- | ------ |
| Age            | 25     |
| Gender         | F      |
| RestingBP      | 110    |
| Cholesterol    | 140    |
| FastingBS      | 0      |
| MaxHR          | 180    |
| Oldpeak        | 0      |
| ChestPainType  | ATA    |
| RestingECG     | Normal |
| ExerciseAngina | N      |
| ST_Slope       | Up     |


# Author
## Shashank B Hunagund
