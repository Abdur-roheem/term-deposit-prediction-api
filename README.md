# 🚀 Term Deposit Subscription Prediction API  

## 📌 Overview  
This project builds and deploys a machine learning model to predict whether a customer will subscribe to a term deposit based on historical marketing data.  

It demonstrates an **end-to-end machine learning pipeline** from data exploration and model training to deployment as a REST API—designed to support **data-driven business decision-making**.


## 🎯 Problem Statement  
Marketing campaigns often target a broad audience, leading to **low conversion rates and wasted resources**.  

This project aims to:
- Identify customers most likely to subscribe  
- Improve targeting efficiency  
- Reduce operational and marketing costs  


## 💡 Solution  
A classification model was developed using Python and deployed as an API that allows real-time predictions based on customer data.


## 🧠 Project Structure  
```
term-deposit-prediction/
│
├── data/ # Raw and processed datasets
├── notebooks/ # EDA and experimentation
│ ├── 01-eda.ipynb
│ └── 02-modeling.ipynb
│
├── src/ # Core ML logic
│ ├── train.py # Model training pipeline
│ ├── predict.py # Prediction logic
│ └── utils.py # Helper functions
│
├── model/ # Saved trained model
│ └── model.pkl
│
├── app/ # API layer
│ └── app.py
│
├── requirements.txt # Project dependencies
├── Dockerfile # Containerization setup
└── README.md
```


## ⚙️ Tech Stack  
- **Python**  
- **pandas, NumPy**  
- **scikit-learn**  
- **FastAPI**  
- **Uvicorn**  
- **Docker**  


## 📊 Model Performance  
- **ROC-AUC Score:** `0.92`  
- Demonstrates strong ability to distinguish between subscribing and non-subscribing customers  


## 🔍 Approach  

### 1. Data Preparation  
- Data cleaning and preprocessing  
- Handling missing values  
- Feature engineering and encoding  

### 2. Model Development  
- Trained classification models using `scikit-learn`  
- Evaluated using ROC-AUC and other metrics  
- Selected best-performing model  

### 3. Deployment  
- Model saved using `pickle`  
- Exposed via REST API using FastAPI  
- Containerised using Docker for reproducibility  


## 🚀 Running the Project  

### 🔹 Option 1: Run Locally  

_#Clone repository_  
`git clone https://github.com/Abdur-roheem/machine-learning-zoomcamp-homework.git`

_Move into project_  
`cd machine-learning-zoomcamp-homework/Term\ Subscription\ deposit`

_Create virtual environment_  
`python -m venv venv`

_Activate environment_  
`venv\Scripts\activate`        # Windows  
`source venv/bin/activate`     # Mac/Linux

_Install dependencies_  
`pip install -r requirements.txt`

_Run API_  
`uvicorn app.app:app --reload`  

### 🔹 Option 2: Run with Docker

_#Build Docker image_  
`docker build -t term-deposit-app .`

_#Run container_  
`docker run -p 9696:9696 term-deposit-app`


**👉 API will be available at:**  
`http://localhost:9696`


### 📡 API Usage

### Endpoint
`POST /predict`

**Example Request**
```
{
  "age": 35,
  "job": "management",
  "balance": 5000,
  "housing": "yes"
}
```
**Example Response**
```
{
  "prediction": 1
}
```

## 💼 Business Impact

This solution can help organizations:

- Improve marketing ROI
- Increase customer conversion rates
- Reduce unnecessary outreach efforts
- Enable data-driven decision-making

🔗 Project Link

👉 https://github.com/Abdur-roheem/machine-learning-zoomcamp-homework/tree/main/Term%20Subscription%20deposit

👤 Author

**Roheem Ogunmakin**
+ Statistics Graduate
+ Data Science & Machine Learning Enthusiast
