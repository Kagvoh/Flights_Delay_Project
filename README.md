# Flights Delay Prediction Project

This is my first Machine Learning project to predict airline flight delays. The model is trained using a Jupyter Notebook, and its performance is demonstrated through a web application built with FastAPI and Streamlit.

# A Machine Learning project for predicting airline flight delays using multiple classification models.

This project provides:
     - FastAPI Backend → serves trained ML models through REST APIs.
     - Streamlit Frontend → interactive web interface for user prediction.
     - Machine Learning Models → trained using Python and Scikit-learn.

Dataset: https://www.kaggle.com/datasets/giovamata/airlinedelaycauses

# Project Structure:
```
        Flights_Delay_Project/
        │
        ├── Backend/
        │   ├── models/                # Saved ML models (.pkl)
        │   ├── FD_api.py              # FastAPI server
        │   └── __pycache__/
        │
        ├── Frontend/
        │   └── FD_streamlit.py        # Streamlit UI
        │
        ├── Notebook - Model Inference/
        │   └── Delayed_Flights.ipynb  # Model training notebook
        │
        ├── README.md
        ├── LICENSE
        └── .gitignore
```
Instructions: You need to create this folder "Flight_Delay_Project/Backend/Models" first, then when you run Notebook File, the .pkl files will be loaded into that folder.

# Technologies Used
# Backend
```
    - Python
    - FastAPI
    - Uvicorn
    - Scikit-learn
    - Pandas
    - NumPy
    - Joblib
```
# Frontend
```
  - Streamlit
  - Requests
  - Machine Learning
  - Logistic Regression
  - Random Forest
  - XGBoost / Gradient Boosting (if used)
```
# Requirements
Before running the project, make sure you have:
Python 3.10+ installed
pip installed
Git installed (optional)

# Installation Guide
Clone Repository
```bash
    git clone https://github.com/Kagvoh/Flights_Delay_Project.git
    Move into project folder: cd Flights_Delay_Project
```

# FastAPI Backend

What is FastAPI?
FastAPI is a modern Python framework used to build high-performance REST APIs quickly and efficiently.
In this project, FastAPI is responsible for:
- Loading trained ML models
- Receiving user input
- Performing predictions
- Returning prediction results as JSON

# Instructions:
Then, start the FastAPI app:
```bash
    uvicorn main:app --reload
```
The API documentation can be accessed at http://localhost:8000/docs.

# Streamlit Frontend

What is Streamlit?
Streamlit is a Python framework used to create interactive web applications for Machine Learning projects with minimal code.
In this project, Streamlit is used for:
- User input interface
- Sending requests to FastAPI
- Displaying prediction results
- Visualizing delay risk
- Finally, start the Streamlit app:
  ```bash
      streamlit run streamlit_app/main.py
  ```
# The web interface can be accessed at http://localhost:8501.

# Web demo Workflow 
```
    User Input
        ↓
    Streamlit Frontend
        ↓
    FastAPI Backend
        ↓
    Machine Learning Model
        ↓
    Prediction Result
        ↓
    Display on Streamlit
```

# Overal Project Pipline:
```
Dataset -> Data Preprocessing -> Feature Engineering -> Model Training
                                                              ↓
                                 Save Trained Model <- Model Evaluation
                                        ↓
                                  FastAPI Backend
                                        ↓
                                Streamlit Frontend
                                        ↓
                                  User Prediction
```
