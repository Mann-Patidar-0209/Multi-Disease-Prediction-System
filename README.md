# 🩺 Multi-Disease Prediction System

A Machine Learning-based web application that predicts the possibility of multiple diseases using patient-provided medical parameters.

## 📌 Project Overview

The Multi-Disease Prediction System is a web-based application developed using Python, Machine Learning, and Streamlit.

It uses trained Machine Learning models to predict four different diseases through a simple and interactive user interface.

### Diseases Covered

- ❤️ Heart Disease Prediction
- 🩸 Diabetes Prediction
- 🎗️ Breast Cancer Prediction
- 🧠 Parkinson's Disease Prediction

## 🚀 Features

- Predicts four different diseases.
- Uses pre-trained Machine Learning models.
- Interactive web interface built with Streamlit.
- User-friendly input forms.
- Fast prediction results.
- Separate trained model for each disease.

## 🛠️ Technologies Used

| Technology | Purpose |
|---|---|
| Python | Programming Language |
| Streamlit | Web Application |
| Scikit-learn | Machine Learning |
| NumPy | Numerical Computing |
| Pandas | Data Processing |
| Pickle | Model Serialization |
| Git & GitHub | Version Control |

## 📂 Project Structure

```text
Multi-Disease-Prediction-System/
│
├── app.py
├── requirements.txt
├── .gitignore
│
├── heart_disease_model.sav
├── diabetes_model.sav
├── breast_cancer_model.sav
└── Parkinsons_model.sav
```

## ⚙️ Installation and Setup

### 1. Clone the repository

```bash
git clone https://github.com/Mann-Patidar-0209/Multi-Disease-Prediction-System.git
```

### 2. Navigate to the project directory

```bash
cd Multi-Disease-Prediction-System
```

### 3. Create a virtual environment (optional)

```bash
python -m venv venv
```

### 4. Activate the virtual environment

For Windows:

```bash
venv\Scripts\activate
```

### 5. Install dependencies

```bash
pip install -r requirements.txt
```

### 6. Run the application

```bash
python -m streamlit run app.py
```

### 7. Open in your browser

```text
http://localhost:8501
```

## 🧠 Machine Learning

The application uses pre-trained Machine Learning models to perform disease predictions.

The trained models are saved as `.sav` files and loaded into the Streamlit application for making predictions.

Machine Learning techniques used in the project include:

- Logistic Regression
- Support Vector Machine (SVM)
- Feature Scaling

## 🔮 Future Improvements

- Improve prediction accuracy through model optimization.
- Enhance the user interface.
- Deploy the application online.
- Add more disease prediction models.
- Integrate additional healthcare-related features.

## ⚠️ Disclaimer

This application is developed for educational and project purposes only. Predictions are not a substitute for professional medical diagnosis or advice.

## 👨‍💻 Author

**Mann Patidar**

GitHub: [Mann-Patidar-0209](https://github.com/Mann-Patidar-0209)

---

⭐ If you find this project interesting, consider giving it a star!