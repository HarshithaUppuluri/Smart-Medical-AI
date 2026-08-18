# MediAssist AI

## Smart Disease Prediction and Patient Support System

MediAssist AI is an educational AI-powered healthcare web application developed for IntelliGen. The system demonstrates how different Artificial Intelligence and Machine Learning techniques can be integrated into a single user-friendly application to support disease prediction and patient information.

The application is designed for **educational and demonstration purposes only** and is not intended to replace professional medical diagnosis or advice.

---

## 🎯 Project Objectives

The main objectives of this project are to:

* Apply Artificial Intelligence and Machine Learning techniques to a healthcare-related problem.
* Predict possible diseases based on user-provided symptoms.
* Analyse skin-condition images using deep learning.
* Analyse patient feedback using sentiment analysis.
* Provide explainability for machine learning predictions.
* Demonstrate an advanced AI feature beyond the core module content.
* Present AI functionality through an accessible web application.

---

## 🤖 AI and Machine Learning Features

### 1. Disease Classification

Machine learning classification is used to predict possible diseases based on symptom information.

The classification component demonstrates the complete machine learning process, including:

* Data preprocessing
* Feature preparation
* Model training
* Model testing
* Performance evaluation
* Disease prediction

---

### 2. Natural Language Processing (NLP)

The application allows users to describe their symptoms using natural language.

For example:

> "I have a headache, high temperature and feel very tired."

The NLP component processes the user's text and uses it to generate a predicted condition.

This provides a more natural method of interacting with the system compared with manually selecting individual symptoms.

---

### 3. Deep Learning — Skin Image Analysis

The application includes a deep learning component for analysing uploaded skin-condition images.

Users can upload an image, and the trained image classification model processes the image and returns a predicted skin-condition class.

This demonstrates the use of neural networks and deep learning for image-based healthcare applications.

---

### 4. Sentiment Analysis

A sentiment analysis feature is included to analyse patient feedback.

Feedback can be classified according to its sentiment, demonstrating how AI can be used to understand user experiences and opinions.

This could potentially help healthcare organisations analyse large quantities of patient feedback more efficiently.

---

### 5. Explainable AI (XAI)

Explainable AI is incorporated into the project to improve understanding of machine learning predictions.

**SHAP (SHapley Additive exPlanations)** is used to investigate how different input features influence model predictions.

Explainability is particularly important when considering healthcare AI because users and organisations should be able to understand why an AI system has produced a particular result.

---

## 🚀 Advanced AI Feature

The project also incorporates an advanced AI/ML feature beyond the core techniques covered in the module.

**Advanced feature: Optical Character Recognition (OCR)

The advanced component demonstrates independent research and extends the functionality of the Smart Medical AI system.

It is implemented as the advanced AI feature of the application. OCR allows users to upload an image containing text, such as a medical prescription or document and automatically extracts the readable text from the image.

This feature extends the application beyond the AI/ML techniques covered in the module and demonstrates how computer vision can be used to convert image-based text into digital information for further processing.
---

## 📊 Data Processing and Exploratory Data Analysis

Before training the machine learning models, the datasets are prepared and explored.

The project includes:

* Data loading
* Data cleaning
* Handling and checking data quality
* Exploratory Data Analysis (EDA)
* Feature preparation
* Model-ready dataset generation

The main data processing code can be found in:

```text
data_cleansing_and_eda.py
```

---

## 🧠 Model Training and Testing

The machine learning components are trained and evaluated using the project's training pipeline.

The main training and testing implementation can be found in:

```text
model_training_testing.py
```

The project evaluates the trained models to determine how effectively they perform on unseen data.

---

## 🔍 Model Explainability

Explainability functionality is implemented in:

```text
explainability.py
```

Generated evaluation and explainability outputs are stored in:

```text
reports_assets/
```

---

## 🌐 Web Application

The AI components are integrated into a web application to provide an accessible interface for demonstrating the system.

The main application file is:

```text
app.py
```

Web page templates are stored in:

```text
templates/
```

Static application resources are stored in:

```text
static/
```

---

## 📁 Project Structure

```text
Smart-Medical-AI/
│
├── data/
│   ├── raw/
│   └── processed/
│
├── legacy/
│
├── models/
│
├── reports_assets/
│
├── static/
│
├── templates/
│
├── app.py
├── config.py
├── data_cleansing_and_eda.py
├── download_data.py
├── explainability.py
├── main.py
├── model_training_testing.py
├── project_overview.txt
├── requirements.txt
├── Smart_Disease_Prediction.ipynb
├── test.py
└── README.md
```

---

## ⚙️ Installation

### 1. Clone the Repository

Clone the project from GitHub:

```bash
git clone https://github.com/HarshithaUppuluri/Smart-Medical-AI.git
```

Move into the project directory:

```bash
cd Smart-Medical-AI
```

### 2. Create a Virtual Environment

On Windows:

```bash
python -m venv .venv
```

Activate it using:

```bash
.venv\Scripts\activate
```

### 3. Install Dependencies

Install the required Python libraries:

```bash
pip install -r requirements.txt
```

---

## ▶️ Running the Application

Run the application from the project directory using:

```bash
python app.py
```

After starting the application, the terminal will display the local address used to access the web application.

For example:

```text
http://127.0.0.1:5000
```

Open the displayed address in a web browser.

---

## 💻 Expected Application Output

The web application provides interfaces for demonstrating the project's AI functionality, including:

* Disease prediction from symptoms
* Natural-language symptom analysis
* Skin image analysis
* Patient feedback sentiment analysis
* AI prediction results and confidence information
* Explainability and supporting model information
* Advanced AI functionality

---

## 🛠️ Technologies Used

The project uses Python and a range of AI, machine learning and data-analysis technologies, including:

* Python
* Pandas
* NumPy
* Scikit-learn
* Deep Learning
* Natural Language Processing
* SHAP
* HTML
* CSS
* Flask
* Jupyter Notebook
* Git
* GitHub

Additional libraries required to reproduce the project are listed in:

```text
requirements.txt
```

---

## ⚖️ Ethical Considerations

Healthcare applications using Artificial Intelligence require careful consideration of issues such as:

* Patient privacy
* Data protection
* Bias in training data
* Fairness
* Model accuracy
* Transparency
* Explainability
* Responsible use of AI

Predictions produced by this project should therefore be interpreted as demonstrations of AI techniques rather than medical diagnoses.

---

## ⚠️ Disclaimer

**This application is an educational prototype and is not a medical device.**

The predictions and information generated by the application are provided for educational and demonstration purposes only.

They should **not** be used as a substitute for diagnosis, treatment or advice from a qualified healthcare professional.

---

## 📚 Academic Project

This repository was created as part of the **Programming for Artificial Intelligence** assessment for IntelliGen.

The project demonstrates the practical application of multiple AI/ML techniques through an integrated healthcare application.

The GitHub repository also maintains the project's commit history to document its development process.

---

## 📄 Licence

This project was developed for educational and academic purposes.
