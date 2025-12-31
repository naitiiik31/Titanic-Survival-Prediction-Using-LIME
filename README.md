# 🚢 Titanic Survival Prediction Using LIME

## 📌 Overview
The Titanic Survival Prediction project uses Machine Learning to predict whether a passenger survived the Titanic disaster based on features like age, gender, passenger class, and fare.  
This project also integrates **LIME (Local Interpretable Model-agnostic Explanations)** to explain individual predictions, making the model interpretable and transparent.  
Users can understand not only the prediction but also the key factors that influenced each decision.

---

## 🚀 Features
- Predict survival of Titanic passengers using ML  
- Uses data preprocessing and feature engineering  
- Integrates LIME for model interpretation  
- Provides insight into how predictions are made  
- Helpful for understanding model decisions  
- Easy to run locally in a notebook

---

## 🧠 Machine Learning Workflow
- Load the Titanic dataset (CSV)  
- Handle missing values and clean data  
- Encode categorical variables  
- Split data into training and testing sets  
- Train a classification model (e.g., Logistic Regression / Random Forest)  
- Evaluate the model using accuracy and other metrics  
- Apply LIME to explain predictions for individual examples

---

## 🛠️ Tech Stack
- Programming Language: Python  
- Libraries: Pandas, NumPy, Scikit-Learn, LIME  
- Model: Supervised Classification  
- Dataset: Titanic Passenger Dataset (CSV)

---

## 🔧 How to Run the Project
- Clone the repository  
  git clone https://github.com/naitiiik31/Titanic-Survival-Prediction-Using-LIME

- Install required dependencies  
  pip install -r requirements.txt

- Open and run the Jupyter Notebook  
  Open `Titanic.ipynb` in Jupyter Notebook / VSCode

---

## 🧪 Sample Output
- Input: Passenger: Female, Age: 28, Class: 1st  
  Output: **Survived**

- Input: Passenger: Male, Age: 45, Class: 3rd  
  Output: **Not Survived**

---

## 📈 Model Performance
- High predictive accuracy on test set  
- Key features like sex, class, and age significantly impact survival  
- LIME explanations help reveal the model’s decision process

---

## 🔮 Future Scope
- Deploy as a web app with UI (Streamlit / Flask)  
- Add performance dashboards and charts  
- Incorporate multiple ML algorithms & model comparison  
- Add user input interface for real-time predictions  
- Enhance LIME visual explanations

---

## 👤 Author
Naitikkumar Patel

---

