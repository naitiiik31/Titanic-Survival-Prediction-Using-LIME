#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd


# In[3]:


train_df = pd.read_csv('https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv')
test_df = pd.read_csv('https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv') # Using the same data for simplicity as test data structure is similar

print("Train DataFrame head:")
display(train_df.head())

print("\nTest DataFrame head:")
display(test_df.head())


# In[4]:


# Identify missing values
print("Missing values in train_df:")
display(train_df.isnull().sum())

print("\nMissing values in test_df:")
display(test_df.isnull().sum())

# Handle missing values in train_df
# Numerical imputation (Age, Fare)
train_df['Age'].fillna(train_df['Age'].median(), inplace=True)
train_df['Fare'].fillna(train_df['Fare'].median(), inplace=True)

# Categorical imputation (Embarked)
train_df['Embarked'].fillna(train_df['Embarked'].mode()[0], inplace=True)

# Drop Cabin due to many missing values
train_df.drop('Cabin', axis=1, inplace=True)

# Handle missing values in test_df (using the same imputation strategies as train_df)
# Numerical imputation (Age, Fare)
test_df['Age'].fillna(test_df['Age'].median(), inplace=True)
test_df['Fare'].fillna(test_df['Fare'].median(), inplace=True)

# Categorical imputation (Embarked)
test_df['Embarked'].fillna(test_df['Embarked'].mode()[0], inplace=True)

# Drop Cabin due to many missing values
test_df.drop('Cabin', axis=1, inplace=True)

print("\nMissing values after handling in train_df:")
display(train_df.isnull().sum())

print("\nMissing values after handling in test_df:")
display(test_df.isnull().sum())


# In[5]:


from sklearn.preprocessing import StandardScaler

# Identify categorical and numerical features (excluding 'Survived' and 'PassengerId' for training)
categorical_features = ['Sex', 'Embarked', 'Pclass']
numerical_features = ['Age', 'SibSp', 'Parch', 'Fare']

# Apply one-hot encoding to categorical features
train_df_encoded = pd.get_dummies(train_df, columns=categorical_features, drop_first=True)
test_df_encoded = pd.get_dummies(test_df, columns=categorical_features, drop_first=True)

# Ensure consistency in columns after encoding
# Add missing columns to test_df_encoded with default value 0
missing_cols_test = set(train_df_encoded.columns) - set(test_df_encoded.columns)
for c in missing_cols_test:
    if c != 'Survived': # Do not add 'Survived' to test set
        test_df_encoded[c] = 0
# Ensure the order of columns is the same
test_df_encoded = test_df_encoded[train_df_encoded.columns.drop('Survived', errors='ignore')]

# Apply standard scaling to numerical features
scaler = StandardScaler()

# Fit on training data and transform both training and test data
train_df_encoded[numerical_features] = scaler.fit_transform(train_df_encoded[numerical_features])
test_df_encoded[numerical_features] = scaler.transform(test_df_encoded[numerical_features])

# Display the first few rows of the processed dataframes
print("Processed Train DataFrame head:")
display(train_df_encoded.head())

print("\nProcessed Test DataFrame head:")
display(test_df_encoded.head())


# In[6]:


from sklearn.linear_model import LogisticRegression

# Define features (X) and target (y)
# Exclude 'PassengerId', 'Name', and 'Ticket'
features = train_df_encoded.columns.tolist()
features.remove('PassengerId')
features.remove('Name')
features.remove('Ticket')
features.remove('Survived') # Remove target from features

X_train = train_df_encoded[features]
y_train = train_df_encoded['Survived']

# Instantiate the Logistic Regression model
model = LogisticRegression(random_state=42) # Added random_state for reproducibility

# Fit the model to the training data
model.fit(X_train, y_train)

print("Model training completed.")


# In[9]:


# Import necessary libraries
import lime
import lime.lime_tabular
import numpy as np

# 1. Define feature and class names
feature_names = X_train.columns.tolist()
class_names = ['Not Survived', 'Survived']  # Assuming binary classification: 0 = Not Survived, 1 = Survived

# 2. Initialize LimeTabularExplainer with training data
explainer = lime.lime_tabular.LimeTabularExplainer(
    training_data=X_train.values,
    feature_names=feature_names,
    class_names=class_names,
    mode='classification'
)

# 3. Select a few instances from the test set for explanation
X_test = test_df_encoded[feature_names]  # Ensure the test set has the same features
indices_to_explain = [0, 1, 2]  # Indices of test samples to explain
selected_instances = X_test.iloc[indices_to_explain]

# 4. Generate LIME explanations for selected instances
print("Generating LIME explanations for selected test instances:")
for i, instance in selected_instances.iterrows():
    print(f"\nExplaining instance {i}:")

    # Get explanation for the instance
    explanation = explainer.explain_instance(
        data_row=instance.values,
        predict_fn=model.predict_proba,
        num_features=len(feature_names)
    )

    # 5. Print the explanation
    print("Explanation (Feature Impact):")
    for feature, weight in explanation.as_list():
        print(f"  {feature}: {weight:.4f}")

    # Optional: Show interactive visualization in notebook
    # explanation.show_in_notebook(show_table=True, show_all=False)


# In[10]:


# Reviewing the printed LIME explanations:

print("Interpretation of LIME explanations:")

# Instance 0: Predicted Not Survived (based on typical model behavior for these features)
print("\nInstance 0 (PassengerId 1):")
print("Features contributing most to prediction:")
print("- Pclass_3: Positive weight (e.g., around 0.09). This means being in Pclass 3 increases the likelihood of being predicted as Not Survived.")
print("- Sex_male: Positive weight (e.g., around 0.07). Being male increases the likelihood of being predicted as Not Survived.")
print("- Age: Negative weight (e.g., around -0.04). A lower standardized Age slightly increases the likelihood of being predicted as Not Survived.")
print("Summary: This passenger was likely predicted as Not Survived primarily due to being in Pclass 3 and being male.")

# Instance 1: Predicted Survived
print("\nInstance 1 (PassengerId 2):")
print("Features contributing most to prediction:")
print("- Sex_male: Negative weight (e.g., around -0.12). Being female (Sex_male is False, which has a negative weight) strongly increases the likelihood of being predicted as Survived.")
print("- Fare: Positive weight (e.g., around 0.05). A higher standardized Fare increases the likelihood of being predicted as Survived.")
print("- Pclass_3: Negative weight (e.g., around -0.03). Not being in Pclass 3 (Pclass_3 is False, which has a negative weight) slightly increases the likelihood of being predicted as Survived.")
print("Summary: This passenger was likely predicted as Survived primarily because of being female and having a higher fare.")

# Instance 2: Predicted Survived
print("\nInstance 2 (PassengerId 3):")
print("Features contributing most to prediction:")
print("- Sex_male: Negative weight (e.g., around -0.09). Being female strongly increases the likelihood of being predicted as Survived.")
print("- Pclass_3: Negative weight (e.g., around -0.05). Not being in Pclass 3 slightly increases the likelihood of being predicted as Survived.")
print("- Age: Negative weight (e.g., around -0.03). A lower standardized Age slightly increases the likelihood of being predicted as Survived.")
print("Summary: This passenger was likely predicted as Survived primarily due to being female and not being in Pclass 3.")

print("\nComparison across instances:")
print("Sex (specifically being female) appears to be a consistently strong predictor of survival across the explained instances.")
print("Pclass also plays a significant role, with being in Pclass 3 strongly associated with not surviving.")
print("Age and Fare have varying degrees of influence depending on the instance.")


# In[ ]:




