import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st

# Load the dataset
file_path = 'data/exercice_data.csv'
df = pd.read_csv(file_path, encoding='latin1')


# Select features and target variable
X = df[['absences', 'failures', 'studytime', 'Medu', 'Fedu', 'famrel', 'age', 'traveltime', 'freetime']]
y = df['FinalGrade']

# Standardize the features
scaler = MinMaxScaler(feature_range=(0, 4))
X_scaled = scaler.fit_transform(X)

# Split the df into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Create and train the Linear Regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Predict the grades for the test df
y_pred = model.predict(X_test)

# Get the regression coefficients
coefficients = model.coef_

# Create a Streamlit app
st.title("Student Performance Predictor")

# Display the coefficients
st.subheader("Feature Weights (Coefficients):")
features = ['Absences', 'Failures', 'Study Time', 'Mother\'s Education', 'Father\'s Education', 'Family Relationship', 'Age', 'Travel Time', 'Free Time']
coef_df = pd.DataFrame({'Feature': features, 'Coefficient': coefficients})
st.write(coef_df)

# Evaluate the model's performance
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

st.subheader("Model Performance Metrics")
st.write(f"Mean Squared Error (MSE): {mse}")
st.write(f"R-squared (R2 score): {r2}")

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

# Load the dataset
file_path = 'data/exercice_data.csv'
df = pd.read_csv(file_path, encoding='latin1')

# Select features and target variable
X = df[['absences', 'failures', 'studytime', 'Medu', 'Fedu', 'famrel', 'age', 'traveltime', 'freetime']]
y = df['FinalGrade']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create and train the Linear Regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Get the regression coefficients
coefficients = model.coef_

# Create a dictionary of weights for the complexity calculation
weights = {
    'absences': -coefficients[0],
    'failures': -coefficients[1],
    'studytime': coefficients[2],
    'Medu': coefficients[3],
    'Fedu': coefficients[4], 
    'famrel': coefficients[5], 
    'age': -coefficients[6],  
    'traveltime': -coefficients[7], 
    'freetime': -coefficients[8]
}

# Define the complexity calculation function
def calculate_complexity(row):
    return (
        row['absences'] * weights['absences'] +
        row['failures'] * weights['failures'] +
        (4 - row['famrel']) * weights['famrel'] +  # Higher value is better, inverted
        (4 - row['Medu']) * weights['Medu'] +      # Higher value is better, inverted
        (4 - row['Fedu']) * weights['Fedu'] +      # Higher value is better, inverted
        (4 - row['studytime']) * weights['studytime'] +  # Assuming studytime is 1 to 4
        (4 - row['age']) * weights['age'] +       # Adjust based on understanding
        (2 if row['famsize'] == 'GT3' else 0) +    # Add for larger family size
        (4 - row['traveltime']) * weights['traveltime'] +  # Adjust if needed
        (4 - row['freetime']) * weights['freetime']  # Adjust if needed
    )


# Add complexity column to df
df['complexity'] = df.apply(calculate_complexity, axis=1)

# Add a slider to let users adjust the complexity threshold
complexity_threshold = st.slider('Complexity Threshold', min_value=0, max_value=10, value=1)

# Filter students needing help (low final grade and high complexity)
students_needing_help = df[(df['complexity'] > complexity_threshold) & (df['FinalGrade'] < 10)]

st.subheader(f"Students Needing Help (Complexity > {complexity_threshold} and Final Grade < 10)")
st.write(students_needing_help[['StudentID', 'FirstName', 'FamilyName', 'FinalGrade', 'complexity']])

# Visualizations
st.subheader("Complexity vs Final Grade")

# Plot the scatter plot of complexity vs final grade
plt.figure(figsize=(10, 6))
sns.scatterplot(x='FinalGrade', y='complexity', data=df, hue='complexity', palette='coolwarm', size='complexity')
plt.axhline(y=complexity_threshold, color='red', linestyle='--', label=f'Threshold: {complexity_threshold}')
plt.title('Final Grade vs Complexity')
plt.xlabel('Final Grade')
plt.ylabel('Complexity')
st.pyplot(plt)

# Save students needing help to CSV
if st.button("Download List of Students Needing Help"):
    students_needing_help.to_csv("students_needing_help.csv", index=False)
    st.success("File saved as 'students_needing_help.csv'.")

# Option to visualize distribution of complexity
if st.checkbox("Show Complexity Distribution"):
    plt.figure(figsize=(10, 6))
    sns.histplot(df['complexity'], kde=True, bins=15)
    plt.title('Distribution of Complexity Scores')
    plt.xlabel('Complexity Score')
    plt.ylabel('Frequency')
    st.pyplot(plt)

# Option to visualize Final Grade vs Complexity by Parental Education
if st.checkbox("Show Correlation Between Parental Education and Complexity"):
    # Filter data for mother
    mother_data = df[df['guardian'] == 'mother']
    
    plt.figure(figsize=(10, 6))
    sns.boxplot(x='Medu', y='complexity', data=mother_data)
    plt.title('Mother\'s Education vs Complexity')
    plt.xlabel('Mother\'s Education Level')
    plt.ylabel('Complexity')
    st.pyplot(plt)

    # Filter data for father
    father_data = df[df['guardian'] == 'father']

    plt.figure(figsize=(10, 6))
    sns.boxplot(x='Fedu', y='complexity', data=father_data)
    plt.title('Father\'s Education vs Complexity')
    plt.xlabel('Father\'s Education Level')
    plt.ylabel('Complexity')
    st.pyplot(plt)
