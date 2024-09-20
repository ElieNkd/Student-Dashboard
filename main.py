import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st

# Load the dataset (make sure the file is in the correct folder)
file_path = 'data/exercice_data.csv'

# Read the CSV file using pandas
data = pd.read_csv(file_path,encoding='latin1')

# Streamlit title and description
st.title("Student Support Dashboard")
st.write("This dashboard helps to prioritize students for support based on complexity and final grades.")

# Show data (for debugging or user exploration)
if st.checkbox("Show raw data"):
    st.write(data.head())

def calculate_complexity(row):
    complexity = (
        0.2 * row['absences'] +
        5 * row['failures'] +
        (5 - row['famrel']) +
        (4 - row['Medu']) +
        (4 - row['Fedu']) +
        (5 - row['studytime']) +  # Assuming studytime is 1 to 4
        (24- row['age']) +  # Age might impact the complexity
        (2 if row['famsize'] == 'GT3' else 0) +  # Add 2 for larger family size
        row['traveltime'] +  # Higher travel time might add to complexity
        (5 - row['freetime'])      # Less free time could increase complexity
    )
    return complexity

# Add complexity column to data
data['complexity'] = data.apply(calculate_complexity, axis=1)

# Add a slider to let users adjust the complexity threshold
complexity_threshold = st.slider('Complexity Threshold', min_value=20, max_value=100, value=30)

# Filter students needing help (low final grade and high complexity)
students_needing_help = data[(data['complexity'] > complexity_threshold) & (data['FinalGrade'] < 10)]

st.subheader(f"Students Needing Help (Complexity > {complexity_threshold} and Final Grade < 10)")
st.write(students_needing_help[['StudentID','FirstName','FamilyName','FinalGrade','complexity']])

# Visualizations
st.subheader("Complexity vs Final Grade")

# Plot the scatter plot of complexity vs final grade
plt.figure(figsize=(10, 6))
sns.scatterplot(x='FinalGrade', y='complexity', data=data, hue='complexity', palette='coolwarm', size='complexity')
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
    sns.histplot(data['complexity'], kde=True, bins=15)
    plt.title('Distribution of Complexity Scores')
    plt.xlabel('Complexity Score')
    plt.ylabel('Frequency')
    st.pyplot(plt)

# Option to visualize Final Grade vs Complexity by Parental Education
if st.checkbox("Show Correlation Between Parental Education and Complexity"):
    # Filter data for mother
    mother_data = data[data['guardian'] == 'mother']

    plt.figure(figsize=(10, 6))
    sns.boxplot(x='Medu', y='complexity', data=data)
    plt.title('Mother\'s Education vs Complexity')
    plt.xlabel('Mother\'s Education Level')
    plt.ylabel('Complexity')
    st.pyplot(plt)

    # Filter data for father
    father_data = data[data['guardian'] == 'father']


    plt.figure(figsize=(10, 6))
    sns.boxplot(x='Fedu', y='complexity', data=data)
    plt.title('Father\'s Education vs Complexity')
    plt.xlabel('Father\'s Education Level')
    plt.ylabel('Complexity')
    st.pyplot(plt)
