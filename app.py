import pandas as pd
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# Set Streamlit page configuration
st.set_page_config(page_title="Titanic Data Analysis", layout="wide")

# Function to load data
@st.cache_data
def load_data():
    url = "https://raw.githubusercontent.com/JerryDoriquez/AJA_Final-Project-Data-Analysis-Techniques/refs/heads/main/titanic_dataset.csv"  
    data = pd.read_csv(url)
    return data

# Load data
data = load_data()

# Sidebar for navigation
st.sidebar.title("Navigation")
options = st.sidebar.radio("Go to:", ['Overview', 'Data Exploration', 'Clustering Analysis', 'Conclusions'])

# Overview Section
if options == 'Overview':
    st.title("Titanic Dataset Analysis")
    st.write("""
    This application analyzes the Titanic dataset, focusing on clustering passengers based on their survival, demographics, and other features using the K-means algorithm.
    """)
    st.subheader("Dataset Structure")
    st.write("**Sample Data**")
    st.write(data.head())

    st.subheader("Column Descriptions")
    st.write("""
    - **Survived**: Survival (0 = No, 1 = Yes)
    - **Pclass**: Passenger class (1 = First, 2 = Second, 3 = Third)
    - **Sex**: Gender of the passenger
    - **Age**: Age of the passenger
    - **SibSp**: Number of siblings/spouses aboard
    - **Parch**: Number of parents/children aboard
    - **Fare**: Passenger fare
    """)

# Data Exploration and Preparation Section
elif options == 'Data Exploration':
    st.title("Data Exploration and Preparation")
    st.subheader("Handling Missing Values")
    st.write("Checking for missing values:")
    st.write(data.isnull().sum())
    
    # Handling missing values
    st.write("**Filling missing values in 'Age' and 'Fare', and dropping rows with missing 'Embarked':**")
    data['Age'].fillna(data['Age'].median(), inplace=True)
    data['Fare'].fillna(data['Fare'].median(), inplace=True)
    data.dropna(subset=['Embarked'], inplace=True)
    st.write("Missing values after cleaning:")
    st.write(data.isnull().sum())

    st.subheader("Descriptive Statistics")
    st.write(data.describe())

    st.subheader("Data Distribution")
    fig, ax = plt.subplots(1, 2, figsize=(15, 5))
    sns.countplot(x='Pclass', hue='Survived', data=data, ax=ax[0])
    ax[0].set_title('Survival by Class')
    sns.histplot(data['Age'], bins=10, kde=True, ax=ax[1])
    ax[1].set_title('Age Distribution')
    st.pyplot(fig)

# Clustering Analysis Section
elif options == 'Clustering Analysis':
    st.title("Clustering Analysis")
    st.subheader("Feature Preparation")
    st.write("Encoding categorical variables and scaling numeric features for clustering.")
    
    # Encoding categorical variables
    data['Sex'] = data['Sex'].map({'male': 0, 'female': 1})
    features = ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare']
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(data[features])

    st.subheader("Determining Optimal Number of Clusters")
    inertia = []
    K = range(1, 11)
    for k in K:
        kmeans = KMeans(n_clusters=k, random_state=42)
        kmeans.fit(scaled_features)
        inertia.append(kmeans.inertia_)
    fig, ax = plt.subplots()
    ax.plot(K, inertia, 'bo-')
    ax.set_xlabel('Number of clusters')
    ax.set_ylabel('Inertia')
    ax.set_title('Elbow Method For Optimal k')
    st.pyplot(fig)

    st.subheader("Applying K-means Clustering")
    optimal_k = st.slider('Select the number of clusters', 2, 10, 3)
    kmeans = KMeans(n_clusters=optimal_k, random_state=42)
    clusters = kmeans.fit_predict(scaled_features)
    data['Cluster'] = clusters
    st.write(data[['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Cluster']].head())

    st.subheader("Cluster Visualization")
    pca = PCA(n_components=2)
    principal_components = pca.fit_transform(scaled_features)
    principal_df = pd.DataFrame(data=principal_components, columns=['PC1', 'PC2'])
    principal_df['Cluster'] = clusters
    fig, ax = plt.subplots()
    sns.scatterplot(data=principal_df, x='PC1', y='PC2', hue='Cluster', palette='tab10', ax=ax)
    ax.set_title('Clusters Visualization (PCA)')
    st.pyplot(fig)

# Conclusions Section
elif options == 'Conclusions':
    st.title("Conclusions and Recommendations")
    st.write("""
    **Key Takeaways:**
    - Passengers are grouped into clusters based on demographic and survival-related features.
    - Clustering reveals patterns, such as certain clusters having higher survival rates.
    - These insights can aid in understanding the characteristics of passengers who survived or perished.

    **Actionable Recommendations:**
    - Apply similar clustering techniques to other historical datasets to uncover patterns.
    - Explore additional data preprocessing steps, such as handling outliers or feature engineering.
    """)

    st.sidebar.markdown("---")
    st.sidebar.markdown("**Developed by:** AJA ")
    st.sidebar.markdown("**Project:** Titanic Data Analysis")
