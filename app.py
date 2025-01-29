import streamlit as st
import pandas as pd
import seaborn as sns
import plotly.express as px
import matplotlib.pyplot as plt
import pickle as pk
from wranglefuncation import wrangle

df = pd.read_csv("WA_Fn-UseC_-HR-Employee-Attrition.csv")


st.title(f"HR Attrition Analysis & Machine Learning model.")
website_link = "https://www.kaggle.com/datasets/pavansubhasht/ibm-hr-analytics-attrition-dataset"
st.write("The Dataset link: [IBM HR Attrition](%s)" % website_link)
st.sidebar.header("Navigation")
st.sidebar.markdown("Created by [Ahmed Yusri](https://www.linkedin.com/in/ahmed-yusri-499a67313)")
st.sidebar.image("attrition.png")

sidebar_option = st.sidebar.radio("Choose an Option:", ["Overview", "EDA", "Modeling", "Insights"])

if sidebar_option == "Overview":
    st.header("Data Overview")
    st.write(f"The Dataset contains {df.shape[0]} rows and {df.shape[1]} columns.")
    st.write(df.describe(include='all'))
    st.write("The percentage of the Attrition classes in the dataset.")
    st.write(df["Attrition"].value_counts())
    fig, ax = plt.subplots(figsize=(10, 5))
    sns.countplot(data=df, x="Attrition", ax=ax)
    plt.title("The percentage of the Attrition classes in the dataset.")
    st.pyplot(fig)
    
    
elif sidebar_option == "EDA":
    st.header("Exploratory Data Analysis")
    st.write("Scatter plot to see the imbalance in the classes and multicollinearity between the features.")
    columnX_options = st.selectbox("Select the X-axis feature", df[["YearsAtCompany", "TotalWorkingYears", "YearsInCurrentRole", "MonthlyIncome"]].columns[:2])
    columnY_options = st.selectbox("Select the Y-axis feature", df[["YearsAtCompany", "TotalWorkingYears", "YearsInCurrentRole", "MonthlyIncome"]].columns[2:])
    fig1 = px.scatter(df, columnX_options, columnY_options, facet_col="Attrition", title=f"{columnX_options} vs {columnY_options}")
    st.plotly_chart(fig1)
    st.write("So as i see the dataset is imbalance and the features are multicollinear.")
    st.markdown("### Heatmap plot to see the higher correlation between the features")
    corr_matrix = df.corr(numeric_only=True)
    fig2, ax = plt.subplots(figsize=(15, 10))
    sns.heatmap(corr_matrix[(corr_matrix >= 0.7) | (corr_matrix <= -0.7)], annot=True, cbar=False, cmap="coolwarm", ax=ax)
    plt.title("Heatmap plot to see the higher correlation between the features")
    st.pyplot(fig2)
     
elif sidebar_option == "Modeling":
    st.markdown("## HR Attrition classification model using `SVC` model.")
    user_age = st.slider("How old are you?", 25, 80, 40)
    user_gender = st.radio("choose gender",
    options=[
        "Male", "Female"],)
    
    user_BT = st.radio("Select your Business Travel situation",
    options=[
        "Non-Travel",
        "Travel_Rarely",
        "Travel_Frequently",
    ],)
    
    user_ms = st.radio("Select your Marital Status",
    options=[
        "Single",
        "Married",
        "Divorced",
    ],)
    
    user_dr = st.number_input('Enter your Daliy rate', min_value=0, max_value=50000, value=0, step=1)
    user_Ncomp = st.number_input('Enter your Number of Companies Worked', min_value=0, max_value=10, value=0, step=1)
    user_OverTime = st.radio("Do you work Over Time?",
    options=[
        "Yes",
        "No",
    ],)
    user_distHome = st.slider("How far from home to work?", 1, 30, 0)   
    data_dict = {
    'Age': user_age,
    'DailyRate': user_dr,
    'DistanceFromHome': user_distHome,
    'Education': df["Education"].mode()[0],
    'EnvironmentSatisfaction': df["EnvironmentSatisfaction"].mode()[0],
    'Gender': 1 if user_gender == "Male" else 0,
    'HourlyRate': df["HourlyRate"].mode()[0],
    'JobInvolvement': df["JobInvolvement"].mode()[0],
    'JobSatisfaction': df["JobSatisfaction"].mode()[0],
    'MonthlyRate': df["MonthlyRate"].mode()[0],
    'NumCompaniesWorked': user_Ncomp,
    'OverTime': 1 if user_OverTime == "Yes" else 0,
    'PercentSalaryHike': df["PercentSalaryHike"].mode()[0],
    'RelationshipSatisfaction': df["RelationshipSatisfaction"].mode()[0],
    'StockOptionLevel': df["StockOptionLevel"].mode()[0],
    'TotalWorkingYears': df["TotalWorkingYears"].mode()[0],
    'TrainingTimesLastYear': df["TrainingTimesLastYear"].mode()[0],
    'WorkLifeBalance': df["WorkLifeBalance"].mode()[0],
    'YearsSinceLastPromotion': df["YearsSinceLastPromotion"].mode()[0],
    'BusinessTravel_Travel_Frequently': 1 if user_BT == "Travel_Frequently" else 0,
    'BusinessTravel_Travel_Rarely': 1 if user_BT == "Travel_Rarely" else 0,
    'JobRole_Human Resources': 0,
    'JobRole_Laboratory Technician': 1,
    'JobRole_Manager': 0,
    'JobRole_Manufacturing Director': 0,
    'JobRole_Research Director': 0,
    'JobRole_Research Scientist': 0,
    'JobRole_Sales Executive': 0,
    'JobRole_Sales Representative': 0,
    'Department_Research & Development': 1,
    'Department_Sales': 0,
    'EducationField_Life Sciences': 1,
    'EducationField_Marketing': 0,
    'EducationField_Medical': 0,
    'EducationField_Other': 0,
    'EducationField_Technical Degree': 0,
    'MaritalStatus_Married': 1 if user_ms == "Married" else 0,
    'MaritalStatus_Single': 1 if user_ms == "Single" else 0}
    
    user_data = pd.DataFrame(data_dict, index=[0])

    X_test = user_data

    st.write(X_test)

    with open("best_svc_pipeline_model.pkl", "rb") as f:
            model = pk.load(f)
            
    if st.button("Predict"):
        y_pred = model.predict(X_test)
        pred = None
        if y_pred[0] == 0:
            pred = "No"
        else:
            pred = "Yes"    
        st.write(f"The model predict that the user will have `{pred}` attrition.")
 
elif sidebar_option == "Insights":
    st.subheader("My Insights after training and Deployment the model.")
    st.write("The Dataset is needed preprocessing and features engineering to get the best model.")
    st.write("The imbalance classes prediction need to imporved to prevent the model from baised.")
    st.write("The model need more features to get the best prediction.")      
