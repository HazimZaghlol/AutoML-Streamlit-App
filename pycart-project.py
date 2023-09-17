import streamlit as st
import pandas as pd
import missingno
import matplotlib.pyplot as plt
import numpy as np
from collections import Counter
import plotly.express as px

# Function to load data from a file and return a DataFrame
def load_data(file):
        df = pd.read_csv(file)  
        return df

# Function to display missing data and a missing data plot              
def miss_data(df):
    st.subheader("2. Missing Data")
    st.write(df.isnull().sum())
    missing_fig = plt.figure(figsize=(10, 5))
    missingno.bar(df, figsize=(10, 5), fontsize=12)
    st.pyplot(missing_fig, use_container_width=True)
    
# Function to perform a classification task on the provided DataFrame    
def perform_classification(df, target_column, numeric_imputation, categorical_imputation, normalize_method, normalize):
    setup(df, target=target_column, numeric_imputation=numeric_imputation, categorical_imputation=categorical_imputation, normalize_method=normalize_method, normalize=normalize, session_id=123) 
    setup_all = pull()
    st.dataframe(setup_all)
    best = compare_models()
    best_all = pull()
    st.dataframe(best_all)
    st.write(best)
    return best

# Function to perform a regression task on the provided DataFrame
def perform_regression(df, target_column, numeric_imputation, categorical_imputation, normalize_method, normalize):
    setup(df, target=target_column, numeric_imputation=numeric_imputation, categorical_imputation=categorical_imputation, normalize_method=normalize_method, normalize=normalize, session_id=123) 
    setup_all = pull()
    st.dataframe(setup_all)
    best = compare_models()
    best_all = pull()
    st.dataframe(best_all)
    st.write(best)
    return best  

# Function to find categorical and numerical columns in the DataFrame
@st.cache_data
def find_cat_cont_columns(df): 
    num_columns, cat_columns = [],[]
    for col in df.columns:        
        if len(df[col].unique()) <= 25 or df[col].dtypes == 'object': 
            cat_columns.append(col.strip())
        else:
            num_columns.append(col.strip())
    return num_columns, cat_columns

# Create Correlation Chart using Matplotlib
def create_correlation_chart(corr_df):
    fig = px.imshow(corr_df,
                    x=corr_df.columns,
                    y=corr_df.columns,
                    color_continuous_scale='Blues',text_auto=True)

    fig.update_xaxes(tickangle=-45, tickfont=dict(size=15))
    fig.update_yaxes(tickfont=dict(size=15))
    fig.update_layout(
        title='Correlation Chart',
        height=600,
        width=600
    )

    return fig
    
# Define a function to clean the data
def clean_data(df, columns_to_drop):
    if not columns_to_drop:
        st.warning("Please select at least one column to drop.")
    else:
        df  = df.drop_duplicates().reset_index(drop=True)
        df.drop(columns=columns_to_drop , inplace= True)
        st.success("Selected columns dropped / Duplicated rows dropped. Updated DataFrame:")
        st.write(df)
        return df
    
# Streamlit app
st.title("Data Analysis and Cleaning App")
upload = st.file_uploader(label="Upload File Here:", type=["csv", "xlsx"])

df = None

if upload:
    df = load_data(upload)
    # Create tabs for different sections of the app
    tab1, tab2, tab3, tab4 = st.tabs(["Dataset Overview", "Data Cleaning", "Charts","machine learning Task"])

    with tab1:  # Dataset Overview Tab
        st.subheader("1. Dataset")
        st.write(df)
        miss_data(df)
        duplicated_rows = df[df.duplicated()]
        if not duplicated_rows.empty:
            st.warning("Duplicated rows found:")
            st.write(duplicated_rows)
        else:
            st.info("No duplicated rows found.")

    with tab2:  # Data Cleaning Tab 
        # This should return the cleaned DataFrame
        st.subheader("3. Data Cleaning")
        columns_to_drop = st.multiselect("Columns to Drop", df.columns.tolist())
        if st.checkbox("Drop Columns and Duplicated Values"):
            df = clean_data(df, columns_to_drop)
            miss_data(df)
                              
        st.subheader("4. Dataset Overview")
        num_columns, cat_columns = find_cat_cont_columns(df) 
        st.markdown("<span style='font-weight:bold;'>{}</span> : {}".format("Rows", df.shape[0]), unsafe_allow_html=True)
        st.markdown("<span style='font-weight:bold;'>{}</span> : {}".format("Duplicates", df.shape[0] - df.drop_duplicates().shape[0]), unsafe_allow_html=True)
        st.markdown("<span style='font-weight:bold;'>{}</span> : {}".format("Features", df.shape[1]), unsafe_allow_html=True)
        st.markdown("<span style='font-weight:bold;'>{}</span> : {}".format("Categorical Columns", len(cat_columns)), unsafe_allow_html=True)
        st.write(cat_columns)
        st.markdown("<span style='font-weight:bold;'>{}</span> : {}".format("Numerical Columns", len(num_columns)), unsafe_allow_html=True)
        st.write(num_columns)
    
    with tab3:  # Dataset Overview Tab              
        st.subheader("3. Correlation Chart")
        corr_df = df[num_columns].corr()
        corr_fig=create_correlation_chart(corr_df)
        st.plotly_chart(corr_fig, use_container_width=True)
             
        st.subheader("Explore Relationship Between Features of Dataset")  
        x_axis = st.selectbox(label="X-Axis", options=num_columns)
        y_axis = st.selectbox(label="Y-Axis", options=num_columns)
        color_encode = st.selectbox(label="Color-Encode", options=[None,] + cat_columns)
        fig = px.scatter(df, x=x_axis, y=y_axis, color=color_encode)        
        st.plotly_chart(fig, use_container_width=True)
  
    with tab4:  # Dataset Overview Tab     
        task = st.radio("Select Task", ["Classification", "Regression"])
        if task == "Classification":
                from pycaret.classification import *
                st.subheader("Classification Task")
                target_columns_classification = [col for col in df.columns if len(df[col].unique()) <= 25 or df[col].dtypes == 'object']
                if len(target_columns_classification) > 0:
                    target_column = st.selectbox("Select the Target Column for Classification", target_columns_classification)
                    numeric_imputation = st.selectbox("Select the numeric_imputation for missing value:", ['mean', 'median', 'mode'])
                    st.write('------------------------------------------')
                    str_missing = st.text_input('Enter the text you want to fill the categorical missing value or leave it empty if not')
                    categorical_imputation = st.selectbox("Select the categorical_imputation for missing value:", ['drop', 'mode', str_missing])
                    st.write('-------------------------------------------')
                    st.write('Active normalize')
                    normalize = st.radio('Active or not', options=['True', 'False'])
                    normalize_method = st.selectbox("Select the normalize_method for missing value:", ['zscore', 'maxabs', 'robust', 'minmax'])
                    if st.button("Machine"):
                        best_model = perform_classification(df, target_column, numeric_imputation, categorical_imputation, normalize_method, normalize)
                        st.write(f"Best Classification Model: {best_model}")
                else:
                    st.warning("No categorical columns found for the classification task.")
        else:
            from pycaret.regression import *
            st.subheader("Regression Task")
            target_columns_regression = [col for col in df.columns if df[col].dtype in ['int64', 'float64']]
            if len(target_columns_regression) > 0:
                target_column = st.selectbox("Select the Target Column for Regression", target_columns_regression)
                numeric_imputation = st.selectbox("Select the numeric_imputation for missing value:", ['mean', 'median', 'mode'])
                st.write('------------------------------------------')
                str_missing = st.text_input('Enter the text you want to fill the categorical missing value or leave it empty if not')
                categorical_imputation = st.selectbox("Select the categorical_imputation for missing value:", ['drop', 'mode', str_missing])
                st.write('-------------------------------------------')
                st.write('Active normalize')
                normalize = st.radio('Active or not', options=['True', 'False'])
                normalize_method = st.selectbox("Select the normalize_method for missing value:", ['zscore', 'maxabs', 'robust', 'minmax'])
                if st.button("Machine"):
                    best_model = perform_regression(df, target_column, numeric_imputation, categorical_imputation, normalize_method, normalize)
                    st.write(f"Best Regression Model: {best_model}")
                else:
                    st.warning("No numeric columns found for the regression task.")