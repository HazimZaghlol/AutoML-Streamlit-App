import streamlit as st
import pandas as pd
import missingno
import matplotlib.pyplot as plt
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
def perform_classification(df, target_column,numeric_imputation,categorical_imputation,normalize_method, normalize):
    setup(df, target=target_column, numeric_imputation=numeric_imputation,categorical_imputation=categorical_imputation, normalize_method=normalize_method, normalize=normalize, session_id=123) 
    setup_all = pull()
    st.dataframe(setup_all)
    best = compare_models()
    best_all = pull()
    st.dataframe(best_all)
    st.write(best)
    return best

# Function to perform a regression task on the provided DataFrame
def perform_regression(df, target_column,numeric_imputation,categorical_imputation,normalize_method, normalize):
    setup(df, target=target_column, numeric_imputation=numeric_imputation,categorical_imputation=categorical_imputation, normalize_method=normalize_method, normalize=normalize, session_id=123) 
    setup_all = pull()
    st.dataframe(setup_all)
    best = compare_models()
    best_all = pull()
    st.dataframe(best_all)
    st.write(best)
    return best  

@st.cache_data
def find_columns_with_missing_values(df): 
    num_columns_with_missing = []
    cat_columns_with_missing = []
    
    for col in df.columns:
        if df[col].dtype in [int, float]:
            if df[col].isnull().any():
                num_columns_with_missing.append(col)
        else:
            if df[col].isnull().any():
                cat_columns_with_missing.append(col)
    
    return num_columns_with_missing, cat_columns_with_missing

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

def fill_missing_values_num(df, column, imputation_method):
    if imputation_method == "None":
        st.info("No imputation selected.")
    else:
        if imputation_method == "Mean":
            imputed_value = df[column].mean()
        elif imputation_method == "Median":
            imputed_value = df[column].median()
        elif imputation_method == "Mode":
            imputed_value = df[column].mode()[0]  # Take the first mode if multiple modes
        elif imputation_method == "Custom Value":
            custom_value = st.number_input(f"Enter custom value for {column}", value=0.0)
            imputed_value = custom_value

        df = df[column].fillna(imputed_value, inplace=True)
        st.success(f"Missing values in '{column}' filled with {imputation_method} value: {imputed_value}")  
        return df
        
    
def fill_missing_values_cat(df, column, imputation_method_categorical):
    if imputation_method_categorical == "None":
        st.info("No imputation selected.")
    else:
        if imputation_method_categorical == "Mode":
                imputed_value = df[column].mode()[0]  # Take the first mode if multiple modes
        elif imputation_method_categorical == "Custom Value":
            imputed_value = custom_value

        df = df[column].fillna(imputed_value, inplace=True)
        st.success(f"Missing values in '{column}' filled with {imputation_method_categorical} value: {imputed_value}") 
        return df      

        
    
# Streamlit app
st.title("Machine learning App")
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
        st.subheader("1. Data Cleaning")
        
        # Display the updated DataFrame
        columns_to_drop = st.multiselect("Columns to Drop", df.columns.tolist())
        
        if st.checkbox("Drop Columns and Duplicated Values"):
            # Display the updated DataFrame after dropping columns
            st.subheader("Updated Dataset")
            df = clean_data(df, columns_to_drop)
        
        
        
        # Get columns with missing values
        num_columns_with_missing, cat_columns_with_missing = find_columns_with_missing_values(df)
        
        # Option to drop rows with missing values based on specific columns
        columns_to_drop_missing_rows = st.multiselect("Columns for Missing Rows Dropping", df.columns.tolist())
        if st.checkbox("Drop Rows with Missing Values Based on Selected Columns"):
            if not columns_to_drop_missing_rows:
                st.warning("Please select columns to drop rows with missing values.")
            else:
                df = df.dropna(subset=columns_to_drop_missing_rows)
                st.success(f"Rows with missing values in selected columns ({', '.join(columns_to_drop_missing_rows)}) dropped.")
        
        # Update the list of columns with missing values after dropping rows
        num_columns_with_missing, cat_columns_with_missing = find_columns_with_missing_values(df)
        
        if st.checkbox("Fill Missing Values"):
            if not num_columns_with_missing and not cat_columns_with_missing:
                st.warning("No columns with missing values found.")
            else:
                # Select column to impute
                column_to_impute = st.selectbox("Select the target column to impute", options=num_columns_with_missing + cat_columns_with_missing)
                
                # Check if the selected column is numeric or categorical
                if column_to_impute in num_columns_with_missing:
                    imputation_method_numeric = st.selectbox("Numerical Imputation Method", ["None", "Mean", "Median", "Mode", "Custom Value"])
                    if st.checkbox(f"Impute Missing Values in '{column_to_impute}'"):
                       fill_missing_values_num(df, column_to_impute, imputation_method_numeric)
            
                           
                            
                elif column_to_impute in cat_columns_with_missing:
                    imputation_method_categorical = st.selectbox("Categorical Imputation Method", ["None", "Mode", "Custom Value"])
                    if imputation_method_categorical == "Custom Value":
                        custom_value = st.text_input(f"Enter custom value for {column_to_impute}", value="")
                    if st.checkbox(f"Impute Missing Values in '{column_to_impute}'"):
                        if imputation_method_categorical == "None":
                            st.info("No imputation selected.")
                        else:
                          fill_missing_values_cat(df, column_to_impute, imputation_method_categorical)
                            
        
        miss_data(df)
                            
        st.subheader("2. Dataset Overview")
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
  
    with tab4:  # machine learning Tab     
        num_columns, cat_columns = find_cat_cont_columns(df)
        st.markdown("<span style='font-weight:bold;'>{}</span> : {}".format("Categorical Columns", len(cat_columns)), unsafe_allow_html=True)
        st.write(cat_columns)
        st.markdown("<span style='font-weight:bold;'>{}</span> : {}".format("Numerical Columns", len(num_columns)), unsafe_allow_html=True)
        st.write(num_columns)
        target_column = st.selectbox("Select the Target Column for Regression", df.columns)
        if target_column in cat_columns :
            from pycaret.classification import *
            st.subheader("Classification Task")
            st.write('Active normalize')
            normalize = st.radio('Active or not', options=['True', 'False'])
            normalize_method = st.selectbox("Select the normalize_method for missing value:", ['zscore', 'maxabs', 'robust', 'minmax'])
            # Text input fields for manual imputation
            numeric_imputation_value = st.text_input("Enter numeric imputation value (if manual) and choose it:")
            categorical_imputation_value = st.text_input("Enter categorical imputation value (if manual):")
            # Selectbox for numeric_imputation and categorical_imputation
            numeric_imputation = st.selectbox("Select numeric imputation method:", ['drop', 'median', 'mode', 'mean','knn',numeric_imputation_value])
            categorical_imputation = st.selectbox("Select categorical imputation method:", ['drop', 'mode',categorical_imputation_value])
            if st.button("Train Data"):
                best_model = perform_classification(df, target_column,numeric_imputation=numeric_imputation,categorical_imputation=categorical_imputation,normalize_method=normalize_method, normalize=normalize)
                st.write(f"Best Classification Model: {best_model}")
        else:
            from pycaret.regression import *
            st.subheader("Regression Task")
            st.write('Active normalize')
            normalize = st.radio('Active or not', options=['True', 'False'])
            normalize_method = st.selectbox("Select the normalize_method for missing value:", ['zscore', 'maxabs', 'robust', 'minmax'])
            # Text input fields for manual imputation
            numeric_imputation_value = st.text_input("Enter numeric imputation value (if manual) and choose it:")
            categorical_imputation_value = st.text_input("Enter categorical imputation value (if manual):")
            # Selectbox for numeric_imputation and categorical_imputation
            numeric_imputation = st.selectbox("Select numeric imputation method:", ['drop', 'mean', 'mean', 'mean','knn',numeric_imputation_value])
            categorical_imputation = st.selectbox("Select categorical imputation method:", ['drop', 'mode',categorical_imputation_value])
            if st.button("Train Data"):
                best_model = perform_classification(df, target_column,numeric_imputation=numeric_imputation,categorical_imputation=categorical_imputation,normalize_method=normalize_method, normalize=normalize)
                st.write(f"Best Classification Model: {best_model}")
            
     