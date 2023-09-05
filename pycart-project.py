import streamlit as st 
import pandas as pd 
import os
import ydata_profiling
from streamlit_pandas_profiling import st_profile_report
from pycaret.regression import *

# Check if 'Sourcedata.csv' file exists and load it if it does
if os.path.exists('Sourcedata.csv'):
    df = pd.read_csv('Sourcedata.csv', index_col=None)

# Configure Streamlit page layout to 'wide'
st.set_page_config(layout='wide')

# Create a sidebar with navigation options
with st.sidebar:
    # Display an image and title in the sidebar
    st.image('machine-learning-1.png')
    st.title('Auto Machine Learning')
    # Create a radio button for navigation
    choice = st.radio('Navigation', ['Upload', 'AutoEDA', 'Machine learning'])
    
# Handle the case when 'Upload' option is selected
if choice == 'Upload':
    st.title('Upload your data for Modelling')
    # Allow users to upload a dataset
    file = st.file_uploader('Upload your dataset here')   
    if file:
        # Read the uploaded file into a DataFrame and save it as 'Sourcedata.csv'
        df = pd.read_csv(file, index_col=None)
        df.to_csv('Sourcedata.csv', index=None)
        st.dataframe(df)

# Handle the case when 'AutoEDA' option is selected
if choice == 'AutoEDA':
    st.title('AutoMated Exploratory Data Analysis')
    # Generate a profile report of the DataFrame
    profile = df.profile_report()
    st_profile_report(profile)

# Handle the case when 'Machine learning' option is selected
if choice == 'Machine learning':
    st.title('Machine Learning ')   
    # Allow users to select the target column for machine learning
    target = st.selectbox('Choose target', df.columns)
    
    # Train the data for machine learning when the 'train data' button is clicked
    if st.button('Train data'):
        # Perform setup for machine learning using pycaret
        setup(df, target=target)
        setup_pull = pull()
        st.info('This is the ML experimental setting')
        st.dataframe(setup_pull)
        
        # Compare machine learning models
        best_model = compare_models()
        compare_df = pull()
        st.info('These are the ML model comparison results')
        st.dataframe(compare_df)
        
        # Display the best model
        best_model
        
        # Save the best model
        save_model(best_model, 'best_model')
