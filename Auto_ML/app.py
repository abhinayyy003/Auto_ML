from operator import index
import streamlit as st
import plotly.express as px
from pycaret.regression import setup as regression_setup
from pycaret.regression import compare_models as regression_compare_models
from pycaret.regression import pull as regression_pull
from pycaret.regression import save_model as regression_save_model
from pycaret.regression import load_model as regression_load_model

from pycaret.classification import setup as classification_setup
from pycaret.classification import compare_models as classification_compare_models
from pycaret.classification import pull as classification_pull
from pycaret.classification import save_model as classification_save_model
from pycaret.classification import load_model as classification_load_model

import pandas_profiling
import pandas as pd
from streamlit_pandas_profiling import st_profile_report
import os 

if os.path.exists('./dataset.csv'): 
    df = pd.read_csv('dataset.csv', index_col=None)

with st.sidebar: 
    st.image("https://thumbs.dreamstime.com/b/machine-learning-concept-robot-education-hud-interface-d-rendering-231800180.jpg")
    st.title("Automatic ML")
    choice = st.radio("Navigation", ["Upload","Profiling","Regression Modelling", "Classification Modelling", "Download"])
    st.info("Abhinay")

if choice == "Upload":
    st.title("Upload Your Dataset")
    file = st.file_uploader("Upload Your Dataset")
    if file: 
        df = pd.read_csv(file, index_col=None)
        df.to_csv('dataset.csv', index=None)
        st.dataframe(df)

if choice == "Profiling": 
    st.title("Data Analysis")
    profile_df = df.profile_report()
    st_profile_report(profile_df)

if choice == "Regression Modelling": 
    chosen_target = st.selectbox('Choose the Target Column', df.columns)
    if st.button('Run Modelling'): 
        regression_setup(df, target=chosen_target)
        setup_df = regression_pull()
        st.dataframe(setup_df)
        best_model = regression_compare_models()
        compare_df = regression_pull()
        st.dataframe(compare_df)
        regression_save_model(best_model, 'best_model')


if choice == "Classification Modelling": 
    chosen_target = st.selectbox('Choose the Target Column', df.columns)
    cate_features = st.multiselect('Choose Categorical Columns', df.columns)
    if st.button('Run Modelling'): 
        classification_setup(df, target=chosen_target, categorical_features=cate_features)
        setup_df = classification_pull()
        st.dataframe(setup_df)
        best_model = classification_compare_models()
        compare_df = classification_pull()
        st.dataframe(compare_df)
        classification_save_model(best_model, 'best_model')

if choice == "Download": 
    with open('best_model.pkl', 'rb') as f: 
        st.download_button('Download Model', f, file_name="best_model.pkl")
