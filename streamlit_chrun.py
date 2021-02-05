#semi auto ML app
import streamlit as st

#text to speech pckgs
import pyttsx3

#EDA pkgs
import pandas as pd
import numpy as np

#data visualization pcks
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
import seaborn as sns
sns.set_theme(style="ticks", color_codes=True)
#import plotly.graph_objects as go
import plotly.express as px


#ML pckgs
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import StandardScaler
from sklearn import metrics

from sklearn import model_selection
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn.model_selection import GridSearchCV
#disable warning message
st.set_option('deprecation.showfileUploaderEncoding', False)
st.set_option('deprecation.showPyplotGlobalUse', False)
# Use the full page instead of a narrow central column
st.set_page_config(layout="wide",page_title="Customer Retention App",
                   initial_sidebar_state="expanded",page_icon="🧊")

#loading the plot module
from plot_learning import run_churn_plots
from model import model

import pickle

import base64
import time
timestr = time.strftime("%Y%m%d-%H%M%S")
# Fxn to Download Result
def download_link(object_to_download, download_filename, download_link_text):
    d=pickle.dump(lr_model, open(filename, 'wb'))
    # some strings <-> bytes conversions necessary here
    b64 = base64.b64encode(object_to_download.encode()).decode()
    
    return f'<a href="data:file/txt;base64,{b64}" download="{d}">"click click"</a>'

def make_downloadable(data):
    csvfile = data.to_csv(index=False)
    b64 = base64.b64encode(csvfile.encode()).decode()
    new_filename = "Analysis_report_{}_.csv".format(timestr)
    st.markdown("🤘🏻  Download CSV file ⬇️  ")
    href = f'<a href="data:file/csv;base64,{b64}" download="{new_filename}">Click here!</a>'
    st.markdown(href, unsafe_allow_html=True)
    
def main():
    """Auto-Machine Learning App with Streamlit """
    
    st.title("Advanced Ai model for Customer Retention Churn Analysis")
    activities = ["EDA","Plots","Model Building","About"]
    
    choice = st.sidebar.selectbox("Select Activity",activities)

    
    if choice == 'EDA':
        st.subheader("Exploratory Data Analysis")

        data1 = st.file_uploader("Pre-Dataset Loaded",type = ["csv","txt"])
        data = 'Churn_Modelling.csv'
        if data is not None:
            df = pd.read_csv(data,sep=',')
            #st.dataframe(df.head(10))
            selected_col = ['RowNumber','CustomerId','Surname']
            df = df.drop(selected_col,axis=1)
            df20=df.head(20)           
            temp_numeric = df.select_dtypes(include=['int64','float64'])
            t_n_c =temp_numeric.columns.to_list()
            st.table(df20.style.bar(subset=t_n_c,align='mid', color=['#ffeceb', '#f0fff4']))
            st.write("showing only 20 rows.....from", df.shape[0],"rows" )  
            
            st.subheader("Visualizing statistical relationships")
            st.write("The process of understanding how variables in a dataset relate to each other and how those relationships depend on other variables.")

            eda_o1, eda_o2 = st.beta_columns((1,3))
            
            temp_char= df.select_dtypes(exclude=['int64','float64'])
            t_c_c =temp_char.columns.to_list()
            kind1 = ['line','scatter']
            col1 = ['Tenure','Gender','Geography','IsActiveMember','Exited']            
            
            with eda_o1:
                x1= st.selectbox("Select Variable 1 (numeric)",t_n_c)
                y1 = st.selectbox("Select Variable 2 (numeric)",t_n_c)
                hue1 = st.selectbox("Select Variable 3 (categorical)",t_c_c)
                kind2= st.selectbox("Select Kind of plot",kind1)
                style1 = st.selectbox("Select variable 4 (categorical)",t_c_c)
                col2 = st.selectbox("Select variable 5 (categorical)",col1)
            
            with eda_o2:
                #distribution of data
                sns.relplot(data=df, x=x1, y=y1, hue=hue1, kind=kind2,style=style1,col=col2,col_wrap=3,legend="auto")
                st.pyplot()
               
            if st.sidebar.checkbox("Summary Statistics"):
                st.subheader("Summary of the data")
                df_describe = df.describe()
                st.table(df_describe.style.bar(subset=t_n_c,align='mid', color=['#ffeceb', '#f0fff4']))
                
                        
            with st.beta_expander("Download The report"):
                href = f'<a href="data:file/txt" download="d">Click here!</a>'
                st.markdown(href, unsafe_allow_html=True)
                       
    elif choice == 'Plots':
        run_churn_plots()
        
    elif choice == 'Model Building':
        model()

    elif choice == 'About':
        st.subheader("About")
        st.text("Thank you for your time")
        
if __name__ == '__main__':
    main()
    
    