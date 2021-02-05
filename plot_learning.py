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
#from xgboost import XGBClassifier
from sklearn.model_selection import GridSearchCV
#disable warning message
st.set_option('deprecation.showfileUploaderEncoding', False)
st.set_option('deprecation.showPyplotGlobalUse', False)



def run_churn_plots():
    
    st.subheader(" Advance In-Depth Vizualizers Analysis")

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
        
        #temp_char= df.select_dtypes(exclude=['int64','float64'])
        #t_c_c =temp_char.columns.to_list()
        #kind1 = ['line','scatter']
        #col1 = ['Tenure','Gender','Geography','IsActiveMember','Exited']    
    
        if st.sidebar.checkbox("Character Variable Analysis"):
            st.subheader("Specialized approach to visualization via Category.")
            st.write("The process of understanding how variables in a dataset relate to each other and how those relationships depend on other variables.")
            
            plot_o1, plot_o2 = st.beta_columns((1,2))
            
            with plot_o1:    
                Tenure1 = st.slider("Select Tenure great than & equal to",1,10,0,key='0svmk33')                
                t = df.Tenure[df.Tenure>=Tenure1]
                sns.catplot(x=t, y="Age", kind="swarm", data=df)
                st.pyplot()

               # sns.catplot(x="Age", kind="count", palette="ch:.25", data=df)
                
                sns.catplot(y="Exited", hue="Tenure", kind="count",palette="pastel", edgecolor=".6",data=df)
                st.pyplot()
                
            with plot_o2:
                sns.catplot(x="Tenure", y="Age", hue="Gender", kind="box",orient="v", data=df)
                st.pyplot()
                st.header("Category Variable Analysis")
                st.subheader("\.........via Seaborn")
            sns.catplot(x="NumOfProducts", y="CreditScore", hue="Gender",col="Geography", aspect=.7,kind="swarm", data=df)
            st.pyplot()
            
            sns.catplot(x="Age", y="CreditScore", hue="Gender",col="Geography", aspect=.7,kind="swarm", data=df)
            st.pyplot()
        
        if st.sidebar.checkbox("Numeric Variable Analysis"):
            st.subheader("Specialized approach to visualization via Numbers.")
            st.write("The process of understanding how variables in a dataset relate to each other and how those relationships depend on other variables.")
                    
            plot_o3, plot_o4 = st.beta_columns((.7,1.5))
            with plot_o3:
                sns.displot(df, x="Tenure", hue="Gender",multiple="stack")
                st.pyplot()
                
                sns.displot(df, x="CreditScore", hue="Geography", kind="kde")
                st.pyplot()
                
            with plot_o4:
                sns.displot(df, x="CreditScore", hue="Geography", stat="density",element="step",multiple="stack")
                st.pyplot()
              
            plot_o5, plot_o6 = st.beta_columns(2)
            with plot_o5:
                sns.displot(df, x="EstimatedSalary", y="Age", binwidth=(2, .5))
                st.pyplot()
                
            with plot_o6:
                #sns.displot(df, x="EstimatedSalary", y="CreditScore", binwidth=(2, .5), cbar=True)
                sns.jointplot(data=df,x="Tenure", y="CreditScore", hue="Gender",kind="kde")
                st.pyplot()
    
            sns.pairplot(df)
            st.pyplot()
        
        if st.sidebar.checkbox("Multi-Variable Analysis"):
            plot_o7, plot_o8,plot_o9 = st.beta_columns(3)
            with plot_o7:
                ordered_days = df.Tenure.value_counts().index
                g = sns.FacetGrid(df, row="Tenure", row_order=ordered_days,
                height=1.7, aspect=4,)
                g.map(sns.kdeplot, "Age")
                st.pyplot()
            with plot_o8:
                ordered_days = df.Tenure.value_counts().index
                g = sns.FacetGrid(df, row="Tenure", row_order=ordered_days,
                height=1.7, aspect=4,)
                g.map(sns.kdeplot, "EstimatedSalary")
                st.pyplot()
            with plot_o9:
                ordered_days = df.Tenure.value_counts().index
                g = sns.FacetGrid(df, row="Tenure", row_order=ordered_days,
                height=1.7, aspect=4,)
                g.map(sns.kdeplot, "CreditScore")
                st.pyplot()

            #attend = sns.load_dataset("attention").query("subject <= 12")
            g = sns.FacetGrid(df, col="Age", col_wrap=4, height=2, ylim=(0, 10))
            g.map(sns.pointplot, "HasCrCard", "Tenure", color=".3", ci=None)
            #need to fix order else incorrect plot g.map(sns.pointplot, "solutions", "score", order=[1, 2, 3], color=".3", ci=None)
            st.pyplot()
            
            g = sns.PairGrid(df)
            g.map_upper(sns.scatterplot)
            g.map_lower(sns.kdeplot)
            g.map_diag(sns.kdeplot, lw=3, legend=False)
            st.pyplot()                
            
            
            g = sns.pairplot(df, hue="Geography", palette="Set2", diag_kind="kde", height=2.5)
            st.pyplot()
    