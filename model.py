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

import pickle


def model():
    
    # load the model from disk
    #filename1 = 'churn_model.sav'
    #loaded_model = pickle.load(open(filename1, 'rb'))
    #Github file size issue
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
        
        X = df.iloc[:,0:9]
        y = df.iloc[:, -1]
        
        st.write(X.head(5))
        st.write(X.columns)   
       
        #Categorical Variable Transfomation 
        #Gender
        X["Gender"] = X["Gender"].astype('category')

        #using the cat.codes accessor
        X["Gender"] =X["Gender"].cat.codes

        #Encoding the GEography Variable
        X = pd.get_dummies(X,drop_first=True, dtype='int64')
        
        #split the dataset
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0)

        #Scale the data
        sc = StandardScaler()
        X_train = sc.fit_transform(X_train)
        X_test = sc.transform(X_test)

        # Fitting Random Forest Classification to the Training set
        rf_model = RandomForestClassifier(n_estimators = 1000, criterion = 'entropy', random_state = 0)
        rf_model.fit(X_train, y_train)
        
        

    st.subheader("Variables contributing to the Churn/Customer Retention")
    st.write("via Feature Importance")
    st.image('feat_imp.png',use_column_width=True)
    
    if st.sidebar.checkbox("What happened"):
        st.subheader("This method will randomly shuffle each feature and compute the change in the model’s performance.")
        st.write("via Permutation Importance")
        st.image('permu_plot.png')
    if st.sidebar.checkbox("Why it happened"):
        st.subheader("This method will use the Shapley values from game theory to estimate the how does each feature contribute to the prediction.")
        st.write("via The SHAP interpretation")
        st.image('shap_plot.png')
    if st.sidebar.checkbox("What could happen?"):
        st.header("Let's Predict")
        
        p_o11, p_o22, p_o33 = st.beta_columns(3)
        with p_o11:
            CreditScore = st.slider("What if the CreditScore is",1,1000,650,key='0svmk33')
            st.write("Default: 650(Average) Taken",CreditScore)
            Tenure = st.slider("What if the Tenure is",1,10,5,key='0svmk33')
            st.write("Default: 5(Average) Taken",Tenure)
            Balance = st.slider("What if the Balance is",1000,250898,76485,key='0svmk33')
            st.write("Default: 76485(Average) Taken",Balance)

                   
        with p_o22:
            Gender = st.slider("What if the Gender is",1,0,1,key='0svmk33')
            st.write("Default:0=Female,1=Male, Taken", Gender)

            NumOfProducts = st.slider("What if the Number of Products are",1,5,1,key='0svmk33')
            st.write("Default: 1(Average) Taken",NumOfProducts)
            
            EstimatedSalary = st.slider("What if the Estimated Salary",12,199992,100090)
            st.write("Defaul:100090(Average) Taken",EstimatedSalary )
            
        with p_o33:
            Age = st.slider("What if the Age is",1,92,38,key='0svmk3')
            st.write("Default: 38(Average) Taken",Age)
            
            Geo_Germany= st.slider("What if the Location is",1,0,1,key='0ger')
            st.write("Default:1=Germany 0=FranceTaken",Geo_Germany)
            Geo_spain= st.slider("What if the Location is",1,0,1,key='0spain')
            st.write("Default:1=Spain 0=France Taken",Geo_spain)               
                                      
                                                           
        p_o111,p_o222,p_o333 = st.beta_columns(3)
        with p_o111:
            HasCrCard = st.slider("What if has Credit Card",1,0,1)
            st.write("Defaul: 1=Has 0=Not Taken", HasCrCard)
            
        with p_o222:
            IsActiveMember = st.slider("What if is/not Active Member",1,0,1)
            st.write("Defaul: 1=Activate 0=Not Taken", IsActiveMember)

        with p_o333:
            pass
            #EstimatedSalary = st.slider("What if the Estimated Salary",12,199992,100090)
            #st.write("Defaul:100090(Average) Taken",EstimatedSalary )
            
        results = rf_model.predict([[CreditScore,Gender,Age,Tenure,Balance,NumOfProducts,HasCrCard,IsActiveMember,
                                         EstimatedSalary,Geo_Germany,Geo_spain]])
        
        #st.write(results)
        Exited=1
        st.header("Predictied ~")
        if results== Exited:
            st.header("86% Change the customer will leave")
        else:
            st.header("86% Change the customer will not leave")

        st.text("Tweak Number of Products to 2 to see the app in action ")


    
    
    