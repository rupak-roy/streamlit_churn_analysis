
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import StandardScaler

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

from sklearn.model_selection import GridSearchCV

#importing the dataset
dataset = pd.read_csv("Churn_Modelling.csv",sep=",")

#drop the missing values
dataset = dataset.dropna()
dataset.isnull().sum()

dataset.shape
X = dataset.iloc[:,3:13]
X.columns
y = dataset.iloc[:, -1]

X.dtypes
y.dtypes

X.Gender.value_counts()
X.Gender.unique()

#Categorical Variable Transfomation 
#Gender
X["Gender"] = X["Gender"].astype('category')
X.dtypes
#using the cat.codes accessor
X["Gender"] =X["Gender"].cat.codes
X.Gender
X.Gender.value_counts()
X.Gender.unique()

#Encoding the GEography Variable
X = pd.get_dummies(X,drop_first=True, dtype='int64')
X.dtypes

'''# Taking care of missing data
from sklearn.preprocessing import Imputer
imputer = Imputer(missing_values = 'NaN', strategy = 'mean', axis = 0)
imputer = imputer.fit(X[:, 1:3])
X[:, 1:3] = imputer.transform(X[:, 1:3]) '''

#split the dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0)

#Scale the data
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# Fitting Random Forest Classification to the Training set
rf_model = RandomForestClassifier(n_estimators = 500, criterion = 'entropy', random_state = 0)
rf_model.fit(X_train, y_train)

# Predicting the Test set results
y_pred_rf = rf_model.predict(X_test)

# Making the Confusion Matrix 
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred_rf)
 
#evaluation Metrics 
from sklearn import metrics
print('Accuracy Score:', metrics.accuracy_score(y_test, y_pred_rf))  
print('Balanced Accuracy Score:', metrics.balanced_accuracy_score(y_test, y_pred_rf)) 
print('Average Precision:',metrics.average_precision_score(y_test, y_pred_rf))  

#CROSS VALIDATION----------------------------------------------------------------------
# Applying k-Fold Cross Validation
from sklearn.model_selection import cross_val_score
accuracies = cross_val_score(estimator = rf_model, X = X_train, y = y_train, cv = 7)
accuracies.mean()
accuracies.std()

# Applying Grid Search to find the best model and the best parameters
from sklearn.model_selection import GridSearchCV
parameters = [{ 'n_estimators': [500,1000,1500,2000]}]

grid_search = GridSearchCV(estimator = rf_model,
                           param_grid = parameters,
                           scoring = 'accuracy',
                           cv = 7)
grid_search = grid_search.fit(X_train, y_train)

best_accuracy = grid_search.best_score_
best_parameters = grid_search.best_params_

#---------------------------------------------------------------------------------
#Re-Running the model with the best fit parameters
#Fitting Random Forest Classification to the Training set
rf_model = RandomForestClassifier(n_estimators = 1000, criterion = 'entropy', random_state = 0)
rf_model.fit(X_train, y_train)

# Predicting the Test set results
y_pred_rf = rf_model.predict(X_test)

# Making the Confusion Matrix 
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred_rf)
 
#evaluation Metrics 
from sklearn import metrics
print('Accuracy Score:', metrics.accuracy_score(y_test, y_pred_rf))  
print('Balanced Accuracy Score:', metrics.balanced_accuracy_score(y_test, y_pred_rf)) 
print('Average Precision:',metrics.average_precision_score(y_test, y_pred_rf))  


# Applying k-Fold Cross Validation
from sklearn.model_selection import cross_val_score
accuracies = cross_val_score(estimator = rf_model, X = X_train, y = y_train, cv = 7)
accuracies.mean()
accuracies.std()
#-------------------------------------------------------------------------------------

#save the model 
import pickle
filename = 'churn_model.sav'
pickle.dump(rf_model, open(filename, 'wb'))

# load the model from disk
filename1 = 'churn_model.sav'
loaded_model = pickle.load(open(filename1, 'rb'))

# Use the loaded model to make predictions 
loaded_model.predict(X_test)
#--------------------------------------------------------------------------------
#Feature importance 
sorted_idx = loaded_model.feature_importances_.argsort()
plt.barh(X.columns[sorted_idx], loaded_model.feature_importances_[sorted_idx])
plt.xlabel("Importance of the variable contributing to the Customer Retention")

#Feature importance by Permutation-------------------------------------------------------- 
from sklearn.inspection import permutation_importance
perm_importance = permutation_importance(loaded_model, X_test, y_test)

sorted_idx = perm_importance.importances_mean.argsort()
plt.barh(X.columns[sorted_idx], perm_importance.importances_mean[sorted_idx])
plt.xlabel("Permutation Importance")

#Feature importance using shap--------------------------------------------------------
import shap
explainer = shap.TreeExplainer(loaded_model)
shap_values = explainer.shap_values(X_test)

#need to add the columns else default feature labels
shap.summary_plot(shap_values, X.columns, plot_type="bar")
 
# plot the SHAP values for the 10th observation 
#shap.force_plot(explainer.expected_value, shap_values[10,:], X_test.iloc[10,:],matplotlib=True)
CreditScore = 547
Gender = 1
Age = 55
Tenure = 4
Balance = 6678
NumOfProducts = 6
HasCard=1
IsActiveMember=1
EstimatedSalary= 5000
Geography_germany = 1
Geography_Spain = 0

loaded_model.predict([[344,1,67,5,6793,3,1,1,7777,1,0]])

results = loaded_model.predict([[CreditScore,Gender,Age,Tenure,Balance,NumOfProducts,HasCard,IsActiveMember,
                       EstimatedSalary,Geography_germany,Geography_Spain]])

Exited=1

if results== Exited:
    print("86% Change the custoemr willnot leave")
else:
    print("86% Change the customer will leave")

