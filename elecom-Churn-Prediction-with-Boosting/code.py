# --------------
import pandas as pd
from sklearn.model_selection import train_test_split
 
#loading the dataset:
df = pd.read_csv(path)

#extracting features:
X = df.drop(['customerID','Churn'],axis=1)
y = df.iloc[:,-1]

#splitting the dataset into train and test
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size = 0.3 ,random_state = 0)



# --------------
import numpy as np
from sklearn.preprocessing import LabelEncoder

#replacling some Nan values

X_train['TotalCharges'] = X_train['TotalCharges'].replace(' ', np.NaN)
X_train['TotalCharges'] = X_train['TotalCharges'].astype('float64')

X_test['TotalCharges'] = X_test['TotalCharges'].replace(' ', np.NaN)
X_test['TotalCharges'] = X_test['TotalCharges'].astype('float64')

X_train['TotalCharges'].fillna(value=X_train['TotalCharges'].mean(), inplace=True)
X_test['TotalCharges'].fillna(value=X_test['TotalCharges'].mean(), inplace=True)


categorical_columns = X_test.select_dtypes(include=['object']).columns.values.tolist()
#label encoding

encoder = LabelEncoder()
for i in categorical_columns:
    X_train[i] = encoder.fit_transform(X_train[i])
    X_test[i] = encoder.fit_transform(X_test[i])

#replacing y_train and y_test values:
y_train.replace({'No':0, 'Yes':1},inplace = True)
y_test.replace({'No':0, 'Yes':1},inplace = True)
    



# --------------
from sklearn.ensemble import AdaBoostClassifier
from sklearn.metrics import accuracy_score,classification_report,confusion_matrix

# printing values after encoding
print(X_train.head(5))
print("####")
print(X_test.head(5))
print("####")
print(y_train.head(5))
print("####")
print(y_test.head(5))
print("####")

#fitting model using AdaBoost
ada_model = AdaBoostClassifier(random_state = 0)
ada_model.fit(X_train,y_train)
y_pred = ada_model.predict(X_test)

#checking accuracy
ada_score = accuracy_score(y_test,y_pred)
print("accuracy score using adaboost :",ada_score)

#confusion matrix
ada_cm = confusion_matrix(y_test,y_pred)
print(ada_cm)

#classification report
ada_cr = classification_report(y_test,y_pred)
print(ada_cr)



# --------------
from xgboost import XGBClassifier
from sklearn.model_selection import GridSearchCV

#Parameter list
parameters={'learning_rate':[0.1,0.15,0.2,0.25,0.3],
            'max_depth':range(1,3)}

# using XGBoost
xgb_model = XGBClassifier(random_state = 0)
xgb_model.fit(X_train,y_train)
y_pred = xgb_model.predict(X_test)

#checking accuracy score:
xgb_score = accuracy_score(y_test,y_pred)
print("accuracy score using xgboost :",xgb_score)

#confusion matrix
xgb_cm = confusion_matrix(y_test,y_pred)
print('confusion matrix of xgboost',xgb_cm)

#classification report
xgb_cr = classification_report(y_test,y_pred)
print("classification report of xgboost :",xgb_cr)


#use of gridsearchcv

clf_model = GridSearchCV(estimator = xgb_model,param_grid = parameters)
clf_model.fit(X_train,y_train)
y_pred = clf_model.predict(X_test)

#accuracy score
clf_score = accuracy_score(y_test,y_pred)
print("final accuracy score :",clf_score)

#confusion matrix
clf_cm = confusion_matrix(y_test,y_pred)
print('Final confusion matrix:',clf_cm)

#Classification report
clf_cr = classification_report(y_test,y_pred)
print("final classification report",clf_cr)



