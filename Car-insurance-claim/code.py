# --------------
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

#loading the dataset
df = pd.read_csv(path)
print("the first 5 columns are:",df.columns[:5])


print(df.info())
col=['INCOME','HOME_VAL','BLUEBOOK','OLDCLAIM','CLM_AMT']
#Removing the '$' and ',' from col
for column_name in col:
    df[column_name] = df[column_name].replace({'\$': '', ',': ''}, regex=True)
    
X = df.iloc[:,:-1]
y = df.iloc[:,-1]
print(X.head(5))
count = y.value_counts()
print(count)

#splitting the dataframe

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.3,random_state=6)




# --------------

for i in col:
    X_train[i] = X_train[i].astype(float)
    X_test[i] = X_test[i].astype(float)
print(X_train.isnull().sum())
print(X_test.isnull().sum())


# --------------
# We can see that the features ['YOJ','OCCUPATION'] varies person to person. We can not deal with that type of missing value so we are going to remove the row from this column.
X_train.dropna(subset=['YOJ', 'OCCUPATION'], inplace=True)
X_test.dropna(subset=['YOJ', 'OCCUPATION'], inplace=True)
y_train = y_train[X_train.index]
y_test = y_test[X_test.index]
column_mean = ['AGE', 'CAR_AGE', 'INCOME','HOME_VAL']

for column in column_mean:
    mean_train = X_train[column].mean()
    mean_test  = X_test[column].mean()
    X_train[column].fillna(value=mean_train, inplace=True)
    X_test[column].fillna(value=mean_test, inplace=True)


# --------------
from sklearn.preprocessing import LabelEncoder
columns = ["PARENT1","MSTATUS","GENDER","EDUCATION","OCCUPATION","CAR_USE","CAR_TYPE","RED_CAR","REVOKED"]

# Code starts here
for i in columns :
    le =LabelEncoder()
    X_train[i] = le.fit_transform(X_train[i].astype(str))
    X_test[i] = le.fit_transform(X_test[i].astype(str))

# Code ends here



# --------------
from sklearn.metrics import precision_score 
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression



# logistic regression 
model = LogisticRegression(random_state = 6)
model.fit(X_train,y_train)

y_pred = model.predict(X_test)

score = accuracy_score(y_test,y_pred)
print('the accuracy using model is :',score)

precision = precision_score(y_test,y_pred)
print('the precision of model is :',precision)



# --------------
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE

# smote
smote = SMOTE(random_state = 9)
X_train,y_train = smote.fit_sample(X_train,y_train)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)



# --------------
# Code Starts here
model = LogisticRegression(random_state = 6)
model.fit(X_train,y_train)

y_pred = model.predict(X_test)

score = accuracy_score(y_test,y_pred)
print('the accuracy using model is :',score)

# Code ends here


