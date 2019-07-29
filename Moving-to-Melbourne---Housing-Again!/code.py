# --------------
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

# path- variable storing file path

#Code starts here
df=pd.read_csv(path)

#displaying 1st five columns
print(df.head(5))
print(df.columns[:5])

#distributing features
X=df.drop('Price',axis=1)
y=df['Price']

#spliting data
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.3,random_state=6)

#finding correlation
corr=X_train.corr()
print(corr)

#heatmap of correlation
import seaborn as sns
import matplotlib.pyplot as plt

plt.figure(figsize=(12,8))
sns.heatmap(corr,annot=True,cmap='viridis')







# --------------
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

# Code starts here

regressor=LinearRegression()

#fitting the model
regressor.fit(X_train,y_train)

#making prediction
y_pred=regressor.predict(X_test)

#checking R^2 score
r2=r2_score(y_test,y_pred)
print(r2)


# --------------
from sklearn.linear_model import Lasso

# Code starts here

#now using lasso
lasso=Lasso()

#fitting model using lass0
lasso.fit(X_train,y_train)

#making predictions
lasso_pred=lasso.predict(X_test)

#checking R^2 score
r2_lasso=r2_score(y_test,lasso_pred)
print(r2_lasso)


# --------------
from sklearn.linear_model import Ridge

# Code starts here

#now using ridge to improve model
ridge=Ridge()

#fitting model using ridge
ridge.fit(X_train,y_train)

#making predictions using ridge
ridge_pred=ridge.predict(X_test)

#checking R^2 score
r2_ridge=r2_score(y_test,ridge_pred)
print(r2_ridge)

# Code ends here


# --------------
from sklearn.model_selection import cross_val_score
import numpy as np
#Code starts here
regressor=LinearRegression()

#using cross validation
score = cross_val_score(regressor,X_train,y_train,cv=10)

#calculating mean of scores
mean_score=np.mean(score)
print(mean_score)







# --------------
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline

#Code starts here

#now using polynomial features
model=make_pipeline(PolynomialFeatures(2), LinearRegression())

#fitting the model
model.fit(X_train,y_train)

#making predictions
y_pred=model.predict(X_test)

#checking r2_score
r2_poly=r2_score(y_test,y_pred)
print(r2_poly)







