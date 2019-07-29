# --------------
import pandas as pd
from sklearn import preprocessing

#path : File path

# Code starts here

# read the dataset
dataset=pd.read_csv(path)


# look at the first five columns
print(dataset.head(5))

# Check if there's any column which is not useful and remove it like the column id
dataset.drop("Id",axis=1,inplace=True)
print(dataset.columns)

# check the statistical description
print(dataset.describe())


# --------------
# We will visualize all the attributes using Violin Plot - a combination of box and density plots
import seaborn as sns
from matplotlib import pyplot as plt

#names of all the attributes 
cols=dataset.columns

#number of attributes (exclude target)
size=dataset.drop("Cover_Type",axis=1).shape[1]
print(size)
#x-axis has target attribute to distinguish between classes
x=dataset['Cover_Type']

#y-axis shows values of an attribute
y=dataset.drop("Cover_Type",axis=1)

#Plot violin for all attributes
for i in range(size-1):
   sns.violinplot(x=dataset[cols[i]], hue=y,data=dataset, palette="muted")


# --------------
import numpy

upper_threshold = 0.5
lower_threshold = -0.5



# Code Starts Here

# create a subset of dataframe with only the first 10 features
subset_train = dataset.iloc[:, :10]

# Calculate the Pearson correlation
data_corr = subset_train.corr()

# Plot a heatmap
f, ax = plt.subplots(figsize = (10,8))
sns.heatmap(data_corr,vmax=0.8,square=True);

# List the correlation pairs
correlation = data_corr.unstack().sort_values(kind='quicksort')

# Select the highest correlation pairs using slicing
corr_var_list = correlation[((correlation>upper_threshold) | (correlation<lower_threshold)) & (correlation!=1)]

# Code ends here



# --------------
#Import libraries 
from sklearn import cross_validation
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
# Identify the unnecessary columns and remove it 
dataset.drop(columns=['Soil_Type7', 'Soil_Type15'], inplace=True)


X=dataset.iloc[:,:-1]
y=dataset.iloc[:,-1]

X_train,X_test,y_train,y_test = cross_validation.train_test_split(X, y, test_size=0.2, random_state=0)

num_feat_cols = ['Elevation', 'Aspect', 'Slope','Horizontal_Distance_To_Hydrology','Vertical_Distance_To_Hydrology', 'Horizontal_Distance_To_Roadways','Hillshade_9am','Hillshade_Noon', 'Hillshade_3pm','Horizontal_Distance_To_Fire_Points']

cat_feat_cols = list(set(X_train.columns) - set(num_feat_cols))
scaler=StandardScaler()

X_train_temp=X_train[num_feat_cols].copy()

X_test_temp=X_test[num_feat_cols].copy()

X_train_temp[num_feat_cols]=scaler.fit_transform(X_train_temp[num_feat_cols])

X_test_temp[num_feat_cols]=scaler.fit_transform(X_test_temp[num_feat_cols])


X_train1=pd.concat([X_train_temp,X_train.loc[:,cat_feat_cols]],axis=1)

print(X_train1.head())

X_test1=pd.concat([X_test_temp,X_test.loc[:,cat_feat_cols]],axis=1)

print(X_test1.head())

scaled_features_train_df=X_train1
#Standardized
#Apply transform only for non-categorical data
scaled_features_test_df=X_test1

#Concatenate non-categorical data and categorical



# --------------
from sklearn.feature_selection import SelectPercentile
from sklearn.feature_selection import f_classif


# Write your solution here:

# Code starts here
skb = SelectPercentile(score_func=f_classif,percentile=20)
predictors = skb.fit_transform(X_train1, y_train)
scores = list(skb.scores_)

Features = scaled_features_train_df.columns

dataframe = pd.DataFrame({'Features':Features,'Scores':scores})

dataframe=dataframe.sort_values(by='Scores',ascending=False)

top_k_predictors = list(dataframe['Features'][:predictors.shape[1]])

print(top_k_predictors)

# Code Ends here


# --------------
from sklearn.multiclass import OneVsRestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, precision_score

clf = OneVsRestClassifier(LogisticRegression())
clf1=OneVsRestClassifier(LogisticRegression())
model_fit_all_features =clf.fit(X_train , y_train)
predictions_all_features=clf.predict(X_test)

score_all_features= accuracy_score(y_test,predictions_all_features )

print(scaled_features_train_df.columns[skb.get_support()])

X_new = scaled_features_train_df.loc[:,skb.get_support()]
X_test_new=scaled_features_test_df.loc[:,skb.get_support()]

model_fit_top_features  =clf1.fit(X_new , y_train)
predictions_top_features=clf1.predict(X_test_new)

score_top_features= accuracy_score(y_test,predictions_top_features )


