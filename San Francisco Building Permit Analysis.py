# -*- coding: utf-8 -*-
"""
Created on Tue Feb 22 00:36:24 2022

@author: Dorcas
"""

#Data manipulation libraries
import pandas as pd
import numpy as np

# Visualisation libraries
from matplotlib import pyplot as plt
%matplotlib inline
import seaborn as sns

# Machine Learning Models
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor,GradientBoostingRegressor
from sklearn.preprocessing import  StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn import tree

# Hyperparameter tuning
from sklearn.model_selection import GridSearchCV,train_test_split,RandomizedSearchCV

# Imputing missing values and scaling values
from sklearn.impute import SimpleImputer
import warnings
warnings.filterwarnings("ignore")

# 1. Load Data

permit = pd. read_csv("C:/Users/Dorcas/Documents/UPitt/Big Data and Forecasting/assignment3_Building_Permits-1.csv",encoding='latin1')
dict_sf = pd. read_csv("C:/Users/Dorcas/Documents/UPitt/Big Data and Forecasting/assignment3_DataDictionaryBuildingPermit.csv",encoding='latin1')

# 2. Explore Data

permit.head()

permit.tail()

dict_sf

#Data Type Info
permit.info()

#Checking length of dataset/ number of rows and columns
print(permit.shape)

#Statistical Information
permit.describe()

# Descriptive statistics (for object)
permit.describe(include=['object']).T

# 3.Transform Dates

#Transform Dates
permit['Issued Date'] = pd.to_datetime(permit['Issued Date'])
permit['Filed Date']  = pd.to_datetime(permit['Filed Date'])

# 4. Missing Values

#Checking for missing values
permit.isnull().sum()

#Drop Missing Columns
#We will use percentage of missing values
#Dropping all missing data would lead to loss of a lot of data
mis_col = permit.isnull().sum()/len(permit)
mis_col

#Find percentage of missing values
mis_perc = (mis_col*100).round(3)
mis_perc

#Create missing value table
mis_table = pd.concat([mis_col, mis_perc],axis=1)
mis_table

#Rename Table
mis_table1 = mis_table.rename(columns = {0 : 'Missing Values', 1 : 'Percentage'})
mis_table1

#Remove columns with 80% missing values.
#8 columns removed
missing_columns = list(mis_table1[mis_table1['Percentage']>80].index)
print('We will remove %d columns'%len(missing_columns))
print('The columns to remove are \n %s'%missing_columns)

#Drop missing columns
permit.drop(columns=list(missing_columns),inplace=True)

#Check shape to determine how many columns were dropped
#We now have 35 columns remaining, 8 were dropped
permit.shape

#Drop rows with missing values
permit.dropna(axis=0,inplace=True)

#Check if we have any missing rows
permit.isnull().any().sum()

permit.shape

# 5. Date Manipulation

#Find wait time for issuance of permit
permit['wait_time'] = permit['Issued Date'] - permit['Filed Date']

permit['wait_time'].head()

#Convert wait time from Datetime format to integer
permit['wait_time']=permit['wait_time'].dt.days

# 6. Exploratory Data Analysis

#Explore target variable
permit['wait_time'].describe()

#We need to examine the max 1262
#Permit type 3 which is additions, alterations, or repairs has the longest wait time
permit.loc[permit['wait_time']== 1262] 

#We need to examine min 0
#Permit type 8 which is otc alterations permit has the shortest wait time
permit.loc[permit['wait_time']== 0][:5]

#Finding out the time it takes to process permits
nu_days=permit['wait_time'].value_counts(sort=True)

#About 50,000 permits processed in less than a day
nu_days.head()

#Less records take longer to be issued
nu_days.tail()

#Visualize Numerical Data
#Data distribution is positively skewed
permit['wait_time'].hist(color='green', bins=40, figsize=(8,4))
plt.xlabel('Wait Time')
plt.ylabel('Number of Records')
plt.title('Distribution of Wait Time')

#We will use log transformation to convert wait time curve to uniform data distribution
#permit['wait_time'] = np.log(permit['wait_time']+1)

#Confirm if wait time is normal distribution
#Data distribution is positively skewed
#permit['wait_time'].hist(color='green', bins=40, figsize=(8,4))
#plt.xlabel('Wait Time')
#plt.ylabel('Number of Records')
#plt.title('Distribution of Wait Time')

#sns.distplot(permit['wait_time'])

#Visualize Categorical Data Permit Type,Permit Type Definition,Current Status, Existing Construction Type
#a.) Permit Type
#Permit type 2 has the longest number of wait time, while permit tyoe 8 has the least number of wait time.
fig=plt.figure(figsize=(8,6))
sns.barplot(x=permit['Permit Type'],y=permit['wait_time'],hue='Permit Type',data=permit)
plt.xlabel('Permit Type')
plt.xticks(size=14)
plt.ylabel('Wait Time')
plt.title('Wait Time per Permit Type')

plt.show();

#Distribution of Permit Type
#Permit type 8 has the highest number of records followed by permit type 8. The rest 2,4,5,6,7 have very few records.
fig=plt.figure(figsize=(8, 8))
plt.hist(permit['Permit Type'], bins = 20, edgecolor = 'black');
plt.xlabel('Permit Type'); 
plt.ylabel('Number of Records'); plt.title('Permit Type Distribution');

#b.) Current Status
pmt_d =permit[permit['Current Status'].isin(['issued','revoked','incomplete']) ] 

#Current Data
#Issued permits has the highest count
fig=plt.figure(figsize=(8,6))
sns.catplot(x='Permit Type',hue='Current Status',kind='count',data=pmt_d)
plt.xlabel('Current Status')
plt.xticks()
plt.ylabel('Number of permits')
plt.title('Current status  permit types')
plt.xticks()
plt.show()

# A Count plot showing distribution of Permit Type Definition with permit type
#otc alteration permit type 8 has the highest number of records
fig=plt.figure(figsize=(8,6))
sns.catplot(y='Permit Type Definition',hue='Permit Type',kind='count',data=permit)
plt.xlabel('Number of Permits')
plt.xticks()
plt.ylabel('Permit Type Definition')
plt.title('Number of permits per permit type definition')
plt.show()

#Construction Type 5 has the highest number of permits
fig=plt.figure(figsize=(8,6))
sns.catplot(x='Permit Type',hue='Existing Construction Type',kind='count',data=permit)
plt.xlabel('Permit Types')
plt.ylabel('Count of Existing Construction types')
plt.title('Count of Permit types for existing construction types')
plt.xticks()
plt.show()

# 7. Dealing With Outliers

#a.) Estimated Cost
#Skewed to the right
sns.kdeplot(data=permit['Estimated Cost'] ,shade=False,alpha=0.8)
plt.show()

#b.) Plansets
#Skewed to the right
sns.kdeplot(data=permit['Plansets'] ,shade=False,alpha=0.8)
plt.show()

#c.) Revised Costs
#Skewed to the right
sns.kdeplot(data=permit['Revised Cost'] ,shade=False,alpha=0.8)
plt.show()

#d.) Existing Units
#Skewed to the right
sns.kdeplot(data=permit['Existing Units'] ,shade=False,alpha=0.8)
plt.show()

#e.) Proposed Units
#Skewed to the right
sns.kdeplot(data=permit['Proposed Units'] ,shade=False,alpha=0.8)
plt.show()

#Remove Outliers
#a.) Estimated Cost
Q1 = permit['Estimated Cost'].describe()['25%']
Q3 = permit['Estimated Cost'].describe()['75%']
IQR = Q3 - Q1
permit = permit[(permit['Estimated Cost'] > (Q1 - 3 * IQR)) &
            (permit['Estimated Cost'] < (Q3 + 3 * IQR))]

#b.) Plansets
Q1 = permit['Plansets'].describe()['25%']
Q3 = permit['Plansets'].describe()['75%']
IQR = Q3 - Q1
permit = permit[(permit['Plansets'] > (Q1 - 3 * IQR)) &
            (permit['Plansets'] < (Q3 + 3 * IQR))]

#c.) Revised Costs
Q1 = permit['Revised Cost'].describe()['25%']
Q3 = permit['Revised Cost'].describe()['75%']
IQR = Q3 - Q1
permit = permit[(permit['Revised Cost'] > (Q1 - 3 * IQR)) &
            (permit['Revised Cost'] < (Q3 + 3 * IQR))]

#d.) Existing Units
Q1 = permit['Existing Units'].describe()['25%']
Q3 = permit['Existing Units'].describe()['75%']
IQR = Q3 - Q1
permit = permit[(permit['Existing Units'] > (Q1 - 3 * IQR)) &
            (permit['Existing Units'] < (Q3 + 3 * IQR))]

#e.) Proposed Units
Q1 = permit['Proposed Units'].describe()['25%']
Q3 = permit['Proposed Units'].describe()['75%']
IQR = Q3 - Q1
permit = permit[(permit['Proposed Units'] > (Q1 - 3 * IQR)) &
            (permit['Proposed Units'] < (Q3 + 3 * IQR))]

#Plot figures after deleting outliers
#a.) Estimated Cost
sns.kdeplot(data=permit['Estimated Cost'] ,shade=False,alpha=0.8)
plt.show()

#b.) Plansets
sns.kdeplot(data=permit['Plansets'] ,shade=False,alpha=0.8)
plt.show()

#c.) Revised Costs
sns.kdeplot(data=permit['Revised Cost'] ,shade=False,alpha=0.8)
plt.show()

#d.) Existing Units
sns.kdeplot(data=permit['Existing Units'] ,shade=False,alpha=0.8)
plt.show()

#e.) Proposed Units
sns.kdeplot(data=permit['Proposed Units'] ,shade=False,alpha=0.8)
plt.show()

# 8.) Correlation Matrix

#Correlation Matrix
corr = permit.corr()
plt.figure(figsize=(20,10))
sns.heatmap(corr, annot=True, cmap='coolwarm')

correlation_matrix =permit.corr()['wait_time'].sort_values()

#Top 7 highly negative correlated
correlation_matrix.head(7)

#Top 7 highly positive correlated 
correlation_matrix.tail(7)

#Pre-Processing
category_col = ['Permit Type','Street Number','Existing Construction Type','Zipcode','Supervisor District']

permit[category_col]=permit[category_col].astype('str')

#Verify data types
permit.dtypes

#Drop Unnecessary Columns
permit.drop(columns =list(['Filed Date','Issued Date','Record ID','Location',
                           'Number of Existing Stories','Estimated Cost',
                           'Existing Units','Current Status',]),axis=1,inplace=True) 

permit.shape



# 9. Data Preprocessing

#Data Preprocessing
    #Separate the data and Label
x = permit.drop('wait_time', axis=1)
y = permit.loc[:, 'wait_time']

# Select categorical columns with relatively low cardinality
cat_cols = [cname for cname in x.columns 
                    if x[cname].nunique() < 10 and x[cname].dtype == "string"]

#Get dummies
cat_cols1= pd.get_dummies(x[cat_cols])

cat_cols1.apply(pd.to_numeric)

#Numerical columns
num_cols = [cname for cname in x.columns 
                  if x[cname].dtype in ['int64', 'float64']]

scaler = StandardScaler()
x[num_cols] = scaler.fit_transform(x[num_cols] )

#Numerical columns have minimum skewness
x[num_cols].skew(axis=0)

num_data = pd.DataFrame(x[num_cols])

#Combine categorical and numerical
X =pd.concat([cat_cols1, num_data],axis=1) 

# Convert y to one-dimensional array (vector)
y = np.array(y).reshape((-1, ))

#Split the data into train and test data
X_train, X_test, y_train, y_test,  = train_test_split(x, y, test_size=0.2, random_state=3)
print(y.shape, y_train.shape, y_test.shape)

#To compare the shape of different testing and training sets
   #printing shapes of testing and training sets :
print("shape of original dataset :", permit.shape)
print("shape of input - training set", X_train.shape)
print("shape of output - training set", y_train.shape)
print("shape of input - testing set", X_test.shape)
print("shape of output - testing set", y_test.shape)

# 10. Model Training

# a). Decision Tree

# define our basic tree classifier
model_tree = tree.DecisionTreeClassifier()

# fit it to the training data
model_tree.fit(X_train, y_train)

