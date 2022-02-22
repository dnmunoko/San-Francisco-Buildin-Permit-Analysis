# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
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

#Convert the time in days from Datetime format to integer
permit['wait_time']=permit['wait_time'].dt.days

# 6. Exploratory Data Analysis

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
fig=plt.figure(figsize=(8,6))
sns.catplot(x='Permit Type',hue='Current Status',kind='count',data=pmt_d)
plt.xlabel('Current Status')
plt.xticks()
plt.ylabel('Number of permits')
plt.title('Current status  permit types')
plt.xticks()
plt.show()

