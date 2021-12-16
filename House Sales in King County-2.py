#!/usr/bin/env python
# coding: utf-8

# # House Sales Prediction 
# 

# **Dataset Description
# 
# This dataset contains house sale prices for King County, which includes Seattle. It includes homes sold between May 2014 and May 2015.
# 
# It's a great dataset for evaluating simple regression models.
# 
# I chose this topic is because predicting the expected (house) price is a typical topic for business analyst. I think it is a great chance for us to have hands-on practice.
# 
# Along with house price (target) it consists of an ID, date, and 18 house features.
# 
# 
# 
# ***Feature Columns***
# 
# id - Unique ID for each home sold
# 
# date - Date of the home sale
# 
# price - Price of each home sold
# 
# bedrooms - Number of bedrooms
# 
# bathrooms - Number of bathrooms
# 
# sqft_living - Square footage of the apartments interior living space
# 
# sqft_lot - Square footage of the land space
# 
# floors - Number of floors
# 
# waterfront - A dummy variable for whether the apartment was overlooking the waterfront or not
# 
# view - An index from 0 to 4 of how good the view of the property was
# 
# condition - An index from 1 to 5 on the condition of the apartment,
# 
# grade - An index from 1 to 13, where 1-3 falls short of building construction and design, 7 has an average level of 
# 
# construction and design, and 11-13 have a high quality level of construction and design.
# 
# sqft_above - The square footage of the interior housing space that is above ground level
# 
# sqft_basement - The square footage of the interior housing space that is below ground level
# 
# yr_built - The year the house was initially built
# 
# yr_renovated - The year of the house’s last renovation
# 
# zipcode - What zipcode area the house is in
# 
# lat - Lattitude
# 
# long - Longitude
# 
# sqft_living15 - The square footage of interior housing living space for the nearest 15 neighbors
# 
# sqft_lot15 - The square footage of the land lots of the nearest 15 neighbors
# 
# 

# ***Data used  : https://www.kaggle.com/harlfoxem/housesalesprediction***

# # Problem Statement
# The goal of this statistical analysis is to help us understand the relationship between house features 
# and how these variables are used to predict house price.

# # Target group
# 
# People interested in home prices and real estate owners

# # Objective
# 
# - Predict the house price
# 
# - Using the the linear regression for predicted and actual rating

# # Exploratory Data Analysis (EDA)

# In[103]:


#import required libraries


import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

get_ipython().run_line_magic('matplotlib', 'inline')


# In[104]:


# Importing my Data with read_csv().

df = pd.read_csv('kc_house_data.csv')


# In[105]:


# Use info() to get a summary of the dataframe and its datatypes

df.info()


# In[106]:


# To look at the first few rows
#The property T is an accessor to the method transpose(),Switch index and columns.


df.head().T


# In[107]:


# to display some basic statistic details 
#The property T is an accessor to the method transpose(),Switch index and columns.
#The date column is not numeric and therefore does not appear in this details

df.describe().T


# In[108]:


#claening the data
# to find the number of NaN in each column of my data 

df.isnull().sum()


# In[109]:


# drop unnecessory featurs (Because it is not important to the target "price")

df = df.drop('date',axis=1)
df = df.drop('id',axis=1)
df = df.drop('zipcode',axis=1)


# In[48]:


# import first and end data  (after the drop few columns) 
#The property T is an accessor to the method transpose(),Switch index and columns.

df.T


# In[65]:


#only use for print index the columns

df.columns


# In[49]:


#visualizing house prices

# tha most of the prices are between 0 and around 1M with few outliers close to 8 million 

fig = plt.figure(figsize=(10,7))

fig.add_subplot(2,1,1)
sns.distplot(df['price'])

fig.add_subplot(2,1,2)
sns.boxplot(df['price'])

plt.tight_layout()


# In[50]:


#Relationship analysis

#use Seaborn heatmap to visualize the correlation matrix of data for feature selection to solve business problems.
#.corr()  is used to find the pairwise correlation of all columns in the dataframe. 
# the bool ‘True‘ value in  annot , for show the value  on each cell of the heatmap.


cormap = df.corr()
plt.figure(figsize=(20,15))
sns.heatmap(cormap, annot=True)
plt.show()


# In[51]:


# the correlation of all columns in the dataframe with main target (price ) 
# we find (sqft_living ,grade ,sqft_above,sqft_living15) are positively correlated with price 

df.corr()['price'].sort_values(ascending=False)


# In[52]:


#To Plot pairwise relationships in a dataset 
# We see the relationship between (bathrooms,sqft_living,grade, sqft_above,sqft_living15 ) and price are positively correlated 

sns.pairplot(df)

plt.show()


# # Dataset Preparation (Splitting and Scaling)

# Data is divided into the Train set and Test set. We use the Train set to make the algorithm learn the data’s behavior and then check the accuracy of our model on the Test set.
# 
# Features (X): The columns that are inserted into our model will be used to make predictions.
# 
# Prediction (y): Target variable that will be predicted by the features
# 

# In[68]:


X = df[[ 'bedrooms', 'bathrooms', 'sqft_living', 'sqft_lot', 'floors','waterfront', 'view', 'condition', 'grade', 'sqft_above','sqft_basement', 'yr_built', 'yr_renovated', 'lat', 'long','sqft_living15', 'sqft_lot15']]
y = df['price']

#splitting Train and Test 

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=0)

#print the data after split 

print ("train size={}, test_size={}, total_size={}".format(X_train.shape[0], X_test.shape[0], df.shape[0]))


# In[70]:


#scaling the data

from sklearn.preprocessing import StandardScaler

s_scaler = StandardScaler()
X_train = s_scaler.fit_transform(X_train.astype(np.float))
X_test = s_scaler.transform(X_test.astype(np.float))


# # Model Selection and Evaluation

# **Model : Linear Regression**

# In[85]:


#Create an instance of a LinearRegression() model named lr
#Train/fit lm on the training data


from sklearn.linear_model import LinearRegression

lr = LinearRegression()

lr.fit(X_train, y_train)


# In[86]:


# calculate and Print out the coefficients of the model

cdf = pd.DataFrame(lr.coef_, X.columns, columns=['Coefficients'])

cdf


# ## Predicting Test Data

# evaluate the model performance by predicting off the test values
# and use lm.predict() to predict off the X_test set of the data.

# In[87]:


predictions = lr.predict(X_test)
predictions


# In[83]:


plt.scatter(y_test,predictions)


# ### Model Evaluation¶

# #### compare actual output and predicted value to measure how far our predictions are from the real house prices.
# 
# we will evaluate our model performance by calculating the residual sum of squares and the explained variance score (R^2).
# and Calculate the Mean Absolute Error, Mean Squared Error, and the Root Mean Squared Error. 
# 

# In[101]:


#compare actual output values with predicted values


y_pred = lr.predict(X_test)
df = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})


# evaluate the performance of the algorithm (MAE - MSE - RMSE)

from sklearn import metrics

print('MAE:', metrics.mean_absolute_error(y_test, y_pred))  
print('MSE:', metrics.mean_squared_error(y_test, y_pred))  
print('RMSE:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))

print('VarScore:',metrics.explained_variance_score(y_test,y_pred))


# In[110]:



from sklearn.metrics import accuracy_score

accuracy = lr.score(X_train,y_train)

accuracy


# In[ ]:





# ## Thank you for listening
# 
# 
