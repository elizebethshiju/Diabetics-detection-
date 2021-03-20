#!/usr/bin/env python
# coding: utf-8

# # Correlation Analysis for Diabetes Detection
# This notebook contain the predictions made for diabetic and non-diabetic patients.

# In[2]:


import pandas as pd
from pandas import DataFrame


# In[39]:


bank = pd.read_csv('diabetes.csv')
bank.head()


# In[40]:


bank.shape


# In[41]:


bank.describe


# In[42]:


bank.info()


# # Finding missing values
# 

# In[43]:


bank.isnull().sum()


# In[44]:



# axis=0 to remove rows having null values and inplace=True to perform operation on the dataframe without creating new variable.
bank.dropna(axis=0,inplace=True)


# # Detecting Outliers¶
# 

# In[48]:


columns = bank.columns.tolist()
bank.boxplot(column=columns)

#For removing outliers

def remove_outlier(df_in, col_name):
    q1 = df_in[col_name].quantile(0.25)
    q3 = df_in[col_name].quantile(0.75)
    iqr = q3-q1 #Interquartile range
    fence_low  = q1-1.5*iqr
    fence_high = q3+1.5*iqr
    df_out = df_in.loc[(df_in[col_name] > fence_low) & (df_in[col_name] < fence_high)]
    return df_out


# In[52]:



bank = remove_outlier(bank,'Glucose')
bank = remove_outlier(bank,'BloodPressure')
bank.boxplot(column=columns)


# In[53]:


list(bank.columns)


# # convert any obvious categorical variables to categories:

# In[54]:


bank['Outcome'] = bank['Outcome'].astype('category')
bank.describe(include='category')


# # Scatterplots 
# 
# To create a bare-bones scatterplot, we must do four things:
# 
# Load the seaborn library
# Specify the source data frame
# Set the x axis, which is generally the name of a predictor/independent variable
# Set the y axis, which is generally the name of a response/dependent variable

# In[55]:


import seaborn as sns
sns.scatterplot(x="Glucose", y="BloodPressure", data=bank);


# # Adding labels 
# Here I assign the results of the scatterplot() call to a variable called ax and then set various properties of ax. I end the last line of the code block with a semicolon to suppress return values:

# In[56]:


ax = sns.scatterplot(x="Glucose", y="BloodPressure", data=bank)
ax.set_title("Glucose vs. BloodPressure")
ax.set_xlabel("BloodPressure");


# # Adding a best fit line 
# The easiest way to "add" a best-fit line to a scatterplot is to use a different plotting method. Seaborn's lmplot() method is one possibility:

# In[57]:


sns.lmplot(x="BloodPressure", y="Glucose", data=bank);


# # Adding color as a third dimension
# 

# In[58]:


sns.lmplot(x="BloodPressure", y="Glucose", hue="Outcome", data=bank);


# # CORRELATION ANALYSIS

# In[59]:


# to find the Coefficient of correlation
from scipy import stats
stats.pearsonr(bank['Glucose'], bank['BloodPressure'])


# In[60]:


#to find the correlation matrix
cormat = bank.corr()
round(cormat,2)


# In[64]:


#Correlation matrix to heat map 
import seaborn as sns
import matplotlib.pyplot as plt

plt.figure(figsize=(16, 6))
heatmap = sns.heatmap(bank.corr(), vmin=-1, vmax=1, annot=True)
heatmap.set_title('Correlation Heatmap', fontdict={'fontsize':12}, pad=12);


# # EDA using pairplots

# In[66]:


sns.pairplot(bank, hue='Outcome')


# In[ ]:




