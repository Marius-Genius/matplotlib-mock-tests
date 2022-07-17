#!/usr/bin/env python
# coding: utf-8

# <a href="https://colab.research.google.com/github/hse-mlwp-2022/hw4-VladimirBakunov/blob/main/regression_exercise.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

# # Regression
# 
# ## Important notes
# 
# 1. *When you open this file on GitHub, copy the address to this file from the address bar of your browser. Now you can go to [Google Colab](https://colab.research.google.com/), click `File -> Open notebook -> GitHub`, paste the copied URL and click the search button (the one with the magnifying glass to the right of the search input box). Your personal copy of this notebook will now open on Google Colab.*
# 2. *Do not delete or change variable names in the code cells below. You may add to each cell as many lines of code as you need, just make sure to assign your solution to the predefined variable(s) in the corresponding cell. Failing to do so will make the tests fail.*
# 3. *To save your work, click `File -> Save a copy on GitHub` and __make sure to manually select the correct repository from the dropdown list__.*
# 4. *If you mess up with this file and need to start from scratch, you can always find the original notebook [here](https://github.com/hse-mlwp-2022/assignment4-template/blob/main/regression_exercise.ipynb). Just open it in Google Colab (see note 1) and save to your repository (see note 3). Remember to backup your code elsewhere, since this action will overwrite your previous work.*
# 
# ## Initialization
# 
# ### Import the libraries you need in the cell below

# In[56]:


# Place your code here to import the libraries, e.g. pandas, numpy, sklearn, etc.
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
import statsmodels.api as sm


# ### 0. Find your task
# Follow the [link](https://docs.google.com/spreadsheets/d/194gX8uSUyqv_aQbJi8_TYuIgXHsDBtMDCofQ1uJ4GvA/edit?usp=sharing) to a Google Sheet with a list of students. Locate your name on the list and take note of the corresponding Student ID in the first column. Fill it in the cell below and run the cell. If you can't find yourself on the list, consult your course instructor.

# In[2]:


### BEGIN YOUR CODE

Student_ID = 10

### END YOUR CODE


# Now run the next cell. It will print all information for you.

# In[3]:


task_id = None if Student_ID is None else Student_ID % 5 if Student_ID % 5 > 0 else 5
_model_power = None if Student_ID is None else (Student_ID % 4) + 3
if task_id is not None:
    print(f"TASKID is {task_id}")
    print(f"Please, choose a dataset No {task_id} below")
    print(f"Your second model must be of power p = {_model_power}")
else:
    print("Please, enter your Student ID in the cell above!")


# #### Datasets
# 
# 1. Poultry meat consumption in Europe, kilograms per person per year
# 
# |     |     |     |     |     |     |     |     |     |     |     |
# | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | 
# | Year | 2000 | 2001 | 2002 | 2003 | 2004 | 2005 | 2006 | 2007 | 2008 | 2009 | 
# | Consumption | 16.0 | 17.9 | 18.6 | 18.3 | 19.0 | 19.3 | 19.2 | 20.3 | 21.1 | 21.9 | 
# 
# 2. Sugar consumption in Russia, grams per person per day
# 
# |     |     |     |     |     |     |     |     |     |     |     |
# | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | 
# | Decade | 1950 | 1960 | 1970 | 1980 | 1990 | 2000 | 2015 |
# | Consumption | 32 | 85 | 115 | 130 | 130 | 96 | 107 |
# 
# 3. Poultry meat consumption in Asia, kilograms per person per year
# 
# |     |     |     |     |     |     |     |     |     |     |     |
# | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | 
# | Year | 2000 | 2001 | 2002 | 2003 | 2004 | 2005 | 2006 | 2007 | 2008 | 2009 | 
# | Consumption | 6.7 | 6.6 | 6.8 | 7.0 | 7.0 | 7.5 | 7.7 | 8.2 | 8.6 | 8.8 | 
# 
# 4. Poultry meat consumption in Africa, kilograms per person per year
# 
# |     |     |     |     |     |     |     |     |     |     |     |
# | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | 
# | Year | 2000 | 2001 | 2002 | 2003 | 2004 | 2005 | 2006 | 2007 | 2008 | 2009 | 
# | Consumption | 4.2 | 4.3 | 4.5 | 4.7 | 4.6 | 4.7 | 4.8 | 5.2 | 5.4 | 5.5 | 
# 
# 5. Demographic situation in Russia, number of marriages per 1000 people per year
# 
# |     |     |     |     |     |     |     |     |     |     |     |
# | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | 
# | Year | 2011 | 2012 | 2013 | 2014 | 2015 | 2016 | 2017 | 2018 | 2019 | 2020 |
# | Marriages per 1000 population | 9.2 | 8.5 | 8.5 | 8.4 | 7.9 | 6.7 | 7.1 | 6.1 | 6.3 | 5.3 |
# 

# ### 1. Define a pandas dataset with the data for your task
# [This](https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.html) documentation might help.
# 
# **Make sure to normalize your $x$ variable, i.e. replace years with sequential numbers 0, 1, ...**

# In[248]:


# Place your code here to create the dataset here

data = {'year': [2000,	2001,	2002,	2003,	2004,	2005,	2006,	2007,	2008,	2009], 'marriages_per_1000': [9.2,	8.5,	8.5,	8.4,	7.9,	6.7,	7.1,	6.1,	6.3,	5.3]}
df = pd.DataFrame(data)
df['year'] = df['year'] - 2000
df


# ## First regression model
# 
# You should build the following model:
# 
# $$ y_1 = \theta_2 \cdot x^2 + \theta_1 \cdot x + \theta_0 $$
# 
# where $y$ is the response variable and $x$ is the explanatory variable (see description of your dataset).
# 
# ### 2. Define feature matrix $X$ for the first model (1 point)
# 
# It should be a `numpy` array or a `pandas` dataframe

# In[253]:


x = df.year.values
y = df.marriages_per_1000.values
feature_matrix_X = np.vstack([np.ones(len(x)), x, x**2]).T # Place your code here instead of '...'
feature_matrix_X


# ### 3. Train first regression model with OLS method by using matrix multiplications (2 points)
# 
# Use the entire dataset for training. You can find the formula on our lectures and in the seminar notebook.
# 
# `first_model_coeffs` should be an iterable, e.g. a list or a numpy array
# 
# $$ (X^TX)^{-1}X^TY $$

# In[142]:


first_model_coeffs = np.linalg.inv(feature_matrix_X.T.dot(feature_matrix_X)).dot(feature_matrix_X.T).dot(y) # Place your code here instead of '...'
first_model_coeffs = [round(i, 3) for i in first_model_coeffs]
print(f"Coefficints of the first regression model are '{first_model_coeffs}'")


# ## Second regression model
# 
# Choose the power $p$ of your model (see step 0 above). You should build the following model:
# 
# $$ y_2 = \sum_{i=1}^{p}{\theta_i \cdot x^p} $$
# 
# where $y$ is the response variable and $x$ is the explanatory variable (see description of your dataset) and $p$ is the power of the model.
# 
# ### 4. Train second regression model with OLS method using `stats.models.regression` module (2 points)
# 
# `second_model_coeffs` should be an iterable, e.g. a list or a numpy array
# 
# I chose the power 3 (to make the model more accurate)

# In[241]:


feature_matrix_X2 = np.vstack([np.ones(len(x)), x, x**2, x**3, x**4]).T


second_model_coeffs = sm.OLS(y, feature_matrix_X2).fit().params # Place your code here instead of '...'
print(f"Coefficints of the second regression model are '{second_model_coeffs}'")


# ## Third regression model
# 
# You should build the following model:
# 
# $$ y_3 = \theta_1 \cdot x + \theta_0 $$
# 
# where $y$ is the response variable and $x$ is the explanatory variable (see description of your dataset).
# 
# ### 5. Train third regression model with gradient descent (3 points, optional)
# 
# You can write your own function for gradient descent or find one on the Internet. It should be possible to change the initial value and learning rate.
# 
# `third_model_coeffs` should be an iterable, e.g. a list or a numpy array

# In[265]:


def gradientDescent(x, y, theta, alpha, m, numIterations):
    xTrans = x.transpose()
    for i in range(0, numIterations):
        hypothesis = np.dot(x, theta)
        loss = hypothesis - y
        cost = np.sum(loss ** 2) / (2 * m)
        gradient = np.dot(xTrans, loss) / m
        theta = theta - alpha * gradient
    return theta

feature_matrix_X3 = np.vstack([np.ones(len(x)), x]).T
m, n = np.shape(feature_matrix_X3)
alpha = 0.0005
theta = np.ones(n)
third_model_coeffs = gradientDescent(feature_matrix_X3, y, theta, 0.0005, m, 100000)



print(f"Coefficints of the third regression model are '{third_model_coeffs}'")


# ## Error estimation
# 
# ### 6. Calculate MSE and RMSE for all your regression models (2 points)
# 
# Error estimations should be floating point numbers

# In[263]:


def calculation_of_MSE(feature_matrix, y, coefficints):
  y_expected = feature_matrix @ coefficints
  MSE = np.sum((y_expected - y)**2)/len(feature_matrix)
  return MSE 

def calculation_of_RMSE(MSE):
  RMSE = MSE**(1/2)
  return RMSE


first_model_mse = calculation_of_MSE(feature_matrix_X, y, first_model_coeffs) # Place your code here instead of '...'
second_model_mse = calculation_of_MSE(feature_matrix_X2, y, second_model_coeffs) # Place your code here instead of '...'
third_model_mse = calculation_of_MSE(feature_matrix_X3, y, third_model_coeffs) # Place your code here instead of '...' (optional)

first_model_rmse = calculation_of_RMSE(first_model_mse) # Place your code here instead of '...'
second_model_rmse = calculation_of_RMSE(second_model_mse) # Place your code here instead of '...'
third_model_rmse = calculation_of_RMSE(third_model_mse) # Place your code here instead of '...' (optional)


# ## Visualization
# 
# ### 7. Use `matplotlib` to visualize your results (graded manually, exam)
# 
# You should build a single plot with all your models (2 or 3) drawn as curves/lines of different type and color. Additional points if you make the curves look smooth. Draw your dataset as dots on the same plot, do not connect them with lines.

# In[262]:


# Place your code here
def plot_model():
  plt.scatter(x, y)
  plt.plot(x, first_model_coeffs[0] + first_model_coeffs[1]*x + first_model_coeffs[2]*(x**2), 'red')
  plt.plot(x, second_model_coeffs[0] + second_model_coeffs[1]*x + second_model_coeffs[2]*(x**2) + second_model_coeffs[3]*(x**3) + second_model_coeffs[4]*(x**4), 'green')
  plt.plot(x, third_model_coeffs[0] + third_model_coeffs[1]*x, 'blue')


  plt.xlabel('Year', fontsize=18)
  plt.ylabel('Marriages per 1000 people', fontsize=16)
  plt.show()

plot_model()

# ### 8. Prepare to discuss your results with the teacher (exam)
# 
# Which model is better? Why? What else can you do to make the predictions better?

# In[ ]:


#

