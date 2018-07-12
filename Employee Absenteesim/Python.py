
# coding: utf-8

# In[ ]:


#Load libraries
import os
import pandas as pd
import numpy as np
from fancyimpute import KNN   
import matplotlib.pyplot as plt
from scipy.stats import chi2_contingency
import seaborn as sns
from random import randrange, uniform


# In[ ]:


#Load data
os.chdir("C:/Users/chinna/Desktop/Data Science/Project/Employee Absenteeism")


# In[ ]:


#read data
employee = pd.read_excel("Absenteeism.xls")


# # Missing value Analysis

# In[ ]:


#Create dataframe with missing percentage
missing_val = pd.DataFrame(employee.isnull().sum())

#Reset index
missing_val = missing_val.reset_index()

#Rename variable
missing_val = missing_val.rename(columns = {'index': 'Variables', 0: 'Missing_percentage'})

#Calculate percentage
missing_val['Missing_percentage'] = (missing_val['Missing_percentage']/len(employee))*100

#descending order
missing_val = missing_val.sort_values('Missing_percentage', ascending = False).reset_index(drop = True)


# In[ ]:


missing_val


# In[ ]:


#details of data
employee.describe()


# In[ ]:


#Missing value Imputation using median method
employee['Reason for absence'] = employee['Reason for absence'].fillna(employee['Reason for absence'].median())
employee['Month of absence'] = employee['Month of absence'].fillna(employee['Month of absence'].median())
employee['Transportation expense'] = employee['Transportation expense'].fillna(employee['Transportation expense'].median())
employee['Distance from Residence to Work'] = employee['Distance from Residence to Work'].fillna(employee['Distance from Residence to Work'].median())
employee['Service time']= employee['Service time'].fillna(employee['Service time'].median())
employee['Age'] = employee['Age'].fillna(employee['Age'].median())
employee['Work load Average/day ']= employee['Work load Average/day '].fillna(employee['Work load Average/day '].median())
employee['Hit target']= employee['Hit target'].fillna(employee['Hit target'].median())
employee['Disciplinary failure']= employee['Disciplinary failure'].fillna(employee['Disciplinary failure'].median())
employee['Education']= employee['Education'].fillna(employee['Education'].median())
employee['Son']= employee['Son'].fillna(employee['Son'].median())
employee['Social drinker']= employee['Social drinker'].fillna(employee['Social drinker'].median())
employee['Social smoker']= employee['Social smoker'].fillna(employee['Social smoker'].median())
employee['Pet']= employee['Pet'].fillna(employee['Pet'].median())
employee['Weight']= employee['Weight'].fillna(employee['Weight'].median())
employee['Height']= employee['Height'].fillna(employee['Height'].median())
employee['Body mass index']= employee['Body mass index'].fillna(employee['Body mass index'].median())
employee['Absenteeism time in hours']= employee['Absenteeism time in hours'].fillna(employee['Absenteeism time in hours'].median())


# In[ ]:


df = employee.copy()


# # Outlier Analysis

# In[ ]:


#Converting variable to factors
employee['ID'] = employee['ID'].astype('category')
employee['Reason for absence'] = employee['Reason for absence'].astype('category')
employee['Month of absence'] = employee['Month of absence'].astype('category')
employee['Day of the week'] = employee['Day of the week'].astype('category')
employee['Seasons'] = employee['Seasons'].astype('category')
employee['Disciplinary failure'] = employee['Disciplinary failure'].astype('category')
employee['Education'] = employee['Education'].astype('category')
employee['Social drinker'] = employee['Social drinker'].astype('category')
employee['Social smoker'] = employee['Social smoker'].astype('category')


# In[ ]:


Numeric = employee[['Transportation expense', 'Distance from Residence to Work',
                  'Service time', 'Age', 'Work load Average/day ', 'Hit target','Son','Pet', 
                  'Weight', 'Height', 'Body mass index','Absenteeism time in hours']]


# In[ ]:


#boxplot of transportation
plt.boxplot(Numeric['Transportation expense'])


# In[ ]:


#boxplot of Distance from residence to work
plt.boxplot(Numeric['Distance from Residence to Work'])


# In[ ]:


#boxplot of service time
plt.boxplot(Numeric['Service time'])


# In[ ]:


#boxplot of age
plt.boxplot(Numeric['Age'])


# In[ ]:


#boxplot of workload
plt.boxplot(Numeric['Work load Average/day '])


# In[ ]:


#boxplot of hit target
plt.boxplot(Numeric['Hit target'])


# In[ ]:


#boxplot of son
plt.boxplot(Numeric['Son'])


# In[ ]:


#boxplot of pet
plt.boxplot(Numeric['Pet'])


# In[ ]:


#boxplot of weight
plt.boxplot(Numeric['Weight'])


# In[ ]:


#boxplot of height
plt.boxplot(Numeric['Height'])


# In[ ]:


#boxplot of bmi
plt.boxplot(Numeric['Body mass index'])


# In[ ]:


#boxplot of absenteeism
plt.boxplot(Numeric['Absenteeism time in hours'])


# In[ ]:


for i in Numeric:
    print(i)
    q75, q25 = np.percentile(Numeric.loc[:,i], [75 ,25])
    iqr = q75 - q25

    min = q25 - (iqr*1.5)
    max = q75 + (iqr*1.5)
    print(min)
    print(max)


# In[ ]:


#Replace with NA
#Extract quartiles
q75, q25 = np.percentile(employee['Transportation expense'], [75 ,25])

#Calculate IQR
iqr = q75 - q25

#Calculate inner and outer fence
minimum = q25 - (iqr*1.5)
maximum = q75 + (iqr*1.5)

#Replace with NA
employee.loc[employee['Transportation expense'] < minimum,:'Transportation expense'] = np.nan
employee.loc[employee['Transportation expense'] > maximum,:'Transportation expense'] = np.nan


# In[ ]:


#Detect and replace with NA
#Extract quartiles
q75, q25 = np.percentile(employee['Service time'], [75 ,25])

#Calculate IQR
iqr = q75 - q25

#Calculate inner and outer fence
minimum = q25 - (iqr*1.5)
maximum = q75 + (iqr*1.5)

#Replace with NA
employee.loc[employee['Service time'] < minimum,:'Service time'] = np.nan
employee.loc[employee['Service time'] > maximum,:'Service time'] = np.nan


# In[ ]:


#Detect and replace with NA
#Extract quartiles
q75, q25 = np.percentile(employee['Age'], [75 ,25])

#Calculate IQR
iqr = q75 - q25

#Calculate inner and outer fence
minimum = q25 - (iqr*1.5)
maximum = q75 + (iqr*1.5)

#Replace with NA
employee.loc[employee['Age'] < minimum,:'Age'] = np.nan
employee.loc[employee['Age'] > maximum,:'Age'] = np.nan


# In[ ]:


#Detect and replace with NA
#Extract quartiles
q75, q25 = np.percentile(employee['Work load Average/day '], [75 ,25])

#Calculate IQR
iqr = q75 - q25

#Calculate inner and outer fence
minimum = q25 - (iqr*1.5)
maximum = q75 + (iqr*1.5)

#Replace with NA
employee.loc[employee['Work load Average/day '] < minimum,:'Work load Average/day '] = np.nan
employee.loc[employee['Work load Average/day '] > maximum,:'Work load Average/day '] = np.nan


# In[ ]:


#Detect and replace with NA
#Extract quartiles
q75, q25 = np.percentile(employee['Hit target'], [75 ,25])

#Calculate IQR
iqr = q75 - q25

#Calculate inner and outer fence
minimum = q25 - (iqr*1.5)
maximum = q75 + (iqr*1.5)

#Replace with NA
employee.loc[employee['Hit target'] < minimum,:'Hit target'] = np.nan
employee.loc[employee['Hit target'] > maximum,:'Hit target'] = np.nan


# In[ ]:


#Detect and replace with NA
#Extract quartiles
q75, q25 = np.percentile(employee['Pet'], [75 ,25])

#Calculate IQR
iqr = q75 - q25

#Calculate inner and outer fence
minimum = q25 - (iqr*1.5)
maximum = q75 + (iqr*1.5)

#Replace with NA
employee.loc[employee['Pet'] < minimum,:'Pet'] = np.nan
employee.loc[employee['Pet'] > maximum,:'Pet'] = np.nan


# In[ ]:


#Detect and replace with NA
#Extract quartiles
q75, q25 = np.percentile(employee['Height'], [75 ,25])

#Calculate IQR
iqr = q75 - q25

#Calculate inner and outer fence
minimum = q25 - (iqr*1.5)
maximum = q75 + (iqr*1.5)

#Replace with NA
employee.loc[employee['Height'] < minimum,:'Height'] = np.nan
employee.loc[employee['Height'] > maximum,:'Height'] = np.nan


# In[ ]:


#Detect and replace with NA
#Extract quartiles
q75, q25 = np.percentile(employee['Absenteeism time in hours'], [75 ,25])

#Calculate IQR
iqr = q75 - q25

#Calculate inner and outer fence
minimum = q25 - (iqr*1.5)
maximum = q75 + (iqr*1.5)

#Replace with NA
employee.loc[employee['Absenteeism time in hours'] < minimum,:'Absenteeism time in hours'] = np.nan
employee.loc[employee['Absenteeism time in hours'] > maximum,:'Absenteeism time in hours'] = np.nan


# In[ ]:


#Impute replaced outliers
employee['Transportation expense'] = employee['Transportation expense'].fillna(employee['Transportation expense'].median())
employee['Service time']= employee['Service time'].fillna(employee['Service time'].median())
employee['Height']= employee['Height'].fillna(employee['Height'].median())
employee['Pet']= employee['Pet'].fillna(employee['Pet'].median())
employee['Hit target']= employee['Hit target'].fillna(employee['Hit target'].median())
employee['Age'] = employee['Age'].fillna(employee['Age'].median())
employee['Work load Average/day ']= employee['Work load Average/day '].fillna(employee['Work load Average/day '].median())
employee['Absenteeism time in hours']= employee['Absenteeism time in hours'].fillna(employee['Absenteeism time in hours'].median())


# In[ ]:


employee['ID'] = df['ID']
employee['Reason for absence'] = df['Reason for absence']
employee['Month of absence'] = df['Month of absence']
employee['Day of the week'] = df['Day of the week']
employee['Seasons'] = df['Seasons']
employee['Distance from Residence to Work'] = df['Distance from Residence to Work']
employee['Disciplinary failure'] = df['Disciplinary failure']
employee['Education'] = df['Education']
employee['Son'] = df['Son']
employee['Social drinker'] = df['Social drinker']
employee['Social smoker'] = df['Social smoker']
employee['Weight'] = df['Weight']
employee['Body mass index'] = df ['Body mass index']


# In[ ]:


Missing = pd.DataFrame(employee.isnull().sum())
Missing


# In[ ]:


#converting datatypes
employee['ID'] = employee['ID'].astype('category')
employee['Reason for absence'] = employee['Reason for absence'].astype('category')
employee['Month of absence'] = employee['Month of absence'].astype('category')
employee['Day of the week'] = employee['Day of the week'].astype('category')
employee['Seasons'] = employee['Seasons'].astype('category')
employee['Disciplinary failure'] = employee['Disciplinary failure'].astype('category')
employee['Education'] = employee['Education'].astype('category')
employee['Social drinker'] = employee['Social drinker'].astype('category')
employee['Social smoker'] = employee['Social smoker'].astype('category')


# # Feature Selection

# In[ ]:


Numerical = employee[['Transportation expense', 'Distance from Residence to Work',
       'Service time', 'Age', 'Work load Average/day ', 'Hit target', 'Son',
       'Pet', 'Weight', 'Height', 'Body mass index']]


# In[ ]:


#correlation plot
corr = Numerical.corr()
#Set the width and hieght of the plot
f, ax = plt.subplots(figsize=(10, 8))

#Plot using seaborn library
sns.heatmap(corr, mask=np.zeros_like(corr, dtype=np.bool), cmap= sns.diverging_palette(220,10,as_cmap=True),annot=True,
            square=True, ax=ax)


# In[ ]:


df = employee.copy()


# In[ ]:


employee.groupby(["ID", "Absenteeism time in hours"]).size().unstack().plot(kind='bar', stacked=True, figsize=(10,10))


# In[ ]:


employee.groupby(["Reason for absence", "Absenteeism time in hours"]).size().unstack().plot(kind='bar', stacked=True, figsize=(10,10))


# In[ ]:


employee.groupby(["Month of absence", "Absenteeism time in hours"]).size().unstack().plot(kind='bar', stacked=True, figsize=(10,10))


# In[ ]:


employee.groupby(["Day of the week", "Absenteeism time in hours"]).size().unstack().plot(kind='bar', stacked=True, figsize=(10,10))


# In[ ]:


employee.groupby(["Education", "Absenteeism time in hours"]).size().unstack().plot(kind='bar', stacked=True, figsize=(10,10))


# In[ ]:


employee.groupby(["Disciplinary failure", "Absenteeism time in hours"]).size().unstack().plot(kind='bar', stacked=True, figsize=(10,10))


# In[ ]:


employee.groupby(["Social drinker", "Absenteeism time in hours"]).size().unstack().plot(kind='bar', stacked=True, figsize=(10,10))


# In[ ]:


employee.groupby(["Social smoker", "Absenteeism time in hours"]).size().unstack().plot(kind='bar', stacked=True, figsize=(10,10))


# In[ ]:


#Droppong variables
employee = employee.drop(['ID', 'Seasons','Education','Hit target','Son','Pet','Height','Body mass index',
                          'Age','Disciplinary failure'], axis=1)


# In[ ]:


employee.shape


# # Feature Scaling

# In[ ]:


cnames = ['Transportation expense','Distance from Residence to Work','Service time','Work load Average/day ','Weight']


# In[ ]:


employee['Transportation expense'] = employee['Transportation expense'].astype('int64')
employee['Distance from Residence to Work'] = employee['Distance from Residence to Work'].astype('int64')
employee['Service time'] = employee['Service time'].astype('int64')
employee['Work load Average/day '] = employee['Work load Average/day '].astype('int64')
employee['Weight'] = employee['Weight'].astype('int64')


# In[ ]:


#Nomalisation
for i in cnames:
    print(i)
    employee[i] = (employee[i] - min(employee[i]))/(max(employee[i]) - min(employee[i]))


# # MODEL APPLICATION

# In[ ]:


#DECISION TREE REGRESSION 
from sklearn.cross_validation import train_test_split
from sklearn.tree import DecisionTreeRegressor


# In[ ]:


train, test = train_test_split(employee, test_size = 0.2)


# In[ ]:


train.shape,test.shape


# In[ ]:


fit = DecisionTreeRegressor(max_depth = 2).fit(train.iloc[:,0:10],train.iloc[:,10])


# In[ ]:


prediction_dt = fit.predict(test.iloc[:,0:10])


# In[ ]:


#ERROR MERTICS
from sklearn import metrics


# In[ ]:


print('MSE:', metrics.mean_squared_error(test.iloc[:,10], prediction_dt))
print('RMSE:', np.sqrt(metrics.mean_squared_error(test.iloc[:,10], prediction_dt)))


# In[ ]:


#RANDOM FOREST REGRESSION
from sklearn.ensemble import RandomForestRegressor


# In[ ]:


rf = RandomForestRegressor(n_estimators = 1000, random_state = 42).fit(train.iloc[:,0:10],train.iloc[:,10])


# In[ ]:


prediction_rf = rf.predict(test.iloc[:,0:10])


# In[ ]:


#ERROR METRICS
print('MSE:', metrics.mean_squared_error(test.iloc[:,10], prediction_rf))
print('RMSE:', np.sqrt(metrics.mean_squared_error(test.iloc[:,10], prediction_rf)))


# In[ ]:


#LINEAR REGRESSION
import statsmodels.api as sm


# In[ ]:


employee['Reason for absence'] = employee['Reason for absence'].astype('float')
employee['Month of absence'] = employee['Month of absence'].astype('float')
employee['Day of the week'] = employee['Day of the week'].astype('float')
employee['Social drinker'] = employee['Social drinker'].astype('float')
employee['Social smoker'] = employee['Social smoker'].astype('float')


# In[ ]:


train, test = train_test_split(employee, test_size = 0.2)


# In[ ]:


model = sm.OLS(train.iloc[:,10],train.iloc[:,0:10]).fit()


# In[ ]:


model.summary()


# In[ ]:


prediction_lr = model.predict(test.iloc[:,0:10])


# In[ ]:


#ERROR METRICS
print('MSE:', metrics.mean_squared_error(test.iloc[:,10], prediction_lr))
print('RMSE:', np.sqrt(metrics.mean_squared_error(test.iloc[:,10], prediction_lr)))


# # LOSS FOR THE COMPANY EACH MONTH

# In[ ]:


new = df[['Month of absence','Service time','Work load Average/day ','Absenteeism time in hours']]


# In[ ]:


new["Loss"]=(new['Work load Average/day ']*new['Absenteeism time in hours'])/new['Service time']


# In[ ]:


new.head()


# In[ ]:


new["Loss"] = np.round(new["Loss"]).astype('int64')


# In[ ]:


No_absent = new[new['Month of absence'] == 0]['Loss'].sum()
January = new[new['Month of absence'] == 1]['Loss'].sum()
February = new[new['Month of absence'] == 2]['Loss'].sum()
March = new[new['Month of absence'] == 3]['Loss'].sum()
April = new[new['Month of absence'] == 4]['Loss'].sum()
May = new[new['Month of absence'] == 5]['Loss'].sum()
June = new[new['Month of absence'] == 6]['Loss'].sum()
July = new[new['Month of absence'] == 7]['Loss'].sum()
August = new[new['Month of absence'] == 8]['Loss'].sum()
September = new[new['Month of absence'] == 9]['Loss'].sum()
October = new[new['Month of absence'] == 10]['Loss'].sum()
November = new[new['Month of absence'] == 11]['Loss'].sum()
December = new[new['Month of absence'] == 12]['Loss'].sum()


# In[ ]:


data = {'No Absent': No_absent, 'Janaury': January,'Febraury': February,'March': March,
       'April': April, 'May': May,'June': June,'July': July,
       'August': August,'September': September,'October': October,'November': November,
       'December': December}


# In[ ]:


WorkLoss = pd.DataFrame.from_dict(data, orient='index')


# In[ ]:


WorkLoss.rename(index=str, columns={0: "Work Load Loss/Month"})

