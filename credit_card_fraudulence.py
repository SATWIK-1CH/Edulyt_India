#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import seaborn as sn
import matplotlib.pyplot as plt


# In[128]:


data = pd.read_excel("credit_card/Credit_cards_App_data.xlsx")


# In[30]:


type(data)


# In[114]:


data


# In[72]:


Total_applications = len(data)
print(Total_applications)


# In[88]:


Approved_applications = (data.tdecision == 'Approve').sum()
print(Approved_applications)


# In[74]:


def sum_Booked(df):
    return(df.Booking == 'Y').sum()


# In[75]:


def divide(x,y):
    return x/y


# In[76]:


Booked_applications = sum_Booked(data)
print(Booked_applications)


# In[78]:


Approval_rate = divide(Approved_applications,Total_applications)
print(Approval_rate)


# In[80]:


Booking_rate = divide(Booked_applications,Approved_applications)
print(Booking_rate)


# In[87]:


cumilative_data = data[data.Booking == 'Y']
cumilative_data.head()


# In[111]:


cumilative_data['Cum_value'] = cumilative_data['Booking_Amt'].cumsum()


# In[124]:


cumilative_data


# In[98]:


for i in range(len(cumilative_data['Booking_Amt'])):
    cumilative_data['Cum_value'][i+1] = cumilative_data['Booking_Amt'][i+1] +  cumilative_data['Cum_value'][i]


# In[64]:


data.describe()


# In[65]:


sn.scatterplot(x="Salary",y="Booking_Amt",data = data)


# In[66]:


data.plot(kind = 'bar',x="Salary",y="Booking_Amt")


# In[67]:


def lowerconvert(w):
    return w.lower()


# In[68]:


ax = plt.gca()
data.plot(kind = 'line',x = 'Booking_Amt',y='Salary',ax = ax)
data.plot(kind = 'line',x = 'Booking_Amt',y='Salary',ax = ax,color = 'red')
plt.show()


# In[69]:


data.groupby(['ExDebt','Booking']).size().unstack().plot(kind='bar',stacked=True)
plt.show()


# In[ ]:


data['debt_sorted'] = data['ExDebt'].sort_values()


# In[ ]:


data['salary_sorted'] = data['Salary'].sort_values()


# In[ ]:


data.groupby(['gender','Booking']).size().unstack().plot(kind='bar',stacked=True)
plt.show()


# In[ ]:


data['gender']


# In[ ]:


data['debt_sorted']


# In[ ]:


data['ExDebt'].max()


# In[43]:


def checkduplicates(listofelements):
    if len(listofelements) == len(set(listofelements)):
        return True
    else:
        return False


# In[44]:


checkduplicates('gender')


# In[ ]:





# In[20]:


checkduplicates(data['Salary'])


# In[21]:


f.assign(dummy = 1).groupby(
['dummy','state']
).size().groupby(level=0).apply(
    lambda x: 100 * x / x.sum()
).to_frame().unstack().plot(kind='bar',stacked=True,legend=False)

# or it'll show up as 'dummy'
plt.xlabel('state')

# disable ticks in the x axis
plt.xticks([])

# fix the legend or it'll include the dummy variable
current_handles, _ = plt.gca().get_legend_handles_labels()
reversed_handles = reversed(current_handles)
correct_labels = reversed(df['state'].unique())

plt.legend(reversed_handles,correct_labels)

plt.gca().yaxis.set_major_formatter(mtick.PercentFormatter())
plt.show()


# In[ ]:





# In[ ]:




