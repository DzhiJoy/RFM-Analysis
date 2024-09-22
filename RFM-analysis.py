#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd


# In[2]:


df = pd.read_csv('C:\\Users\\Джойчик\\Desktop\\sales_apteka.csv', encoding='windows-1251', sep=';')


# In[3]:


df.head()


# In[4]:


df['datetime'] = pd.to_datetime(df['datetime'])


# In[5]:


df.info() #посмотрим какие есть типы данных


# In[6]:


df = df[df['card'].str.startswith('2000')]


# In[7]:


df = df.sort_values(['card', 'datetime'])
df


# In[8]:


max(df['datetime']) #посмотрим дату самой давней покупки


# In[9]:


df2 = df.groupby('card').agg(
    purchase_sum = ('summ_with_disc', 'sum'),
    purchase_amount = ('summ_with_disc', 'count'),
    last_purchase = ('datetime', 'last'),
).reset_index()


# In[10]:


df[df['card'] == '2000200150091']


# In[11]:


df2


# In[12]:


df2['days_since_last_purchase'] = (max(df['datetime']) - df2['last_purchase']).dt.days


# In[13]:


df2


# In[14]:


import seaborn as sns


# In[15]:


sns.violinplot(df2['purchase_sum']); #построим график "скрипичный ключ" по суммам покупок


# In[16]:


import numpy as np


# In[17]:


quantiles = np.arange(0.1, 1.1, 0.1)


# In[18]:


quantiles


# In[19]:


quantiles = [round(el, 2) for el in np.arange(0.1, 1.1, 0.1)]
x_values = df2['purchase_sum'].quantile(quantiles)


# In[20]:


df2['purchase_sum'].quantile(quantiles) #смотрим перцентили по суммам покупок


# In[21]:


import matplotlib.pyplot as plt


# In[22]:


ax = sns.barplot(x=x_values, y=quantiles, orient='h')

for p in ax.patches:
    ax.annotate(f'{p.get_width():.2f}', (p.get_width(), p.get_y() + p.get_height() / 2),
                ha='left', va='center', xytext=(5, 0), textcoords='offset points')

plt.show() #визуализируем перцентили


# In[23]:


quantiles = [round(el, 2) for el in np.arange(0.1, 1.1, 0.1)]
x_values = df2['purchase_amount'].quantile(quantiles)

ax = sns.barplot(x=x_values, y=quantiles, orient='h')

for p in ax.patches:
    ax.annotate(f'{p.get_width():.2f}', (p.get_width(), p.get_y() + p.get_height() / 2),
                ha='left', va='center', xytext=(5, 0), textcoords='offset points')

plt.show()


# In[24]:


quantiles = [round(el, 2) for el in np.arange(0.1, 1.1, 0.1)]
x_values = df2['days_since_last_purchase'].quantile(quantiles)

ax = sns.barplot(x=x_values, y=quantiles, orient='h')

for p in ax.patches:
    ax.annotate(f'{p.get_width():.2f}', (p.get_width(), p.get_y() + p.get_height() / 2),
                ha='left', va='center', xytext=(5, 0), textcoords='offset points')

plt.show()


# In[25]:


quantiles = [0.33, 0.66] #зададим 33 и 66 - самые оптимальные проценты


# In[26]:


df2['days_since_last_purchase'].quantile(quantiles)


# In[27]:


df2


# Далее считаем собственно Recency, Frequency, Monetary

# In[28]:


#создаём функцию
def set_score(val, var, perc_33, perc_66):
  if val < perc_33:
    return 3 if var != 'R' else 1
  elif val < perc_66:
    return 2
  else:
    return 1 if var != 'R' else 3
#создаём переменные
recency_quantiles = df2['days_since_last_purchase'].quantile(quantiles)
df2['R'] = df2['days_since_last_purchase'].apply(set_score, args=('R', recency_quantiles.iloc[0], recency_quantiles.iloc[1]))

frequency_quantiles = df2['purchase_amount'].quantile(quantiles)
df2['F'] = df2['purchase_amount'].apply(set_score, args=('F', frequency_quantiles.iloc[0], frequency_quantiles.iloc[1]))

monetary_quantiles = df2['purchase_sum'].quantile(quantiles)
df2['M'] = df2['purchase_sum'].apply(set_score, args=('M', monetary_quantiles.iloc[0], monetary_quantiles.iloc[1]))


# In[29]:


df2


# In[30]:


df2['RFM'] = df2.apply(lambda row: f"{row['R']}{row['F']}{row['M']}", axis=1)


# In[31]:


df2


# In[32]:


import plotly.express as px


# In[33]:


df3 = df2.groupby('RFM')['RFM'].agg({'count'}).reset_index()
df3


# In[34]:


px.treemap(df3, path=['RFM'], values='count')

