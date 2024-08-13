#!/usr/bin/env python
# coding: utf-8

# ## Pizza Sales Analysis

# In[1]:


# Required Libraries:
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
pd.set_option('display.max_columns', None)
import datetime as date
from dateutil import parser


# In[2]:


# Read Data:
df = pd.read_csv("pizza_sales.csv")


# In[3]:


df.head()


# Data Cleaning and Exploratory Data Analysis(EDA):

# In[4]:


df.shape


# In[5]:


print("Numb of Rows : 48620, Columns: 12 ")


# In[6]:


df.columns


# In[7]:


df.info()


# In[9]:


# Checking for duplicates:


# In[8]:


sum(df.duplicated())


# In[ ]:


# Checking for NULL Values:


# In[9]:


df.isna().sum()


# In[12]:


# General statistics info:


# In[10]:


df.describe()


# In[11]:


df.nunique()


# In[19]:


## Determining KPI's Required:


# In[12]:


Total_Revenue = df['total_price'].sum()
print(f"The Total Revenue is: ${Total_Revenue}")


# In[26]:


# Average Order Value:


# In[13]:


Distinct_Order_Count = df['order_id'].nunique()


# In[14]:


print(Distinct_Order_Count)


# In[15]:


Avg_Order_Value = df['total_price'].sum() / df['order_id'].nunique()


# In[16]:


round(Avg_Order_Value, 2)
print(f"${Avg_Order_Value: .2f}")


# In[17]:


print(f"The Average Order Value is: ${round(Avg_Order_Value, 2)}")


# Chart Requirements:

# In[18]:


df.columns= df.columns.str.strip()    # To remove any leading or trailing spaces.


# In[19]:


# Extracting the hourly component of the order_time:
def extract_hour (order_time):
    if pd.notnull (order_time):
        return pd.to_datetime(order_time).hour
    else:
        return None


# In[20]:


df['Hour'] = df['order_time'].apply(extract_hour)
df['Hour']


# In[21]:


## Determine unique pizza categories per Order_id:
uniq_catPr_id = df.groupby('order_id')['pizza_category'].unique().reset_index()

# Count of unique categories:
uniq_catPr_id['category_count'] = uniq_catPr_id['pizza_category'].apply(len)   
uniq_catPr_id


# In[22]:


# Grouped order_id per hour:

grped_orderId= df.groupby('Hour')[['order_id']].count().sort_values(by='order_id', ascending=False).reset_index()
grped_orderId


# In[23]:


sns.barplot (x = 'Hour', y = 'order_id', data=grped_orderId)
plt.xlabel('Hour')
plt.ylabel('Order_id Count')
plt.title('Barplot of Order ID Count Vs Hour of order')
plt.show()


# In[24]:


# From the grouping and the plot above, the company receives its hieghest orders at 12th hour for
# each working day.


# In[24]:


# Grouped total price per hour:

grped_totalPrice= df.groupby('Hour')[['total_price']].sum().sort_values(by='total_price',ascending=False).reset_index()
grped_totalPrice


# In[25]:


sns.barplot(x='Hour', y= 'total_price', data=grped_totalPrice, palette='viridis')
plt.title('Plot of Total Price Vs Hour')
plt.show()


# In[27]:


# For each working day, the company receives the hieghest revenue at the 12th hour which corresponds to when hieghest 
# orders are received.


# In[26]:


# Pizza cat. ordered at 9th Hour:

filtered_df1=df[df['Hour']==9]
uniq_pizza_cat= filtered_df1[['pizza_category','pizza_size']]
uniq_pizza_cat


# In[27]:


# Count of each cat. at 9th hour:
pizza_cnt_9H= filtered_df1.groupby('pizza_category').size()
pizza_cnt_9H


# In[ ]:





# In[28]:


# Pizza cat. ordered at 10th hour:

filtered_df2=df[df['Hour']==10]
uniq_pizza_cat= filtered_df2[['pizza_category','pizza_size']]
uniq_pizza_cat


# In[29]:


# Count of each category at 10th hour:
pizza_cnt_10H= filtered_df2.groupby('pizza_category').size()
pizza_cnt_10H


# In[30]:


# Pizza cat. ordered at 11th hour:

filtered_df3=df[df['Hour']==11]
uniq_pizza_cat= filtered_df3[['pizza_category','pizza_size']]
uniq_pizza_cat


# In[31]:


# Count of each cat at 11th hour:
pizza_cnt_11H= filtered_df3.groupby('pizza_category').size()
pizza_cnt_11H


# In[34]:


# The above groupings justify the fact that from the plots above, the 9th and 10th hours had the least orders and thus
# generated the least revenue for the company. 


# In[32]:


# Grouped category total price:
grped_cat = df.groupby('pizza_category')[['total_price']].sum().sort_values(by='total_price',ascending=False)
grped_cat


# In[36]:


# Classic pizza generated the hieghest revenue for the company.


# In[33]:


# Distribution of pizza category sales:
sns.histplot(x= 'pizza_category', data=df)
plt.title('Distribution of pizza category sales')
plt.show()


# In[34]:


# Plot of total price vs category based on size:
sns.barplot(x='pizza_category', y= 'total_price', hue='pizza_size', data=df)
plt.title('Plot of total price Vs pizza category')
plt.show()


# In[38]:


### The plot above reveals the following: 
## For classic pizza, XXL and XL sizes have a significantly higher total price than other sizes and the XXL size is only
# observed in the classic category where it has the highest total price.
## For chicken, supreme and veggie categories, large pizzas have the total highest price.
## The total price for medium and small sizes is relatively consistent across chicken, supreme and veggie categories.


# In[35]:


sns.lineplot(x='Hour', y='order_id', marker='o',  data= grped_orderId)
plt.title('Line plot of order ID vs Hour of order')
plt.xlabel('Hour')
plt.ylabel('Order ID')
plt.show()


# In[41]:


# The line plot equally confirms that the hieghest orders were received at the 12th hour of each working day.


# In[40]:


sns.heatmap (df.corr(numeric_only=True), annot=True, cmap='BuGn_r')
plt.show()


# In[43]:


# The heatmap shows some positive correlation between the variables:total_price, quantity,and unit_price.


# In[36]:


## Hourly trend for total orders ( Stacked bar plot for over a period of time):
# Pivot the DataFrame for a stacked bar chart:

pivot_df = df.pivot_table (index = 'Hour', columns = 'order_id', values = 'total_price', aggfunc='sum', fill_value = 0)
     # Grouping the data, fill_value=0 ensures a complete pivot table.


# In[37]:


# Stacked bar chart using seaborn:

sns.barplot(x=pivot_df.index, y=pivot_df.sum(axis=1), color='Blue')

plt.xlabel('Hour of the day')
plt.ylabel('Total Pizza Orders')
plt.title ('Hourly trend of total pizza orders')
plt.show()


# In[46]:


# The company made the most sales in a day at the 12th hour.


# In[38]:


## Weekly trend for pizza orders:

def parse_date(date):
    return parser.parse(date, dayfirst=True)   # For proper interpretation of mixed date formats

# Apply function to order_date:
df['order_date'] = df['order_date'].apply(parse_date)

df['Week'] = df['order_date'].dt.isocalendar().week
df['Week'].head()


# In[39]:


# Year data was collected:

df['Year']= df['order_date'].dt.year
df['Year'].unique()


# In[40]:


# Grouping Pizza id per year:
grped_pizza_id = df.groupby('pizza_id')[['Year']].value_counts()   # Confirming year data was collected.
grped_pizza_id   


# In[50]:


# The above grouping further cnfirms data was collected in the year 2015.


# In[41]:


# Grouped weekly total price:

grped_wklyTotPrice= df.groupby('Week')[['total_price']].sum().sort_values(by='total_price',ascending=False).reset_index()
grped_wklyTotPrice


# In[47]:


plt.figure(figsize=(12,6))
sns.barplot(x= 'Week', y= 'total_price', data= grped_wklyTotPrice)
plt.title('Plot of Total Price per Week of 2015')
plt.xticks(rotation=45)
plt.show()


# In[ ]:





# In[42]:


# Week day orders:
df['Week_day'] = df['order_date'].dt.day_name()
df['Week_day'].head()


# In[43]:


# Average total price per week_day of 2015:

avg_TotPr_perWkday= df.groupby('Week_day')[['total_price']].mean().sort_values(by='total_price',ascending=False).reset_index()
avg_TotPr_perWkday


# In[50]:


# Plot of average total sales per week day:

plt.figure(figsize=(10,6))
Week_day=['Sunday', 'Monday', 'Tuesday', 'Wednesday','Thursday','Friday', 'Saturday']
sns.barplot(x= 'Week_day', y= 'total_price',order=Week_day, data= avg_TotPr_perWkday)
plt.title('Plot of total price per week_day')
plt.xticks(rotation=90)
plt.show()


# In[56]:


# Averagely the company generated the hieghest revenue on Tuesdays across the year(2015).


# In[44]:


# Monthly Orders:

df['Month'] = df['order_date'].dt.month_name()
df['Month'].unique()


# In[45]:


df.head()


# In[46]:


# Grouped Monthly Total Price:

grped_mnthlySales= df.groupby('Month')[['total_price']].sum().sort_values(by='total_price', ascending=False).reset_index()
grped_mnthlySales              # .reset_index() converts groupby() to a DataFrame.


# In[48]:


plt.figure(figsize=(7,4))

Month_name=['Jan', 'Feb', 'Mar', 'Apr','May','Jun', 'Jul','Aug','Sep','Oct','Nov','Dec']


plt.bar(grped_mnthlySales['Month'], grped_mnthlySales['total_price'], color='blue')
plt.title('Plot of total price per month.')
plt.xlabel('Month')
plt.ylabel('Total Price')
plt.xticks(rotation=45)
plt.show()


# In[61]:


# Based on data, the company generated the hieghest revenue in the month of July-2015.


# In[55]:


# Grouped quantity of pizza sold per day in a week:

grped_quantity= df.groupby('Week_day')[['quantity']].sum().sort_values(by='quantity',ascending=False).reset_index()
grped_quantity


# In[56]:


# Line plot quantities ordered through the week:

Week_day=['Sunday', 'Monday', 'Tuesday', 'Wednesday','Thursday','Friday', 'Saturday']

df['Week_day'] = pd.Categorical(df['Week_day'], categories= Week_day, ordered=True) # Week day to category

sns.lineplot(x='Week_day', y='quantity',marker='o', data=grped_quantity )
plt.title('Plot of quantity of pizza sold per day')
plt.xticks(rotation=90)
plt.show()            # Line function does not take order=week_day


# In[64]:


# From the plot above, the company sold the hieghest quantity of pizza on Fridays.


# In[57]:


# A scattered plot between unit price and total price:

sns.scatterplot( x='unit_price', y='total_price', data=df)
plt.title('Scattered plot of Total Price Vs Unit Price')
plt.show()


# In[58]:


# Reg plot of total price vs quantity:

sns.regplot(x='quantity', y='total_price', data=df, line_kws={'color':'green'})
plt.title('Plot of total price vs quantity sold')
plt.show()


# In[59]:


# A boxplot of monthly total price:

sns.boxplot(x='Month', y='total_price', data= df)
plt.title('Box plot of Total Price vs Month')
plt.xticks(rotation=45)
plt.show()


# #**Predictive analysis** on company's future monthly sales.

# In[68]:


# Use Random Forest model to predict future monthly total price:


# In[60]:


# Required libraries:
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error,mean_squared_error, r2_score
import matplotlib.dates as mdates


# In[61]:


df.columns


# In[63]:


drop_cols=['pizza_id','order_id','pizza_name_id','order_date','order_time','pizza_ingredients',
           'pizza_name','Week_day','Week']


# In[64]:


# Drop columns:
df=df.drop(drop_cols, axis=1)


# In[65]:


# Pizza size is hierachichal: S, M, L, XL and XXL, thus ordinal, so should be better converted to numbers.

pizza_sizeMapping={'S':1, 'M': 2, 'L': 3, 'XL':4, 'XXL': 5}
df['pizza_size']= df['pizza_size'].map(pizza_sizeMapping)


# In[ ]:





# In[66]:


# Dummy encoding:
df= pd.get_dummies(df, columns=['Month','pizza_category'],drop_first=False)


# In[67]:


df.info()


# In[68]:


# Splitting data into X-variables and y-variable(target):
y= df['total_price']

X= df.drop('total_price', axis=1)


# In[69]:


# Splitting data into train and test sets:
X_train, X_test, y_train, y_test = train_test_split(X, y,test_size=0.3, random_state=42)


# In[70]:


# Instatiating the Random Forest Regressor model:
rf_m= RandomForestRegressor(n_estimators=100, random_state=42)

# Fit and train the model:
rf_m.fit(X_train,y_train)


# In[71]:


## Evaluating the model:
y_pred= rf_m.predict(X_test)

# Calculating the regression metrics:
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print(f'Mean Absolute Error: {mae}')
print(f'Mean Squared Error: {mse}')
print(f'R-Squared: {r2}')


# In[1]:


# The values of the metrics above indicate that the model is performing very well in predicting 
# future monthly total prices of pizza for the company. An R-Squared score of 0.999 (99.9%) means the model captured almost all  
# of the variability in the dataset.Hence, suggesting that the model fits the data very well and predictions are very close to actual values.


# In[72]:


# Assuming y_test and y_pred are already defined
plt.figure(figsize=(12, 6))
plt.plot(y_test.values, label='Actual Total Price')
plt.plot(y_pred, label='Predicted Total Price')

# Formatting the x-axis to show months
plt.gca().xaxis.set_major_locator(mdates.MonthLocator(interval=5))
plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%b'))

# Rotate the x-axis labels to prevent congestion
plt.gcf().autofmt_xdate(rotation=90)

# Increase the font size of x-axis labels for better readability
plt.xticks(fontsize=8)

# Set axis labels
plt.xlabel('Month')
plt.ylabel('Total Price (Currency Unit)')

plt.title('Actual Vs Predicted Monthly Total Prices')
plt.legend()
plt.tight_layout()  # Adjust subplots to fit into figure area.
plt.show()


# In[ ]:





# In[ ]:





# In[ ]:




