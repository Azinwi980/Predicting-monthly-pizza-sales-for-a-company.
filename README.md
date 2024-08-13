**Overview:** The goal of this project was to develop a plausible model that could be used to predict future monthly sales of pizza for a company in order to enhance
effective imput. The dataset was downloaded from YouTube. In order to minimize the impact of outliers, the Random Forest Regressor Model was used and it recorded a significant
performance.
**Data Understanding:** The 'pizza_sales' dataset consists of 48,620 rows and 12 columns. Some of the columns are: pizza_id, order_id, quantity, order_date, order_time,
total_price, pizza_size, pizza_category, pizza_name.
**Modeling and Evaluation:** To minimize sensitivity to outliers and step-up model performance, Random Forest Regressor Model was used. This model recorded an oustanding
performance with an R-Squared score of 99.9%. 
Below is a plot of the total_price per month of pizza sales:
![image](https://github.com/user-attachments/assets/31eeb693-e8d0-41c3-a9c3-f0a0fac75ac0)

**Conclusion:** The model captured almost all of the variability in the data. As such, it can be used by the company management to predict future sales.
