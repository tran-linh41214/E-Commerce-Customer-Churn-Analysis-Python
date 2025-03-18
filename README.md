# 📊 Project Title: E-Commerce Customer Churn Analysis   
Author: Linh Tran  
Tools Used: Python  

---

## 📑 Table of Contents  
1. [📌 Background & Overview](#-background--overview)  
2. [📂 Dataset Description & Data Structure](#-dataset-description--data-structure)
3. [⚒️ Main Process](#-main-process)
4. [🔎 Final Conclusion & Recommendations](#-final-conclusion--recommendations)

---

## 📌 Background & Overview  

### Objective:
### 📖 What is this project about?

This project utilizes **Python and machine learning** to analyze e-commerce customer data and:  

✔️ **Predict potential customer churn** and identify high-risk segments.  
✔️ **Optimize retention strategies** by detecting early churn indicators.  
✔️ **Compare different missing value handling methods** to assess their impact on model performance.  
✔️ **Enhance customer loyalty** through personalized re-engagement initiatives.  


### 👤 Who is this project for?  

✔️ **Marketing teams** looking to improve customer retention.  
✔️ **E-commerce business owners & decision-makers** aiming to reduce churn and maximize customer lifetime value.  
✔️ **Data analysts & data scientists** interested in predictive modeling for customer segmentation.  


---

## 📂 Dataset Description & Data Structure  

### 📌 Data Source   
- Size: 20 columns, 5630 rows  
- Format: .csv  

### 📊 Data Structure & Relationships  

This project used 1 table:
  <details>
  <summary>Churn_prediction table</summary>

| Column Name                        | Data Type | Description |
|------------------------------------|----------|-------------|
| CustomerID                         | int64    | Unique identifier for each customer |
| Churn                              | int64    | Indicates if the customer has churned (1) or not (0) |
| Tenure                             | int64    | Number of months the customer has been with the company |
| PreferredLoginDevice               | object   | Device most frequently used by the customer to log in |
| CityTier                           | int64    | Tier classification of the customer's city |
| WarehouseToHome                    | float64  | Distance from the warehouse to the customer's home (in km) |
| PreferredPaymentMode               | object   | Customer's most frequently used payment method |
| Gender                             | object   | Customer's gender |
| HourSpendOnApp                     | float64  | Average number of hours spent on the app per month |
| NumberOfDeviceRegistered           | int64    | Total number of devices registered by the customer |
| PreferredOrderCat                  | object   | Customer's most frequently ordered product category |
| SatisfactionScore                   | int64    | Customer's satisfaction rating (scale from 1 to 10) |
| MaritalStatus                      | object   | Customer's marital status |
| NumberOfAddress                    | int64    | Number of addresses registered by the customer |
| Complain                           | int64    | Indicates if the customer has made a complaint (1) or not (0) |
| OrderAmountHikeFromLastYear        | float64  | Percentage increase in order amount compared to last year |
| CouponUsed                         | float64  | Number of coupons used by the customer |
| OrderCount                         | float64  | Total number of orders placed by the customer |
| DaySinceLastOrder                  | float64  | Number of days since the last order was placed |
| CashbackAmount                     | int64    | Cashback amount received by the customer |

  </details>

---

## ⚒️ Main Process

1️⃣ Data Cleaning & Preprocessing  
<details>
  <summary> Import libraries</summary>

```python
# Import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn import linear_model
```

  </details>

```python
from google.colab import drive
drive.mount('/content/drive')
```

```python
# Load the dataset
df = pd.read_csv('/content/drive/My Drive/Colab Notebooks/churn_prediction.csv')
```

![image](https://github.com/user-attachments/assets/1ec5b779-84f1-4217-926d-7cf4a08193fe)

![image](https://github.com/user-attachments/assets/298bdd3b-d149-49eb-b253-6696238773e4)


2️⃣ Exploratory Data Analysis (EDA)  


3️⃣ SQL/ Python Analysis 

 

---

## 🔎 Final Conclusion & Recommendations  

👉🏻 Based on the insights and findings above, we would recommend the [stakeholder team] to consider the following:  

📌 Key Takeaways:  
✔️ Recommendation 1  
✔️ Recommendation 2  
✔️ Recommendation 3



# Machine-Learning
## **Question 1:**

1. Churned users usually are new users &rarr; Provide more promotion for new users, or increase the new users experience
2. Churned users usually receive less cashback than not churn &rarr; Increase the cashback ratio
3. Churned users complain more &rarr; deep dive what these churned users complain about, and provide the solution

As Feature Importance show, we can see these features can have high relation with target columns:
* Tenure
* Cashback amount
* Distance from warehouse to home
* Complain
* Days since Last order

&rarr; We will analyse and visualize these features for more insights.
#### **1.4 Analyse features from initial Random Forest model:**

* Tenure
* Cashback amount
* Distance from warehouse to home
* Complain
* Days since Last order
##### **1.4.1 Tenure**  New users are churned more than old users (tenure = 0 or 1)
![image](https://github.com/user-attachments/assets/62293f54-2df8-4195-8fca-63947b5382a2)

##### **1.4.2 Warehouse to home**  Not significantly related
![image](https://github.com/user-attachments/assets/47023f6e-4cd9-4882-8d76-c31d00ad6970)

For both churn & not churn:
* The median, pt25, mean, pt75 is quite the same --> The centralize of data is the same
* For not churn, data has some outliers --> This can be not significant enough to consider it as an insight for not churn

&rarr; There're no strong evidences show that there different between churn and not churn for warehousetohome --> We will exclude this features when apply model for not being bias.
##### **1.4.3 Days since last order:** churn users with complain = 1 have higher days since orders than churned users with complain = 0  
![image](https://github.com/user-attachments/assets/f7c936a1-9757-4a65-ae11-38e85e3d97fa)

From this chart, we see for churned users, they had orders recently (the day since last order less than not churned users) --> This quite strange, we should monitor more features for this insight (satisfaction_score, complain,..)
![image](https://github.com/user-attachments/assets/24707f49-28f7-46a2-a908-7c93d70763c8)

For churned users with complain = 1, they had daysincelastorder higher than churn users with compain = 0
##### **1.4.4 Cashback amount**  Churn users recevied cashback amount less than not churn users.
![image](https://github.com/user-attachments/assets/1cf77adf-4c4d-470d-8a86-53068b3fa4ab)

Churn users recevied cashback amount less than not churn users.
##### **1.4.5 Complain** The number of users complain on churn is higher than not churn
![image](https://github.com/user-attachments/assets/f156facf-cfc5-4e2a-a601-236f72a97fc5)

##### **1.4.6 Conclusion & Suggestion**
1. Churned users usually are new users &rarr; Provide more promotion for new users, or increase the new users experience
2. Churned users usually receive less cashback than not churn &rarr; Increase the cashback ratio
3. Churned users complain more &rarr; deep dive what these churned users complain about, and provide the solution
## **Question 2:**

* Use K-Means to clustering churn-users groups.
* Find the insight between the groups
### **1. Get the data prepared**

We will get all features of churned users for clustering
#Prepare data:
df_churned = df[df['churn']==1]
df_churned.drop(columns = ['customerid','churn'],inplace=True)
print(df_churned.shape)
df_churned.head(2)
### **2. Apply KMeans model**
#### 2.1. Choosing K:
![image](https://github.com/user-attachments/assets/b6283998-703d-4fae-abae-491d053380c2)

- When applying Elbow method, we see there're no clear elbow points.
- Our hypothesis is that the data is sporadic, which means there're no clearly common patterns between data, and we can not cluster them into groups.

## Our suggestions for next steps:

* We can collect more data of churned users: by collect real data or using our above supervised model to predict and use it as ground truth data for clustering model

* Business can offer the promotion for all churned users and collect results. These results can be used as features in the data for the next model.

