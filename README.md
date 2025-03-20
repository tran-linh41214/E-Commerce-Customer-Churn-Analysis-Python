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

  <details>
  <summary>Figuring Out Null Values</summary>

    
| Column Name                  | Missing Values |
|------------------------------|---------------|
| CustomerID                   | 0             |
| Churn                        | 0             |
| Tenure                       | 264           |
| PreferredLoginDevice         | 0             |
| CityTier                     | 0             |
| WarehouseToHome              | 251           |
| PreferredPaymentMode         | 0             |
| Gender                       | 0             |
| HourSpendOnApp               | 255           |
| NumberOfDeviceRegistered     | 0             |
| PreferedOrderCat             | 0             |
| SatisfactionScore            | 0             |
| MaritalStatus                | 0             |
| NumberOfAddress              | 0             |
| Complain                     | 0             |
| OrderAmountHikeFromLastYear  | 265           |
| CouponUsed      |   256  |
| OrderCount  |   258  |
|  DaySinceLastOrder  |  307  |
|  CashbackAmount  | 0 |

  </details>

- Quick Statistical Overview

  Numeric Data Type  
![image](https://github.com/user-attachments/assets/161acb08-3b4c-4927-a4b0-6f784ee4015b)  
![image](https://github.com/user-attachments/assets/f5ae7069-6789-488f-83cb-2c465ffcfea5)


Object Data Type

![image](https://github.com/user-attachments/assets/146558d6-75c7-44f2-9ff6-b859710fe530)  

- The dataset contains 20 columns, 7 of which has null values, listed as the following: Tenure, WarehouseToHome, HourSpendOnApp, OrderAmountHikeFromlastYear, CouponUsed, OrderCount, DaySinceLastOrder  

- The dataset has no duplicate values on row level.  

- All columns have the right data type (There's no need for data type conversion)  

- Quick statistical summary indicates that there could be potential outliers in: Tenure, WarehouseToHome, SatisfactionScore, NumberOfAddress, CouponUsed, DaySinceLastOrder


<details>
<summary>Handling/Figuring out outliers</summary>

  <details>
  <summary>Analyze outliers for Tenure, WarehouseToHome, SatisfactionScore, NumberOfAddress, CouponUsed, DaySinceLastOrder</summary>  

-  **Column Tenure**: Tenure column has extrem unlogical value below 0 and outliers starting from a value of 50.0+
  ➡️ Excluding values above 50.0  
  
![image](https://github.com/user-attachments/assets/720f718b-d14b-412a-87c3-c3be0e1f9bc1)

- **Column WarehouseToHome**:
  ![image](https://github.com/user-attachments/assets/cde77c8b-2028-4afa-b1f4-54d3ede15f7a)

Column WarehouseToHome has outliers on values 126.0 and 127.0
➡️ Keep values less than 126.

![image](https://github.com/user-attachments/assets/d51f19fa-90cd-40fa-b467-7dc26782ec7b)  
- **Column SatisfactionScore**:
  ![image](https://github.com/user-attachments/assets/4dd2e917-029e-4336-bcad-dabd41a20f9e)  

➡️ There seems to be no extreme outliers, we will carry on with other columns  
- **Column NumberOfAddress:**
  ![image](https://github.com/user-attachments/assets/e94177df-f742-4de2-bd7d-c9c5be9848cd)

➡️ Keep the values < 20  
![image](https://github.com/user-attachments/assets/a8eab6e4-ff43-4349-bd6b-818a23a26a10)


- **Column CouponUsed:**
  ![image](https://github.com/user-attachments/assets/cf50cbc0-d584-472b-aa12-a22b927fa029)

  ➡️ Outliers in column [CouponUsed] don't represent wrong measurement and they could provide useful information, Hence, I've decided to leave them

- **Column DaySinceLastOrder**  

![image](https://github.com/user-attachments/assets/be828b7d-95f1-4693-9fb3-4b403cd31042)  

➡️ There are outliers yet since the column represents Recency of order then the outliers provide a relevant information and hence it is not problematic to leave them.
  
  </details>

✅ **Outliers Conclusion:**  
- As Expected, There were outliers in columns: Tenure, WarehouseToHome, NumberOfAddress, CouponUsed, DaySinceLastOrder  
- Now, Let's make sure that the other columns don't contain outliers. Churn, CityTier, HoursSpendOnApp, NumberOfDeviceRegistered, Complain, OrderAmountHikeFromlastYear, CashbackAmount

  <details>
  <summary>Analyze outliers for other columns</summary>
- **Column Churn:**  
  ![image](https://github.com/user-attachments/assets/d41fb9cf-ad2b-40a9-af7c-e091ad7c736a)

- **Column CityTier:**  

![image](https://github.com/user-attachments/assets/7f69c4ab-5354-4fdb-8fcc-0ffe907672d3)  

- **Column HoursSpendOnApp**  

![image](https://github.com/user-attachments/assets/58037f33-5ef0-4ebe-a5b5-97f0d808bd0e)  

![image](https://github.com/user-attachments/assets/62d0590a-0b2e-4cd3-a270-767a5b5a5a79)  

=> keep the values more than 0.0 and less than 5.0

![image](https://github.com/user-attachments/assets/4185a491-d030-4196-9803-b05f43662b3b)  

- **Column NumberOfDeviceRegistered**  
![image](https://github.com/user-attachments/assets/24aeab38-228d-40c2-82b2-4df961af21c1)  

- **Column Complain:**

  ![image](https://github.com/user-attachments/assets/261e3445-399d-4ba5-bd6e-1dac4c70f175)
- **Column OrderAmountHikeFromlastYear**  
![image](https://github.com/user-attachments/assets/eb41e577-c656-4315-89ea-b7b5f52a329a)

![image](https://github.com/user-attachments/assets/364e0fbb-d8f4-4b24-832f-a0ce58868e52)  
- **Column CashbackAmount**  
![image](https://github.com/user-attachments/assets/005fbf6c-69ad-4042-86db-a4d8ac96b30e)  
=> Since this column represents an average cash back amount on monthly basis, it is normal to be a fluctuated amount and hence the outliers could be left as it is.  

  </details>
  </details>




<details>
<summary>Handling/Figuring out null values and Handling/Figuring out wrong values</summary>

![image](https://github.com/user-attachments/assets/8ed2c20b-52aa-4862-88f7-66beea751ce8)

Figuring out unique values of each column => Nothing unusual.

Replace the abbriviate in PreferredPaymentMode column:
```python
#PreferredPaymentMode
OnlineRetail['PreferredPaymentMode'].unique()
```

```python
#Handling 'COD' and 'CC' values
#Replacing COD with Cash On Delivery
OnlineRetail['PreferredPaymentMode'].replace('COD','Cash on Delivery', inplace=True) 

#Replacing CC with Credit Card
OnlineRetail['PreferredPaymentMode'].replace('CC','Credit Card', inplace=True) 

#Replacing nan with Other
OnlineRetail['PreferredPaymentMode'].fillna(value='Other', inplace=True)
```

</details>


3️⃣ Feature Engineering  
Feature Engineering: One Hot Encoding  
```python
#Turning Nominal Object columns into seperate columns
OneHot_Columns = ['PreferredLoginDevice','PreferredPaymentMode','Gender','PreferedOrderCat','MaritalStatus']
OnlineRetail = pd.get_dummies(OnlineRetail, columns= OneHot_Columns)
```

<details>
<summary>Handling nulls</summary>

  <details>
  <summary>Method 1: Dropping nulls</summary>
    
 - Take a copy and drop nulls
 - **Modeling: Decision Tree**  
![image](https://github.com/user-attachments/assets/5d64aac2-520c-4b50-bec0-c014e3907c27)  
  

- **Modeling: Random Forest**

  ![image](https://github.com/user-attachments/assets/9d427935-447d-446b-b017-5053c3fe6702)

✅ **Method 1 Conclusion:**  
Handling nulls through dropping them results in a not bad model, Yet, Let's try how the model would perform if we filled in the nulls.

  </details>

   <details>
  <summary>Method 2: Filling nulls with Multivariate Imputation By Chained Equations algorithm</summary>

```python
#Taking a copy of columns with null values
df_missing_columns = OnlineRetail.filter(
    ['CustomerID','OrderAmountHikeFromlastYear','CouponUsed', 'OrderCount', 'DaySinceLastOrder'], axis=1).copy()

#Defining MICE imputer and filling in missing values
missing_imputer = IterativeImputer(estimator=linear_model.BayesianRidge(), n_nearest_features=None, 
                                   imputation_order='ascending')
df_missing_imputed = pd.DataFrame(missing_imputer.fit_transform(df_missing_columns), 
                                  columns=df_missing_columns.columns)
Cleaned_OnlineRetail = OnlineRetail.copy()
Cleaned_OnlineRetail.drop(['OrderAmountHikeFromlastYear','CouponUsed','OrderCount', 'DaySinceLastOrder'], axis = 1 , inplace = True)
OnlineRetail_MICE = Cleaned_OnlineRetail.set_index('CustomerID').join(df_missing_imputed.set_index('CustomerID'))
OnlineRetail_MICE.reset_index(inplace=True)
```

- **Modeling: Decision Tree**  
![image](https://github.com/user-attachments/assets/aa32cecf-cf2c-430c-869f-181a80f3d395)

- **Modeling: Random Forest**
![image](https://github.com/user-attachments/assets/55405ffa-51c5-478f-94d1-bb9e5d8a1ef8)

✅ **Method 2 Conclusion:**   
Decision Tree returned an OK result yet Method 1 was better, Yet, In Method 2 Random Forest, The model was off or can be considered inaccurate.  

  </details>

  <details>
  <summary>Method 3: Filling nulls with interpolate in Pandas that could predict the nulls using the correlation with other columns</summary>
- The nulls are handled first through distributions  

  <details>
  <summary>Handling nulls in the following columns: OrderAmountHikeLastYear, CouponUsed, OrderCount and DaySinceLastOrder</summary>

- **Column: OrderAmountHikeFromlastYear**
  ![image](https://github.com/user-attachments/assets/7c153b77-3534-416f-9f45-8df836e3a00b)

```python
OnlineRetail_Interpolate['OrderAmountHikeFromlastYear'].interpolate(method = 'linear', inplace=True)
```
*Making sure the column still has the same distribution*  
![image](https://github.com/user-attachments/assets/46db18a4-8001-4df4-b4ff-5a20fac1b5ec)  

```python
#Validating the changes in nulls
OnlineRetail_Interpolate['OrderAmountHikeFromlastYear'].isnull().sum()
```
=> 0 nulls
- **Column: CouponUsed**
![image](https://github.com/user-attachments/assets/b7e815d6-d7c8-4f00-ad77-234457eb2be2)
```python
OnlineRetail_Interpolate['CouponUsed'].interpolate(method = 'linear', inplace=True)
```
*Making sure the column still has the same distribution*    

```python
#Validating the changes in nulls
OnlineRetail_Interpolate['CouponUsed'].isnull().sum()
```
=> 0 nulls

- **Column: OrderCount**
  ![image](https://github.com/user-attachments/assets/906ed333-7965-4cf4-afdb-8cc311ffba59)
 
```python
OnlineRetail_Interpolate['OrderCount'].interpolate(method = 'linear', inplace=True)
```
*Making sure the column still has the same distribution*   
```python
#Validating the changes in nulls
OnlineRetail_Interpolate['OrderCount'].isnull().sum()
```
=> 0 nulls

- **Column: DaySinceLastOrder**
  ![image](https://github.com/user-attachments/assets/5b51974d-45fd-4fe3-b797-407715365c22)

```python
OnlineRetail_Interpolate['DaySinceLastOrder'].interpolate(method = 'linear', inplace=True)
```
*Making sure the column still has the same distribution*   
```python
#Validating the changes in nulls
OnlineRetail_Interpolate['DaySinceLastOrder'].isnull().sum()
```
   </details>
   
- **Modeling: Decision Tree**  
![image](https://github.com/user-attachments/assets/30c8ec70-8aff-4d7e-8f7f-095177973adc)
- **Modeling: Random Forest**
  ![image](https://github.com/user-attachments/assets/afaa9c55-214b-4c0a-9ef2-78f800a650ed)
    </details>

</details>



---


**Modeling Conclusion**
- We have used Classification Report as a matric of model evaluation to use **Recall** in particular because we want to evaluate the maximum number of churned customers out of the total churned customers.  

- Out of the 2 models **(Decision Tree, Random Forest)** in the 3 different null handling methods, We figured out that Random Forest has the best **Recall Percentage**, Hence, We have decided to use **Random Forest** as our final model.  

- Finally, It terms of **Recall Score** of the 3 null handling methods, There's a close score between **Dropping Nulls** and **Interpolate** and since **Interpolate** is better in terms of keeping rows on a larger scale dataset, We figured out that it is the best way to use in this case.  

<details>
<summary>Figuring Out Duplicate Values</summary>



</details>

4️⃣ Data Analysis  
 ❓ Q1) ***Analyze the number of days since the last order by the customer to create targeted marketing campaigns and offer personalized discounts***  

![image](https://github.com/user-attachments/assets/832f8c2c-d9d3-4441-b949-da7756ec2a95)

![image](https://github.com/user-attachments/assets/c47cac33-9053-49c2-b5b3-eb063b080b1f)


> ***The Majority of the customers fall into a value of 0 to 10 days since their last order, Hence, Targeting this segment would encourage them to reorder and reduce the 10 days cycle and increase sales.***

❗ Churn customers  
![image](https://github.com/user-attachments/assets/222ea0ec-685a-411b-808b-a70476edbdb5)


  >***Approximately 50% of customers churn after 3 days of their last order while 75% of customers churn after 6 days of their last order, Moreover, on average, it takes customers from 3 to 4 days to churn after their last order***  

❓ **Q2) Is there any difference in the buying behavior of male and female customers?**
![image](https://github.com/user-attachments/assets/d8bc74c6-26f1-4367-a15b-0f6faf1c7be0)

![image](https://github.com/user-attachments/assets/dbbe75e7-ec5b-4979-87f3-deff8d5373a5)  

> ***Approximately 16,21% of male customers churned and yet only 13,07% on female customers were churned***

*Further Analysis on the Gender behaviour*  
👨 Male  
![image](https://github.com/user-attachments/assets/b9ffb729-eec8-4273-83cc-44d62acc09fc)  

![image](https://github.com/user-attachments/assets/c1bf9659-97d5-4af7-b2a4-d38fef483767)  
![image](https://github.com/user-attachments/assets/d5680aa6-81f6-4cbc-9dcb-4d2bb626467e)

![image](https://github.com/user-attachments/assets/5afa0b00-02b0-411f-acd4-d910366cf920)  

average_order_per_male = 3.150848631797714  

![image](https://github.com/user-attachments/assets/c158850a-5457-4d95-a01e-b387b73f6949)  


👩 Female  

![image](https://github.com/user-attachments/assets/c1024ac3-e9c0-4fc8-a8e0-0bb1fa20df42)
![image](https://github.com/user-attachments/assets/8c803169-9745-4a61-99e4-641e24ab8233)  
![image](https://github.com/user-attachments/assets/d2083e57-a87a-4b5b-b299-7a835fec8375)  
![image](https://github.com/user-attachments/assets/0d4bfdd8-5117-43bc-b8e9-0feb5ca39211)  

average_order_per_female = 3.3217568947906027  
 
![image](https://github.com/user-attachments/assets/8b29253e-b8dd-44fe-8df7-f58eaf61163a)




✅ ***Females and Males spend almost the same number of hours on app***  

✅ ***Both genders have the same preferred payment mode***   

✅ ***Both gender do have the same categories of interest***  


✅ ***Although female customers are less than male customers but on average a female customer tends to have an average of 3.3397 order versus 3.1601 per male***  

✅ ***Males have the share of the lion when it comes to the cash-back amount***






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

