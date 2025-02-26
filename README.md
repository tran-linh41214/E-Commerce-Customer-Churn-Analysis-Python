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

