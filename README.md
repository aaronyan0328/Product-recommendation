# Product-recommendation
Product recommendation for bank customer.
Abstract
	The problem is to predict which products the Santander bank’s existing customers will use in the next month based on their past behavior and that of similar customers. The team performs data exploration, cleaning, feature extraction and utilizes various machine learning models, MLkNN, Random Forest, and XGBoost, to solve the problem. The Random Forest model outperforms achieving the highest F1 score of 0.949.
Introduction
Provided with a numerous amount of financial activities, customers are facing the challenge to choose the right financial product that best meets their needs. Currently most of the banks only offer the consulting service of product purchase to VIP customers or customers who take the initiative to request this service, while the service is provided by experienced product managers of the bank using their expertise in this field. 
This project is an application of the bank financial services. This project aims to leverage the machine learning technology to provide personalized product recommendations for all customers. In this way, all customers will have easy access to the product recommendation service, resulting in a minimized human labor force involvement in this process. The ultimate goal is building a prediction (recommendation) system to show products that customers have a higher possibility to use in the future, based on their features (age, income, service purchased, and etc.), and those of customers with similar attributes.
Baseline
We choose our baseline to be the KNN model and compare the performance between XGBoost, KNN and random forest under different parameters and the same features.
Background & Prior Work
Majority of existing research in recommender systems focus on domains including movies, news, commercial products, and etc., and relatively less common in the field of banking services. Gallego etal .(2015) developed a context-aware recommender system for banks, but its major focus was to recommend customers with restaurants and locations where the bank partnered with. More study can be found on recommending best stock based on stock and transaction data, including the study by Sayyed FR etal. (2013). Even though his study is in the finance industry, there exist significant differences between stock and banking products. 
The data is provided by Santander Bank, a wholly owned subsidiary of the Spanish Santander Group. Since the project aims to recommend which products their existing customers will use, the dataset includes demographic and financial information of customers. The main location where customers are from Spain, and therefore the address information contains spanish names. 
3.1 Data
According to the source file of the dataset, the original dataset includes 2.39G training data, with 48 features provided. The first 24 features are customers’ personal information, such as sex and age. The following 24 features indicate if the customer possesses those products, with each one feature corresponding to each product. 
For training purposes, we selected the first 7,000,000 rows, and then selected randomly 12,000 unique customers based on the provided customer id. We obtained a data frame with 92,695 rows. A full description of features is provided in the appendix a. Here is a sample of features:

Variable
Description
sexo
Customers’ Age 
renta
Gross income of the household
ind_nuevo
New customer Index. 1 if the customer registered in the last 6 months.
segmento
segmentation: 01 - VIP, 02 - Individuals 03 - college graduated

Data Preprocessing
4.1 Exploratory Data Analysis & Data Cleaning
For the general data cleaning procedure, there are four stages: deleting redundant features (i.e. province name and province id), deleting irrational data, dealing with values with similar values but in different formats (i.e. string “1.0” and integer 1), and filling in NA values. 
First, we select the “age” column as the criteria to identify irrational customers. The histogram of age is shown in appendix b. There are customers who are over 100 years old and who are only 2 years old, and these entries may bring significant noise to the recommendation system. We first try to remove the outliers by calculating the IQR, but the data still contains customers with age around 0, which is not realistic. Based on common knowledge of customers of banks, we decide to limit the age range to 15-80. As from the age distribution graph, this range covers over 95% of the samples selected. 
After deletion, we discover that many columns, which previously contained N/A values, no longer contain N/A values. We believe that this operation is vital as it removes a system of irrational data. Around 4,000 rows are deleted. 
Second, there are data format issues in some columns. For example, for the column indrel_1mes which represents the type of customers (1: First/Primary customer, 2: co-owner, P: Potential, 3: former primary, 4: former co-owner), the data contains values of 1.0, '1.0', '1', '3', '3.0', nan, 'P', 2.0, 3.0, '2.0'. To solve this problem, we create a dictionary to ensure all similar values are represented in the same format. 
Third, we try to fill in NA values with three methods. First, for household income, we fill the N/A values with the median of the province that the customer is in. There exists data where the entire province income is missing, and we fill them in with the median of the income of all provinces. Second, for categorical values which we can’t predict, we fill in them with suitable values that represent “UNKNOWN”. (i.e., province code ranges from 1 to 50, but for NA we fill in -1000). Third, if the column contains significantly few NA values, we simply delete the entries as it would not influence the scale of data. 
Because the focus is to recommend products, we take a look at the Product Recommendations Distribution in all training data, shown in appendix c. The first two products, ‘ind_ahor_fin_ult1’ and ‘ind_aval_fin_ult1’, are barely recommended. The third product, ‘ind_cco_fin_ult1’, is dominant in the market, while the rest products hold mostly equivalent shares. 
4.2 Feature Processing and Extraction
To reduce the computational expense, data features are being processed and selected without redundancy. 
After converting all categorical features into numeric ones, we examine the descriptive statistics for features, drop ‘tipodom’ because its standard deviation=0, meaning no variation at all. We ended up with the following descriptive statistics, shown in appendix d.
We examine the correlation among all features, shown in appendix e. tiprel_1mes and ind_actividad_cliente are highly negatively correlated, so drop ind_actividad_cliente (less informative); ncodpers and canal_entrada are highly positively correlated, so drop ncodpers (less informative); age and [segmento, canal_entrada] are highly positively correlated, so drop segmento and canal_entrada; cod_prov and indresi are highly positively correlated, so drop indresi (less informative). The selected features are listed in appendix f.
Machine Learning Models 
	To solve this multi-labeling problem, taking the preprocessed data as the input, the following machine learning models learn the customers behavior and predict which products they’re likely to purchase, and hence provide the product recommendations.
6.1 MLkNN model
	The MLkNN model builds uses k-NearestNeighbors to find nearest examples to a test class. For each unseen instance, based on the statistical information gained from the label sets of these neighboring instances, the label of the unseen instance is determined. 
80% of the data is randomly selected as training data with 20% being the testing data. With the training data fed into the modeling, the randomized grid search, among range of 2 to 20 with 5-fold, is used to tune the number of kneighbors. That gave the result: the best number of neighbors to choose and its cross validation score: {'k': 2} 0.8330859808856081. The detailed process of the grid search is shown in the appendix g.
6.2 Random Forest model
 We are using the Random Forest model to collect various decision trees to arrive at any solution when making suggestions to customers. We use LabelEncoder to convert all categorical features to numbers and we use the split data in MLkNN model to be our train set and test set. 
We then use a random grid to search for best hyperparameters. The reason using random grid is normal grid search takes too much time. We use 3 fold cross validation when searching over 100 different combinations. 
6.3 XGboost model
	XGBoost is a Gradient Boosted Trees algorithm that is based on function approximation by optimizing specific loss functions as well as applying several regularization techniques. The objective function of XGBoost is shown as below, where we minimize the loss and regularization at time t.

Figure 2:XGBoost Objective
	The regularization part of the objective helps reduce overfitting and  it’s fast due to parallel processing. 
Evaluation and Results
	To further tune and improve the model performance, the error analysis is done to analyze how models work across different classes of targets. With that improvement done, we evaluate the models with f1 score, in which a higher f1 score is preferred. To assure the comparison justified, the training set is kept consistent across different models by setting the random_state = 22.
7.1 MLkNN model
The error analysis is shown in appendix h. The ind_aval_fin_ult1product, which is barely being purchased in the training data, is not learnt, resulting in 0 scores. All other products, especially popular products such as ‘‘ind_cco_fin_ult1’, are well learnt and achieve a reasonably good f1 score. The micro-average f1 score is 0.93.
7.2 Random Forest model
We test for the all remaining features to achieve an accuracy of 0.878 and F1 score of 0.941. The most important features turn out to be. The importance of the feature is shown in the appendix i. The four most important features are “ind_empleado”, “pais_residencia”, “sexo” and “age”. The random grid shows that the best parameters are min samples split is 2, min sample leaf is 2, max features set to None and bootstrap set to false. After that, we achieve an accuracy of 0.893 and F1 score of 0.949.
7.3 XGboost model
	Following the same features in random forest, we achieve an accuracy of 0.64 and F1 score of 0.646. Since the default setting is not a multiclass test so we transfer the target to a range from 0 to number of classes. Since we don’t achieve a good result compared to previous experiments. We stop the experiment.

Discussion and Prior Work
As mentioned in the Background Section, we see the majority of existing research in recommender systems focus on domains including movies, news, commercial products, and etc., and relatively less common in the field of banking services. Financial recommendation engines mainly focus on insurance, stock and partnered corporations. This model can help banks predict and recommend services to customers. 
Both our results show that the features “ind_empleado”, “pais_residencia”, “sexo” and “age” accounts for big impact when deciding which product the company should recommend to the customers. It make sense as sex identity and age can reflect people perference in choosing the product. People who are young might be risk tolent and senior people can be risk averse. Since the company serves customers all over the world and there are different products in different countries, thus “pais_residencia”, which represents the customer’s country of residence should have a huge impact when determining the recommendation as there is a certain pattern we need to follow in different countries. The “ind_empleado” feature, which is the employee index, has the information of whether the employee is active, unemployed or passive. That reflects the lifestyle quality of customers which should be an important factor as well. Our prior work expects that “indrel”, which is the customer type to indicate if he or she is a primary customer, and “ind_nuevo”, which is indicating whether a customer new, should have a big impact, but it turns out that it’s not. We’ve changed our strategy when determining the relationship between customers and we think more relevant data is needed, for example the customers review on products and he or she’s recognition on other’s review.
Conclusion
KNN and decision trees algorithms both achieve good results in making recommendations for customers as the logic behind is pretty close. Based on the data we have, we should either cluster customers to similar groups and the pattern of products within that group, or we should go nodes by nodes based on customer information to decide what product we should recommend. However, more data should be gathered by the company to identify the relationship between customers rather than just personal information like customer type, so that we can make better suggestions on the product.

