# Credit_Risk_Analysis

## Overview of Project 
In this project, I am using different machine learning techniques to predict the loan status. 
## Results
### 1. Predicting credit risk with resampling and logistic regression 
#### Cleaning the data:
-	The data set LoanStats_2019Q1.csv was stripped of na
-	Issued loans were dropped
-	Current was substituted with low risk
-	'Late (31-120 days)', 'Late (16-30 days)', 'Default', 'In Grace Period' were substituted with high risk
-	Interest rates were changed to numerical. 
Outcome:

![res1](/imgs/fig1.png?raw=true)

-	I used get_dummies on all non-float columns and converted the resulting columns to float.
-	I dropped the loan_status_low_risk column since it is anticorrelated with the loan_status_high_risk column (See the image below) and renamed loan_status_high_risk to loan_status. Therefore loan_status = 0 for low risks and 1 for high risks 

![res2](/imgs/fig2.png?raw=true)

-	Split the data into features (X) and targets (y=loan)
-	I split the X and y into training and testing sets
-	I scaled the X_train and X_test using standard scaling and did a logistic regression which resulted in balance accuracy score of 0.60381 and the following imbalanced classification report:
![res3](/imgs/fig3.png?raw=true)

-	A closer look at the set indicates that there are 6847 low risk samples compared to 347 high risk samples. That’s why the model successfully predicts the low-risk cases while it is only 33% successful in picking the high risk ones. To remedy this issue, I implemented oversampling, undersampling and combination approaches.
#### Oversampling
##### Naïve Random oversampling
The code was implemented according to the following.

![res4](/imgs/fig4.png?raw=true)

With the following outcome

![res5](/imgs/fig5.png?raw=true)

As is shown in the imbalanced classification report the model still not very sensitive to high-risk cases (f1 = 0.06)
##### SMOTE oversampling
It was implemented according to the following

![res6](/imgs/fig6.png?raw=true)

See the following image for balanced accuracy, confusion matrix and imbalanced classification report

![res7](/imgs/fig7.png?raw=true)

Despite the marginal improvement, the model still significantly lacks sensitivity to high risk cases.

#### Undersampling
I used ClusterCentroids resampler as following

![res8](/imgs/fig8.png?raw=true)

With the following balanced accuracy, confusion matrix and imbalanced classification report
![res9](/imgs/fig9.png?raw=true)

This model does not appear to be more sensitive to high risk cases either.

#### Combination
I used SMOTEENN to combine the oversampling and undersampling according to the following

![res10](/imgs/fig10.png?raw=true)

With the following balanced accuracy, confusion matrix and imbalanced classification report

![res11](/imgs/fig11.png?raw=true)

The problem with low f1 with respect to high-risk cases still exists.

### 2. Predicting credit risk with balanced random forest classifier and easy ensamble Adabooster classifies
####   Balanced random Forest Classifier
I implemented the code according to the following:
![res12](/imgs/fig12.png?raw=true)

The new model still doesn’t show improvement over the last ones
![res13](/imgs/fig13.png?raw=true)

This model detects the following features as the ones with the highest contribution.
![res14](/imgs/fig14.png?raw=true)

####   Easy Ensemble AdaBooster Classifier
The implementation:
![res15](/imgs/fig15.png?raw=true)

The outcome:
![res16](/imgs/fig16.png?raw=true)

This model has the highest f1 score for high risk cases (0.16)

## Summary
I used several machine learning tools to predict risk. The data set is very skewed towards low risk cases. After cleaning and scaling the data, I tested several machine learning tools to predict risk.  
Among all these models, easy ensemble with adabooster classifier has the highest f1 score for high-risk cases which is only 0.16. Since it is very important that the model can detect high risk cases and it is more important for it to be able to predict high risk cases rather than low risk cases, I don’t recommend any of the machine learning models discussed here for risk prediction.
