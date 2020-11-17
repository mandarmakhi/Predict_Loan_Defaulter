![](https://img.shields.io/github/followers/mandarmakhi?label=Follow%40mandarmakhi&style=social)
![](https://img.shields.io/github/forks/mandarmakhi/Predict_Loan_Defaulter?label=Fork&style=social)
![](https://img.shields.io/github/stars/mandarmakhi/Predict_Loan_Defaulter?style=social)
![](https://img.shields.io/github/watchers/mandarmakhi/Predict_Loan_Defaulter?style=social)
![](https://img.shields.io/github/issues/mandarmakhi/Predict_Loan_Defaulter)
![Hits](https://hits.seeyoufarm.com/api/count/incr/badge.svg?url=https://mandarmakhi.github.io/Predict_Loan_Defaulter/)]
![](https://img.shields.io/github/repo-size/mandarmakhi/Predict_Loan_Defaulter)
![](https://img.shields.io/github/languages/code-size/mandarmakhi/Predict_Loan_Defaulter)



# Predict_Loan_Defaulter

In this project we have to predict the Whether the customer will fall under default or not.The main use of classification models is to score the likelihood of an event occuring. For loan data, the model will be used to predict whether a loan will be paid off in full or the loan needs to be charged off and possibly go into default. You can use the model to score the quality of current loans and identify the ones most likely to default.😀🏹 🥇💯

You will Need Anaconda to Execute all the Codes So Install it first and then Go through the Below Codes.
To Download anaconda, [Click Here](https://www.anaconda.com/products/individual).

Then after installation of anaconda open anaconda navigator and install the jupyter notebook inside the anaconda environment then execute the .ipynb code inside the jupyter notebook.

***
## Objective
To classify if the borrower will default the loan using borrower’s finance history. That means, given a set of new predictor variables, we need to predict the target variable as
1 -> Defaulter
0 -> Non-Defaulter.
A MIS_Status of a Bank will be Default when a borrower would not be able to make payment on time or delays payments , refuses or avoids payment. An Individual , Organizations and even states could fall under default. To avoid the loss we have come up with a model which could predict whether one could fall under Default or will he be able to pay in full.
***
## PROJECT FLOW
1. Data Preparation – Out of the 26 features in our dataset, Many of them had empty cells. We have Imputed such cells. Also, the features which didn’t seem relevant to our goal were removed.
2. Data Modelling – Data will be divided into two parts. Training data and testing data. The plan is to develop a  model. The model will be developed on Training data and Evaluated on Test data.
3. Model Evaluation – After developing the Model it is evaluated on accuracy, precision, recall and F1 score .
4. Model Deployment – The Model is deployed using flask.
***
## DATA Preparation
### EDA(Exploratory Data Analysis)
Total number of rows – 149999
Total number of columns – 26
Date ranges (Year’s) – 1962 to 2007
Missing Values -  110938
Target Variable -  MIS_Status
The we find Unique Values of each variable(Column).
### Loan Data: Data types
#### Target Variable:
1. Charge OFF
2. Paid in Full
#### Categorical:
1. New Exist
2. LowDoc
3. Franchise Code
4. Urban Rural
5. RevLineCr
6. MIS_Status
#### Numerical:
1.  Zip
2. CCSC
3. ApprovalFy
4. Term
5. NoEmp
6. Create Job
7. Retained Job
8. Disbursement Gross
9. Balance Gross
10. Chg Off Prin  Gr
11. GrAppv
12. SBA_Appv
#### Unique:
1. Name
2. City
3. State
4. Bank
5. BankState
6. ApprovalDate
7. ChgoffDate
8. DisbursementdaDate

### DATA Engineerin
The dataset had many empty and irrelevant features They have been removed.
● String values have been formatted to integers.
● Categorical values have been transformed to numerical.
● Redundant variables have been dropped.
● Filled NAN values with mode values of corresponding columns.

1. Before Dealing with the Missing values
2. After Dealing with the Missing values 

### Variable Transformation
“DisbursementGross”, “BalanceGross”, “ChgOffPrinGr”, “GrAppv”, “SBA_Appv”  these variables are converted to numeric.
● MIS_Status : PIF : 0(110831) ChgOff : 1(38926) 
● Franchise Code: - No Franchise : 0(144504) ,  Franchise :1(5253)
● NewExist : 1 = Existing Business(101773), 2 = New Business(47984)
● Low Doc : No(0) : 137716    Yes(1) :12041
● RevLineCr : No (0):- 95157   Yes(1) :-54600

### Univariate Graph Description
1. City – LOS ANGELES, NEW YORK & MIAMI Are Top Three city.
2. State – CA, NY & TX Are Top Three State.
3. Bank -  Bank OF AMERICA NATL ASSOC, CITIZENS BANK NATL     ASSOC & CAPITAL ONE NATL ASSOC Are Top Three Bank.
4. Bank State – NC, RI & IL Are Top Three Bank State.
5. CCSC – There are total 1184 Unique codes.
6. APPROVAL DATE -  Most of the Small Business Administration Commitment were Approved  on 1997-09-30.
7. Term – Most of the Loans have a term of  84.
8. NoEmp –  Number of Business Employees(SUM) -  30787
9. NewExist –  Existing Business -101648   New Business-47984
10. Create job & Retained job – Number of Jobs created 37130
11. LowDoc – LowDoc Loan program: Y=12041  , N=137632
12. Urban Rural – Urban-81724 rural-51370
13. RevLineCr - Revolving Line of Credit Yes=71503, NO=49780
14. MIS_Status(target Variable) - Loan Status CHGOFF(high risk)-38926, PIF( lower risk)- 110831
15. Disbursement Date –Most of loan amount was Disbursed on 2006-05-31 
16. GrAppv –  Minimum-200 and Maximum-4000000
17. SBA_Appv – Minimum-100 and Maximum-4000000

### Bivariate Graph Description
● Bank With Highest Gross Approval – The Top 3 bank with highest Gross Approval amount are SMALL BUS. GROWTH CORP, FIRSTMERIT BANK, N.A and GE CAP. SMALL BUS FINAN CORP.
● Bank State With Highest Gross Approval – The top 3 Bank States with highest Gross Approval are ILLINOIS , OHIO and TEXAS
● Urban/Rural with Gross Approval – People Staying in Urban areas are having  more Gross Approval than the other
● Urban/Rural with Gross Approval and MIS_Status – People staying in Rural with Gross Approval of less than 100000 are mostly falling under PIF as Compared to the Urban.
● NewExist With Term period and MIS_Status -  From the above graph we can see the people with Existing Bussiness with Term period of 120 fall under PIF
● GrAppv,LowDoc with MIS_Status- People with No LowDoc Loan Program are mostly under PIF 
● RevLineCr with MIS_Status - People with No Revolving Line of Credit fall more under PIF than the people with those.
● NewExist  and MIS_Status – People with the Existing Business are more falling under PIF
***
## MODEL BUILDING
### Feature Importance
Using ExtraTreesClassifier, feature importance is determined and unnecessary variables are removed again. Below columns shows high Importance:
 “Zip”, “CCSC” , “ApprovalFY” , “Term”, “UrbanRural” , “LowDoc”, “DisbursementGross”,“GrAppv”,“SBA_Appv”
We are dropping ChgoffPrinGr because it depends on the target variable.

### MoDels Built
SGDClassifier.  
Support Vector Machine (SVM) 
KNN (K- Nearest Neighbour)
Random Forest 
Random Forest with RandomizedSearchCV
Random Forest with SMOTE
XGBClassifier

### Model Evaluation
For our Loan default prediction project, False Negatives Rate is the best metric to evaluate the model. Lower the number of false negatives, better the model is.
In this project, False negative is when model predicting “a borrower will not default a loan even though he will “. Our model cannot afford having higher False Negatives as it leads to negative impact on the investors and the credibility of the company. So, we evaluated our models using the number of False negatives and accuracies.

### MODEL DEPLOYMENT
We did Model deployment using Flask. Flask is a micro web framework written in Python. It can create a REST API that allows you to send data, and receive a prediction as a response.


## Installation Process
1. open Command Prompt(cmd)
2. Inside cmd we can open folder n crate env foldr and download all library ### command(conda create--prefix ./env pandas numpy matplotlib scikit-learn jupyter flask)
3. Activate anaconda in the cmd through the command  ##### Active Conda.    
4. After activate conda the give path using Active conda cd (path of the env)
5. Opening the env folder then run active cd (main foldr path)
6. Run the app1.py file in cmd using command ### python app1.py
7. After run command all library activated then one localhost address show in cmd copy address
8. The localhost address past in web browser and run
