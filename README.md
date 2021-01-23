# Problem Statement:
A Bank wants to take care of customer retention for its product: savings accounts. The bank wants you to identify customers likely to churn balances below the minimum balance. You have the customers information such as age, gender, demographics along with their transactions with the bank.
Your task as a data scientist would be to predict the propensity to churn for each customer.
# Live Demo:
check out this project live here [Lets go to Project](http://hmisweb.herokuapp.com/)

![alt text](https://github.com/HarryMishra/Bank_user_churn_prediction/blob/main/projectDemo.png)
# Approaching the Solution:


# Step One: Collecting Data
We take data From UCi machine learning reposatry. There are many other sources available for taking data like kaggle , open source databases etc.

# Step two: setup your computer:
First of all download all the required library and tools. and setup our Machine.
- Anaconda
- VS Code
- Numpy
- Pandas
- Sklearn
- Git
- Django
- setup vertual enviornment
- Create heroku account
- download heroku on local computer

# Step Three: Data Analisys:
Analyse the data with the help of matplotlib and seaborn library. done analisys by Univariate and Bivariate method and
find out meaningfull insights of the data. this insights help us to taking some decision regarding data engineering and model building and selection.
for whole analysys insights go to the final project file and read it.

# Step Four: Data Engineering:
After geting the data undestanding we move forward for the data cleaning or data engineering parts. In this data have lots have null values and there are many outliers we 
fill null values with some logical undestanding, delete unwanted columns, and standardize the data, encode the data also.
- Remove unwanted Columns.
- Fill Null values with analisys.
- Remove outliers
- create dummy variable for categorical data
- Scale the data with min max scaler
# Step five: Model building:
After data cleaning build a ML model on that. split data and build a five models
- Decision tree
- Random Forest
- SVM
- Logistic Regression
- KNN
after building model Evalute uits performance metrics find out confusion metrix , precision , recall and accuracy for each model.


# Step Six: HyperparameterTuning:
 For each model tune the hyper parameter with the help of gridSearch CV. 
 
 # Step Seven: Model Selection:
 
 SElect best model on the basis of Best performance we select random forest becouse its accuracy, precision, recall is better than other model.
 
 # Step Eight: Build Website and Deploye:
 We build a HtML CSS website and build Backend with Django and connect with model and deploy it on heroku platform
 



