# Credit-Card-Fraud-Detection-SMOTE-ADASYN
With Machine Learning (ML) techniques we can efficiently discover credıt card fraudulent patterns and predict transactions that are probably to be fraudulent. In this study, ı tried to balance the unbalanced data with SMOTE and ADASYN methods with the classification algorithms I determined and find the best possible algorithm. 
- The Plotly Library do not seems as output. You can reach the all of the visualization [https://www.kaggle.com/mirzazer/credit-card-fraud-detection-smote-adasyn]

![Resim1](https://user-images.githubusercontent.com/88277713/157431834-f38b2f3e-7574-4dc3-9585-f7bbe005e4b4.png)

# CONTENT

(1. cONCLUSN)[https://github.com/Dodger22/Credit-Card-Fraud-Detection-SMOTE-ADASYN/blob/main/README.md#conclusion]

# CREDIT CARD FRAUD DETECTION

## INTRODUCTION
Credit card fraud happens when consumers give their credit card number to unfamiliar individuals, when cards are lost or stolen, when mail is diverted from the intended recipient and taken by criminals, or when employees of a business copy the cards or card numbers of a cardholder

In recent years credit card usage is predominant in modern day society and credit card fraud is keep on growing. Financial losses due to fraud affect not only merchants and banks (e.g. reimbursements), but also individual clients. If the bank loses money, customers eventually pay as well through higher interest rates, higher membership fees, etc. Fraud may also affect the reputation and image of a merchant causing non-financial losses that, though difficult to quantify in the short term, may become visible in the long period.
A Fraud Detection System (FDS) should not only detect fraud cases efficiently, but also be cost-effective in the sense that the cost invested in transaction screening should not be higher than the loss due to frauds . The predictive model scores each transaction with high or low risk of fraud and those with high risk generate alerts. Investigators check these alerts and provide a feedback for each alert, i.e. true positive (fraud) or false positive (genuine).

Most banks considers huge transactions, among which very few is fraudulent, often less than 0.1% . Also, only a limited number of transactions can be checked by fraud investigators, i.e. we cannot ask a human person to check all transactions one by one if it is fraudulent or not.

Alternatively, with Machine Learning (ML) techniques we can efficiently discover fraudulent patterns and predict transactions that are probably to be fraudulent. ML techniques consist in inferring a prediction model on the basis of a set of examples. The model is in most cases a parametric function, which allows predicting the likelihood of a transaction to be fraud, given a set of features describing the transaction.

## METHODOLOGY

Fraud detection is a binary classification task in which any transaction will be predicted and labeled as a fraud or legit. In this Notebook state of the art classification techniques were tried for this task and their performances were compared.

   -Logistic Regression
   -Linear Discriminant Analysis
   -KNeighbors Classifier
   -RandomForest Classifier
   -Decision Tree Classifier
   -XGB Classifier
   -GaussianNB
   -Gradient Boosting Classifier
   -LGBM Classifier


![Resim2](https://user-images.githubusercontent.com/88277713/157432026-d1dd20f2-8a6a-4643-8e3e-4a98b2345c51.png)


 The dataset contains transactions made by credit cards in September 2013 by European cardholders. This dataset presents transactions that occurred in two days, where we have 492 frauds out of 284,807 transactions. The dataset is highly unbalanced, the positive class (frauds) account for 0.172% of all transactions.

It contains only numerical input variables which are the result of a PCA transformation. Unfortunately, due to confidentiality issues, we cannot provide the original features and more background information about the data. 

Features V1, V2, … V28 are the principal components obtained with PCA, the only features which have not been transformed with PCA are 'Time' and 'Amount'. Feature 'Time' contains the seconds elapsed between each transaction and the first transaction in the dataset. 

The feature 'Amount' is the transaction Amount, this feature can be used for example-dependant cost-sensitive learning. Feature 'Class' is the response variable and it takes value 1 in case of fraud and 0 otherwise. 

There are not any null variable. The data set contains 284,807 transactions. The mean value of all transactions is 88.35 USD while the largest transaction recorded in this data set amounts to 25,691 USD.

![Resim3](https://user-images.githubusercontent.com/88277713/157432124-896d44b3-5b9f-4cce-92f3-a356e81bba21.png)

However, as you might be guessing right now based on the mean and maximum, the distribution of the monetary value of all transactions is heavily right-skewed. The vast majority of transactions are relatively small and only a tiny fraction of transactions comes even close to the maximum. 

![Resim4](https://user-images.githubusercontent.com/88277713/157432173-2417be6c-416d-4716-ba8a-0be844eb4940.png)


As you can see, there are 284315 "Not Fraud" transaction and 492 "Fraud" transaction . Only %0.17 transaction is "Fraud". "Not Fraud " transaction prediction might be very easy but "Not Frauds" transactions are very low according to "Not Fraud" transaction so "Not Fraud" transaction predicting is hard. 


![Resim5](https://user-images.githubusercontent.com/88277713/157432244-8670b5f6-ed1f-4c75-bc9d-00b6eeda8d36.png)


When we scale the transactions hourly, we can see that between 6 and 24 frauds have increased and the density has increased. When we do the same scaling in minutes, we can see that the distribution is equal, but the density of fraud transactions is still dominant. However, we should not forget that only 0.17% of transactions are fraudulent transactions.

![Resim10](https://user-images.githubusercontent.com/88277713/157432338-e418566c-d91d-479b-8eb8-dfb398b5f0a3.png)

![Resim8](https://user-images.githubusercontent.com/88277713/157432370-2c0b7544-1a36-4ab7-8452-936b4afe38bc.png)

I wanted to show the ROC curve for each algorithm anyway. Random Forest Classifier/ADASYN and XGB Classifier/ADASYN seem to be the best values.

![Resim9](https://user-images.githubusercontent.com/88277713/157432470-0cf6a47c-e638-4858-83df-fc2bf803e563.png)

## CONCLUSION

After using all the methods we determined, I decided that the method with the Classification table the most logical and efficient confusion matrix results is Random Forest Classifier/ADASYN. But you can also examine the tables of other methods from my kaggle or github account.
I think Random Forest Classifier/ADASYN method is better. When we look the charts, ı see the results are beter on the Random Forest Classifier/ADASYN charts. But don't forget, did not use Future Selection methods and when I got data , the data was PCA format so see the outiler data or noisy data is very hard. I would not wanted to incorrect predict. The data already was İnbalanced.
There are 35 False Positive in Forest Classifier/ADASYN method. This means, per 284772 transactions will have 35 wrong predict. When you looking first it can be looks good. But there are doing millions transactions by customers every day in the bank. This means, the banks might lock up hundered of customers account unnecessary and this would reduce bank confidence.This method can be developed with different methods and implement feature selection or Cross Validation methods. The data reviewing again with Neural Network or using Genetic algorithms.







