# Project 1 - Classification


## Overview

** Table of Contents **
- [Cortex Classification](##introduction)


The data comes from a Kaggle competition called "[Acquired Shoppers Challenge](https://www.kaggle.com/competitions/acquire-valued-shoppers-challenge/data)". 
It includes retail store data on a number of companies and brands. I used the three files transactions.csv, trainHistory.csv, and offers.csv. They included transaction data on customers, history of offers given to customers and whether they returned or not, and lastly information about the offers. The objective of the project was to predict whether a customer would return to the store upon receiving a certain offer. 

It is a large dataset of about 22GB and I used snowflake to manage and interact with all the data. 

**Schemas:** 
_history_
**id** - A unique id representing a customer  
**chain** - An integer representing a store chain  
**offer** - An id representing a certain offer  
**market** - An id representing a geographical region  
**repeattrips** - The number of times the customer made a repeat purchase  
**repeater** - A boolean, equal to repeattrips > 0  
**offerdate** - The date a customer received the offer

_transactions  
_**id** - see above  
**chain** - see above  
**dept** - An aggregate grouping of the Category (e.g. water)  
**category** - The product category (e.g. sparkling water)  
**company** - An id of the company that sells the item  
**brand** - An id of the brand to which the item belongs  
**date** - The date of purchase  
**productsize** - The amount of the product purchase (e.g. 16 oz of water)  
**productmeasure** - The units of the product purchase (e.g. ounces)  
**purchasequantity** - The number of units purchased  
**purchaseamount** - The dollar amount of the purchase

_offers  
_**offer** - see above  
**category** - see above  
**quantity** - The number of units one must purchase to get the discount  
**company** - see above
**offervalue** - The dollar value of the offer  
**brand** - see above


## KPI


The desired KPI was

Plot cost per offer. 

Define baseline

Cost per offer. Then TOTAL = return

There is no way the baseline is better? Establish this!

How many people returned x value of a customer / cost of sending offer

value of customer = average shop x average num repeats after offer

Where I previously fucked up was I used only the history of people that had received offers. 73% of 27% will always be less than 27%, you genius. 73% of 100 is more tho. You bastard. Idiot.
This as well:
If I have a customer base of 100, and then I have a model with an accuracy of 70%. Would the number of money gained be TP times average gain per customer + TN * spend on each customer? Consider the test set size when you count this up to get the percentage!

Percentage of correct TP

Even if recall only gives 26% TP, you have to consider TN. 


****HEY**** Prepare notebooks for running. Then rum them all.
- Data preparation done. Move on to the next:)


### Model Registry

Time to log model:  53s
- You set your metrics etc. separetely
**Model.run**

It is when you have a reference to the model, and then call the inbuilt snoiwflake ml function to run it. meaning you don't load the model back into your notebook. You give it the X_test data and the function_name "predict". This takes about 6s, with some variability. I put it in a loop running it multiple times to see whether it stored the model on the backend in cache or something similar to improve inference speed, but no. Takes the same amount of time. 


**Model.predict**

In this case you load the model back into your workspace/notebook and then call model.predict like you would normally. It takes just under 6s, 5.8 ish, to load the model back into the notebokk and then 0.35s to do the prediction. Preferable if you have to do multiple predictions with different datasets. 

## Snowflake

As mentioned I used snowflake to manage and interact with the data. Here are some takeaways from working with snowflake for the first time:

Cost considerations:
- The real-time cost estimation.
	- When using the worksheets, costs are updated with each query in the logs so it is easy to monitor how much credits you are using. Additionally, resource monitors are able to monitor your usage and alert you.
	- When using the notebooks you have to start it before running any cells. At which point it will import all your dependencies. There is no update to your costs until you end the notebook instance. So the resource monitor is not alerting you when you reach a desired threshold, and you might be able to run everything you wanted, but then the next day you open it up and you have run out of credits. 

Notebook usability
- Some basic comforts like keyboard shortcuts etc. that you are used to from working in code editors or on jupyter notebooks don't yet exist. 
- The error handling is terrible. It is hard to debug both when writing SQL and Python as the messages often give little indication as to what failed. For SQL errors I would run them in a worksheet first to debug and then copy them over into a Notebook.
- The notebooks are somewhat unreliable in that they will stop working or throw unexplainable edits. Usually fixed by restarting the notebook.

Model Registry
- Ability to store the actual model with different versions and accompanying performance metrics. 
- Inference works great
	- Option 1: You can call the models methods using .run(), and then specify predict. This returns the predictions in a snowflake df in 6.9s using an XS Warehouse. 
	- Option 2: You can load the model object back into your work environment. It requires the exact same environment as when you added it to the registry. I am unable to avoid this error despite being in the same notebook, but you can add a force parameter to ignore the errors. Returns the predictions on a pandas df in 11.0s using a XS Warehouse. 
- It can be shared across snowflake accounts
- Drawbacks:
	- Incompatible with confusion matrices. They have an example in their docs, but I could not get it to work for me. 
	- If you are on a free trial and you run out of credits, you will not be able to get a hold of the model or your results. 


## Classification Models

See [data_preparation.ipynb](paste_link_here)

The Data Preparation was mainly using SQL to select the desired features and create two new ones. One a boolean for previous purchase of that product, and one for a previous purchase of a product in that category. Then doing some simple preprocessing by checking for null values, and splitting sotre_chain_id (160 uid's) into three categories based on number of transactions. Finally, after one-hot-coding the categorical parameters it was ready to be imported into   pandas df.

### Lazypredict

It's useful, but very resource-intensive and time-consuming to run. Used up more or less all $400 credits running on an S Warehouse. Ran for a few hours. 

I used the results below to select RandomForestClassifier as the model to use. 

| Model                         | Accuracy            | Balanced Accuracy  | ROC AUC             | F1 Score           | Time Taken          |
| ----------------------------- | ------------------- | ------------------ | ------------------- | ------------------ | ------------------- |
| NearestCentroid               | 0.6425715356741222  | 0.6024429963505807 | 0.6024429963505806  | 0.6564043966021561 | 0.08494138717651367 |
| GaussianNB                    | 0.6515994002249157  | 0.5983896179894634 | 0.5983896179894633  | 0.6621297541370642 | 0.09207797050476074 |
| KNeighborsClassifier          | 0.7113270023741097  | 0.5619663486869398 | 0.5619663486869397  | 0.678383020006725  | 43.834614753723145  |
| BernoulliNB                   | 0.7312570286142697  | 0.5597742739418267 | 0.5597742739418267  | 0.6814532885750648 | 0.10557317733764648 |
| BaggingClassifier             | 0.7315381731850557  | 0.5569189264412695 | 0.5569189264412695  | 0.6791366591845888 | 0.6451404094696045  |
| PassiveAggressiveClassifier   | 0.6835561664375859  | 0.5569050775402822 | 0.5569050775402822  | 0.6655869668050571 | 0.15097713470458984 |
| RandomForestClassifier        | 0.7315694114706985  | 0.5567610671857744 | 0.5567610671857744  | 0.6790080041395006 | 3.2399861812591553  |
| XGBClassifier                 | 0.7315381731850557  | 0.5567037343111041 | 0.5567037343111041  | 0.6789555387745458 | 0.31449270248413086 |
| DecisionTreeClassifier        | 0.7315381731850557  | 0.5567037343111041 | 0.5567037343111041  | 0.6789555387745458 | 0.13331866264343262 |
| SVC                           | 0.7315381731850557  | 0.5567037343111041 | 0.5567037343111041  | 0.6789555387745458 | 1360.5421550273895  |
| ExtraTreeClassifier           | 0.7315381731850557  | 0.5567037343111041 | 0.5567037343111041  | 0.6789555387745458 | 0.10158371925354004 |
| ExtraTreesClassifier          | 0.7315381731850557  | 0.5567037343111041 | 0.5567037343111041  | 0.6789555387745458 | 3.810225009918213   |
| AdaBoostClassifier            | 0.7315381731850557  | 0.5566678689560766 | 0.5566678689560766  | 0.6789253089049748 | 1.8869056701660156  |
| RidgeClassifierCV             | 0.7315381731850557  | 0.5566678689560766 | 0.5566678689560766  | 0.6789253089049748 | 0.12720847129821777 |
| RidgeClassifier               | 0.7315381731850557  | 0.5566678689560766 | 0.5566678689560766  | 0.6789253089049748 | 0.08763623237609863 |
| LogisticRegression            | 0.7315381731850557  | 0.5566678689560766 | 0.5566678689560766  | 0.6789253089049748 | 0.11116838455200195 |
| LinearSVC                     | 0.7315381731850557  | 0.5566678689560766 | 0.5566678689560766  | 0.6789253089049748 | 0.23384642601013184 |
| LinearDiscriminantAnalysis    | 0.7315381731850557  | 0.5566678689560766 | 0.5566678689560766  | 0.6789253089049748 | 0.1454603672027588  |
| CalibratedClassifierCV        | 0.7315381731850557  | 0.5566678689560766 | 0.5566678689560766  | 0.6789253089049748 | 0.7068722248077393  |
| LGBMClassifier                | 0.7315381731850557  | 0.5566678689560766 | 0.5566678689560766  | 0.6789253089049748 | 0.3807377815246582  |
| QuadraticDiscriminantAnalysis | 0.38607397226040235 | 0.5450668444800567 | 0.5450668444800568  | 0.3511332126252    | 0.10405135154724121 |
| Perceptron                    | 0.7122329126577533  | 0.5270822052793005 | 0.5270822052793004  | 0.6509602160061491 | 0.17434334754943848 |
| SGDClassifier                 | 0.7275709109084093  | 0.5                | 0.5                 | 0.6128367027455206 | 0.3092503547668457  |
| DummyClassifier               | 0.7275709109084093  | 0.5                | 0.5                 | 0.6128367027455206 | 0.06411385536193848 |
| NuSVC                         | 0.3881669373984756  | 0.4629030257268911 | 0.46290302572689124 | 0.3998141972849268 | 1763.2876596450806  |



**
### Random Forest

**Include link to notebook here!**


**Data Preparation
- Easy to import from database into snowflake df's and then turn those into Pandas df's
- Time to run the data imports and preparation (train/test split etc.) was less than 5 seconds. 

**Performance Metric**
- On the assumption that the FP cost was low I chose to use recall as my main performance metric. 
- I also kept track of the confusion matrices and accuracy.

**Hyperparameter optimisation**
I chose to use a randomised search to optimise the models parameters. Compared to grid search I thought it made more sense because I did not have any great reasons for choosing the various values to include in the param_grid. Additionally, it is more time and cost efficient. That being said, given the training time of XXX, the randomised search returned ok parameters, but did take up a lot of resources. In retrospect, it would be helpful to run the model a couple of times myself at first to identify reasonable parameters to input into the param_grid.
  

**Imbalanced dataset problem**
- I tried oversampling, undersampling, and smote and they gave improved results and were more or less identical to one another. 
- Using class_weight=’balanced’ in the model parameters was the best option. 
**Add some fucking numbers here. A cheeky table would be nice**
  

**Probability thresholds**
- Used different probability thresholds to improve recall. Settled on a probability threshold that gave almost 100% recal
## Cortex Classification

