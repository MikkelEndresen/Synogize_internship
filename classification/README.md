# Project 1 - Classification


## Overview

Note, all code provided for this project are jupyter notebooks downloaded from snowflake. As such, they include functions exclusive to snowflake and are not possible to run outside of a snowflake environment. 

**Table of Contents**
- [Cortex Classification](#cortex-classification)
- [Data Overview](#brief-overview-of-the-data)
- [Model Registry](#model-registry)


## Brief overview of the data

The data comes from a Kaggle competition called "[Acquired Shoppers Challenge](https://www.kaggle.com/competitions/acquire-valued-shoppers-challenge/data)". 
It includes retail store data on a number of companies and brands. I used the three files transactions.csv, trainHistory.csv, and offers.csv. They included transaction data on customers, history of offers given to customers and whether they returned or not, and lastly information about the offers. The objective of the project was to predict whether a customer would return to the store upon receiving a certain offer. 

It is a large dataset of about 22GB and I used snowflake to manage and interact with all the data. 

**Schemas:** <br>
_history_<br>
**id** - A unique id representing a customer  
**chain** - An integer representing a store chain  
**offer** - An id representing a certain offer  
**market** - An id representing a geographical region  
**repeattrips** - The number of times the customer made a repeat purchase  
**repeater** - A boolean, equal to repeattrips > 0  
**offerdate** - The date a customer received the offer

_transactions_<br>
**id** - see above  
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

_offers_<br>
**offer** - see above  
**category** - see above  
**quantity** - The number of units one must purchase to get the discount  
**company** - see above
**offervalue** - The dollar value of the offer  
**brand** - see above


## Model Registry

Link to notebook: []

This is a snowflake function that allows you to save your ml models, when created by using the supported libraries like sklearn and pytorch. It comes with several different attributes, like you can set your evaluation metrics and store relevant information with the model. 

Interestingly there are two main methods of running inference on saved models. 

Time to log model:  53s
- You set your metrics etc. separetely
<todo> Create seperate ntoebook for this. Create several different random forest models with </todo>


**Model.run**

It is when you have a reference to the model, and then call the inbuilt snoiwflake ml function to run it. meaning you don't load the model back into your notebook. You give it the X_test data and the function_name "predict". This takes about 6s, with some variability. I put it in a loop running it multiple times to see whether it stored the model on the backend in cache or something similar to improve inference speed, but no. Takes the same amount of time. 


**Model.predict**

In this case you load the model back into your workspace/notebook and then call model.predict like you would normally. It takes just under 6s, 5.8 ish, to load the model back into the notebokk and then 0.35s to do the prediction. Preferable if you have to do multiple predictions with different datasets. 


## Cortex Classification


Snowflake has an "inbuilt" classification that is based on a gradient boost machine. I tried it on this classification problem in order to compare it to the performance of my manually defined models both in terms of accuracy, but also in usability and time. 

Main Takeaways:
- The data preparation is a bit inconvenient. You can use python, but it does require you to load the data into df's before creating a new table in snowfalke. The second option is to use sql so a lot of the inbuilt comforts form pandas is gone, but it is definietlely doable. Either way it is not too troublesome compared to the alternative. 
- It is very simple and easy to use. 
- In terms of XAI, it has an inbuilt explainability function using SHAP, but I could not make it work despite utilising the provided code in the docs. However, it does have a feature importance function that works.
- There are inbuilt functinos to calcualte the most common performance metrics and also to calcualte probability thresholds which I found very useful. 

- Considerably slower to train and run than sklearn in notebooks. GET SOME ACTUAL NUMBERS HERE! <TODO>
- Easy to use and results in a model comparable to the equivalent in sklearn. GET SOME ACTUAL NUMBERS HERE! <TODO>


Read about it here: [Snowflake Cortex Classifcation](https://docs.snowflake.com/en/user-guide/ml-functions/classification)

[Snowflake notebook with my code]( link )
