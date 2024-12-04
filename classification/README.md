# Project 1 - Classification


## Overview

** Table of Contents **
- [Cortex Classification](#cortex-classification)



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
