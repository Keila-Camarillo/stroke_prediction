# stroke_prediction

# Project Description
This project aims to predict whether a patient is likely to get stroke based on the input parameters like gender, age, various diseases, and smoking status. 

# Project Goal
* 

# Initial Thoughts

The initial hypothesis for this project is that patient's with high bmi,hypertension and heart disease may be significant drivers of strokes.

# The Plan

# Acquire
    * Obtain the dataset from [Kaggle](https://www.kaggle.com/datasets/fedesoriano/stroke-prediction-dataset).
    * The dataset contained 5110 rows and 12 columns before cleaning.
    * Each row represents a patient.
    * Each column provides relevent information about a patient.

# Prepare
    * Perform the following preparation actions on the dataset:
        * Filter out columns that do not provide useful information.
        * Rename columns to improve readability.
        * Handle null values:
            * 
            * 
    * Check and adjust column data types as needed.
    * 
    * Encode categorical variables.
    * Split the data into training, validation, and test sets (approximately 60/20/20 split).
    * Remove outliers (2641 outliers removed based on falling outside 3 standard deviations).

- Create Engineered columns from existing data
    * 
* Explore data in search of drivers of upsets

- Answer the following initial questions
    * 

* Develop a Model to predict patient's strokes

    - Use drivers identified in explore to build predictive models of different types
    - Evaluate models on train and validate data
    - Select the best model based on lowest RMSE and highest R2
    - Evaluate the best model on test data

* Draw conclusions



# Data Dictionary
Here is a data dictionary describing the features in the dataset:

| Feature | Definition |
|:--------|:-----------|
|Feature| definition|
|Feature| definition|
|Feature| definition|
|Feature| definition|
|Feature| definition|


# Steps to Reproduce
    1. Clone this repo.
    2. Acquire the data from Kaggle
    3. Place the data in the file containing the cloned repo.
    4. Run notebook.

# Takeaways and Conclusions
* 

# Recommendations
* To increase the model performance an additional feature
    - 
