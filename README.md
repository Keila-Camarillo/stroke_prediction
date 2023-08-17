# stroke_prediction

# Project Description
This project aims to predict whether a patient is likely to get stroke based on the input parameters like gender, age, various diseases, and smoking status. 

# Project Goal
* Discover drivers of strokes
* Use drivers of strokes to develop machine learning models to predict if a patient having a stroke.
* Provide data based solutions to reduce strokes

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
            * replace nulls in mbi with the average bmi
    * Check and adjust column data types as needed.
    * Encode categorical variables.
    * Split the data into training, validation, and test sets (approximately 60/20/20 split).
    * Outliers were not removed

- Create Engineered columns from existing data
    * is_child: patients under the age of 18 years old
* Explore data in search of drivers of upsets

- Answer the following initial questions
    * Does Hypertension Affect Whether a Patient Stroke?
    * Does Heart Disease Affect Whether a Patient Stroke?
    * Does Being Married Affect Whether a Patient has a Stroke?
    * Does Residence Type Determine Whether a Patient Stroke?
    * Does Smoking Status Type Determine Whether a Patient Stroke?
    * Does Being a Child Determine Whether a Patient Stroke?
    * Does Average Glucose Level Affect Whether a Patient Stroke?
    * Does BMI Affect Whether a Patient Stroke?
    * Does Gender Affect Whether a Patient Stroke?

* Develop a Model to predict patient's strokes

    - Use drivers identified in explore to build predictive models of different types
    - Evaluate models on train and validate data
    - Select the best model based on accuracy
    - Evaluate the best model on test data

* Draw conclusions



# Data Dictionary
Here is a data dictionary describing the features in the dataset:

| Feature | Definition |
|:--------|:-----------|
|gender| Gender of the individual (Male, Female, Other)|
|age| Age of the individual|
|hypertension| Whether the individual has hypertension (1 for yes, 0 for no)|
|heart_disease| Whether the individual has heart disease (1 for yes, 0 for no)|
|ever_married| Marital status of the individual (Yes or No)|
|work_type| Type of work the individual is engaged in (e.g., children, private, self-employed)|
|residence_type| Type of residence (Urban or Rural)|
|avg_glucose_level| Average glucose level in the individual's blood|
|bmi| Body Mass Index (BMI) of the individual|
|smoking_status| Smoking status of the individual (e.g., never smoked, formerly smoked, currently smoking)|
| is_child | Whether the individual is a child (1 for yes if age < 18, 0 for no) |
| stroke | 	Whether the individual had a stroke (1 for yes, 0 for no) |

# Steps to Reproduce
    1. Clone this repo.
    2. Acquire the data from Kaggle
    3. Place the data in the file containing the cloned repo.
    4. Run notebook.

# Takeaways and Conclusions
* Those with hypertension, high gluclose levels, and heart diease were having strokes more often.

# Recommendations
* Those with hypertension, high gluclose levels, and heart diease were having strokes more often.
    - The data should consist 
