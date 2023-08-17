import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
import seaborn as sns
import plotly.express as px
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler
from scipy import stats
import wrangle as w


#model
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, precision_score, accuracy_score, recall_score, classification_report
from scipy import stats
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB

def relationship_graph(train, graph_title, feature, target, tk_label=None):
    '''
    This function will take the train, graph_title, feature, and target,
    and it will display a bargraph based on the information provided for the churn dataset 

    '''
    fig, ax =plt.subplots()
    plt.title(graph_title)
    sns.barplot(x=feature, y=target, data=train)
    population_stroke_rate = train.stroke.mean()

    # tick_label = ["Female", "Male"]
    # ax.set_xticklabels(tick_label)

    plt.axhline(population_stroke_rate, label="Population Stroke Rate")
    plt.legend()
    plt.show()


# Stats test functions
def chi_stats(train, feature, target):
    '''
    This function runs a chi2 stats test on feature and target variable.
    It returns the contingency table and results in a pandas DataFrame.
    '''
    # Create a contingency table
    contingency_table = pd.crosstab(train[feature], train[target])

    # Perform the chi-square test
    chi2, p_value, dof, expected = stats.chi2_contingency(contingency_table)
    
    # Decide whether to reject the null hypothesis
    alpha = 0.05
    if p_value > alpha:
        decision = "Fail to Reject Null Hypothesis"
    else:
        decision = "Reject Null Hypothesis"

    # Create a DataFrame for the results
    results = pd.DataFrame({
        'Chi-square statistic': [chi2],
        'p-value': [p_value], 
        'Decision': [decision]
    })

    # Return the contingency table and results DataFrame
    return results

def perform_t_test(stroke_sample, overall_mean, alpha=0.05):
    '''
    This function runs a One Sample T-Test on feature and target variable.
    It returns the contingency table and results in a pandas DataFrame.
    '''
    t, p = stats.ttest_1samp(stroke_sample, overall_mean)
    
    if p / 2 > alpha:
        return "We fail to reject the null hypothesis."
    elif t < 0:
        return "We fail to reject the null hypothesis."
    else:
        return "We reject the null hypothesis."
    

#creating X,y
def get_xy(df, target):
    '''
    This function generates X and y for train, validate, and test to use : X_train, y_train, X_validate, y_validate, X_test, y_test = get_xy()

    '''
    train, validate, test = w.split_data(df,target)

    X_train = train.drop([target], axis=1)
    y_train = train[target]
    X_validate = validate.drop([target], axis=1)
    y_validate = validate[target]
    X_test = test.drop([target], axis=1)
    y_test = test[target]
    return X_train,y_train,X_validate,y_validate,X_test,y_test


def create_models(seed=123):
    '''
    Create a list of machine learning models.
            Parameters:
                    seed (integer): random seed of the models
            Returns:
                    models (list): list containing the models
    This includes best fit hyperparamaenters                
    '''
    models = []
    models.append(('k_nearest_neighbors', KNeighborsClassifier(n_neighbors=100)))
    # models.append(('logistic_regression', LogisticRegression(random_state=seed)))
    models.append(('DecisionTreeClassifier', DecisionTreeClassifier(max_depth=3,min_samples_split=4,random_state=seed)))
    models.append(('random_forest', RandomForestClassifier(max_depth=3,random_state=seed)))
    models.append(('support_vector_machine', SVC(random_state=seed)))
    models.append(('naive_bayes', GaussianNB()))
    # models.append(('gradient_boosting', GradientBoostingClassifier(random_state=seed)))
    return models


def get_models(X_train, y_train, X_validate, y_validate):
    """
    Fits multiple machine learning models to the training data and evaluates their performance on the training and validation sets.

    Parameters:
    X_train (array-like): Training feature data.
    y_train (array-like): Training target data.
    X_validate (array-like): Validation feature data.
    y_validate (array-like): Validation target data.
    X_test (array-like): Test feature data (not used in this function).
    y_test (array-like): Test target data (not used in this function).

    Returns:
    pandas.DataFrame: A dataframe containing the model names, set (train or validate), accuracy, recall, and precision scores.
    """

    # create models list
    models = create_models(seed=123)

    # initialize results dataframe
    results = pd.DataFrame(columns=['model', 'set', 'accuracy'])

    # loop through models and fit/predict on train and validate sets
    for name, model in models:
        # fit the model with the training data
        model.fit(X_train, y_train)
        
        # make predictions with the training data
        train_predictions = model.predict(X_train)
        
        # calculate training accuracy, recall, and precision
        train_accuracy = accuracy_score(y_train, train_predictions)

        
        # make predictions with the validation data
        val_predictions = model.predict(X_validate)
        
        # calculate validation accuracy, recall, and precision
        val_accuracy = accuracy_score(y_validate, val_predictions)

        
        # append results to dataframe
        results = results.append({'model': name, 'set': 'train', 'accuracy': train_accuracy}, ignore_index=True)
        results = results.append({'model': name, 'set': 'validate', 'accuracy': val_accuracy}, ignore_index=True)

    return results


# test model function 
def rf_model(X_train, y_train, X_test, y_test):
    """
    Trains a Gradient Boosting Classifier on the given training data (X_train, y_train),
    makes predictions on the test data (X_test), and calculates accuracy, recall, and precision
    on the test set.

    Parameters:
        X_train (array-like): Training data features.
        y_train (array-like): Training data target labels.
        X_test (array-like): Test data features.
        y_test (array-like): Test data target labels.

    Returns:
        pandas.DataFrame: A DataFrame containing the results of the model evaluation on the test set.
                          The DataFrame has the following columns:
                          - 'model': Name of the model used ('gradient_boosting').
                          - 'set': Indicates the data set evaluated ('test').
                          - 'accuracy': Accuracy score on the test set.
                          - 'recall': Weighted recall score on the test set.
                          - 'precision': Weighted precision score on the test set.
    """
    # Create and fit the Gradient Boosting model
    model = RandomForestClassifier(random_state=3, min_samples_leaf=5, max_depth=6)
    model.fit(X_train, y_train)

    # Make predictions on the test set
    test_predictions = model.predict(X_test)

    # Calculate accuracy, recall, and precision on the test set
    test_accuracy = accuracy_score(y_test, test_predictions)
    # test_recall = recall_score(y_test, test_predictions, average='weighted')
    # test_precision = precision_score(y_test, test_predictions, average='weighted')

    # Create a results DataFrame
    results = pd.DataFrame({
        'model': ['Random Forest'],
        'set': ['test'],
        'accuracy': [test_accuracy]
    })

    return results

def run_support_vector(X_train, y_train, X_test, y_test):
    """
    Trains a Gradient Boosting Classifier on the given training data (X_train, y_train),
    makes predictions on the test data (X_test), and calculates accuracy, recall, and precision
    on the test set.

    Parameters:
        X_train (array-like): Training data features.
        y_train (array-like): Training data target labels.
        X_test (array-like): Test data features.
        y_test (array-like): Test data target labels.

    Returns:
        pandas.DataFrame: A DataFrame containing the results of the model evaluation on the test set.
                          The DataFrame has the following columns:
                          - 'model': Name of the model used ('gradient_boosting').
                          - 'set': Indicates the data set evaluated ('test').
                          - 'accuracy': Accuracy score on the test set.
                          - 'recall': Weighted recall score on the test set.
                          - 'precision': Weighted precision score on the test set.
    """
    # Create and fit the Gradient Boosting model
    model = SVC(random_state=123)
    model.fit(X_train, y_train)

    # Make predictions on the test set
    test_predictions = model.predict(X_test)

    # Calculate accuracy, recall, and precision on the test set
    test_accuracy = accuracy_score(y_test, test_predictions)
#     test_recall = recall_score(y_test, test_predictions, average='weighted')
#     test_precision = precision_score(y_test, test_predictions, average='weighted')

    # Create a results DataFrame
    results = pd.DataFrame({
        'model': ['support_vector'],
        'set': ['test'],
        'accuracy': [test_accuracy]
    })

    return results