import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
import seaborn as sns
import plotly.express as px
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler
from scipy import stats

def relationship_graph(train, graph_title, feature, target):
    '''
    This function will take the train, graph_title, feature, and target,
    and it will display a bargraph based on the information provided for the churn dataset 

    '''
    fig, ax =plt.subplots()
    plt.title(graph_title)
    sns.barplot(x=feature, y=target, data=train)
    population_stroke_rate = train.stroke.mean()

    tick_label = ["Female", "Male"]
    ax.set_xticklabels(tick_label)

    plt.axhline(population_stroke_rate, label="Population Stroke Rate")
    plt.legend()
    plt.show()



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