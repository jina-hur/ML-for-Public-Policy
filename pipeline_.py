### 5/30 with split equally version
'''
Build simple and modular functions to be utilized for machine learning,
including analyzing data, training classification models, and evaluating models.
'''

import datetime
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC, SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, r2_score, precision_score, recall_score,\
    mean_absolute_error, mean_squared_error
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV

#1) Read data
def read_data(filename):
    '''
    Read csv data into pandas dataframe.
        Inputs: filename
        Output: pandas dataframe
    '''
    return pd.read_csv(filename)


#2) Explore data
def find_min_max(dataframe, col):
    '''
    Find minimum and maximum value of a column.
        Inputs: dataframe, column name (str)
        Output: minimum and maximum in tuple
    '''
    col_min = dataframe[col].min()
    col_max = dataframe[col].max()
    return col_min, col_max

def find_mean_std(dataframe):
    '''
    Find mean and standard deviation for each columns in a dataframe.
        Inputs: dataframe
        Output: mean and standard deviation
    '''
    return dataframe.describe().loc[['mean', 'std']]

def plot_distribution(dataframe, figure_size, x_val, y_label, x_label, title):
    '''
    Plot visualization graph in cdf form
        Inputs: dataframe after resetting index
                figure size (tuple)
        Output: graph visualization of cdf
    '''
    sns.set(rc={'figure.figsize':figure_size})
    sns.lineplot(y=dataframe.index, x=x_val, data=dataframe)
    plt.ylabel(y_label)
    plt.xlabel(x_label)
    plt.title(title)
    return plt.show()


#3) Create training and testing sets
def create_train_test(dataframe, test_split_size, rand_state):
    '''
    Split into train and test data
        Inputs: dataframe, split size, random state
        Output: train data, test data
    '''
    train_df, test_df = train_test_split(dataframe, test_size=test_split_size,
                                         random_state=rand_state)
    return train_df, test_df


#4) Pre-process data
def convert_to_numeric(dataframe, col_list):
    '''
    Convert columns in a dataframe into numeric features.
        Inputs: dataframe, list of columns to be converted to numeric
        Output: dataframe
    '''
    for col in col_list:
        dataframe[col] = dataframe[col].astype(int)
    return dataframe

def impute_missing_values_train(train_df, col_list):
    '''
    Impute missing values with median for training data.
        Inputs: training dataframe, list of columns to be imputed
        Output: training dataframe
    '''
    for col in col_list:
        train_df[col].fillna(train_df[col].median(), inplace=True)

def impute_missing_values_test(train_df, test_df, col_list):
    '''
    Impute missing values with median for testing data.
        Inputs: training dataframe, testing dataframe,
                list of columns to be imputed
        Output: testing dataframe
    '''
    for col in col_list:
        test_df[col].fillna(train_df[col].median(), inplace=True)

def normalize_train(train_df, col_list):
    '''
    Normalize training data with means and standard deviations.
        Inputs: training dataframe, list of columns to be normalized
        Output: training dataframe
    '''
    for col in col_list:
        df_col = pd.DataFrame(train_df[col])
        train_df[col] = (df_col - df_col.mean()) / df_col.std()
    return train_df

def normalize_test(train_df, test_df, col_list):
    '''
    Normalize testing data with means and standard deviations
    from training data.
        Inputs: training dataframe, testing dataframe,
                list of columns to be normalized
        Output: testing dataframe
    '''
    for col in col_list:
        train_col = pd.DataFrame(train_df[col])
        test_col = pd.DataFrame(test_df[col])
        test_df[col] = (test_col - train_col.mean()) / train_col.std()
    return test_df


#5) Generate features
def one_hot_encoding(dataframe, col, prefix_name):
    '''
    Create columns for categorical features to avoid ordering of features.
        Inputs: dataframe, column to convert, prefix name for new columns
        Output: dataframe with one-hot-coded columns
    '''
    coded_df = pd.get_dummies(dataframe[col], prefix=prefix_name)
    return coded_df

def discretize_cont_var(dataframe, col, num_categories, labels_lst):
    '''
    Discretize continuous variables.
        Inputs: dataframe, column, nunber of discrete categories, labels list
        Output: None
    '''
    dataframe[col] = pd.cut(dataframe[col], num_categories, labels=labels_lst)

def categorize_col_by_n_groups(df, col, n, lower_bound, upper_bound):
    '''
    Categorizes the values of the given columns by n groups of equal length,
    bounded by the given lower/upper bounds.

    Inputs:
        df (pd.DataFrame): Pandas dataframe being explored
        col (str): name of the column to be sectioned
        n (int): number of groups
        lower_bound (float): lower bound defining the range of the values in 
                             the given column
        upper_bound (float): upper bound defining the range of the values in
                             the given column
    Output:
        (pd.DataFrame): Pandas dataframe with n groups of categorized values.
                        Labeling scheme of each group is 'g1', 'g2', ..., 'gn'.
    '''
    bins = np.linspace(lower_bound, upper_bound, num=n+1)
    labels = ['g' + str(x+1) for x in range(n)]
    return pd.cut(df[col], bins, labels=labels, include_lowest=True)

def categorize_col_equally_by_n_groups(df, col, n):
    '''
    Categorizes the values of the given columns by n groups of equal length,
    bounded by the given lower/upper bounds.

    Inputs:
        df (pd.DataFrame): Pandas dataframe being explored
        col (str): name of the column to be sectioned
        n (int): number of groups
        lower_bound (float): lower bound defining the range of the values in 
                             the given column
        upper_bound (float): upper bound defining the range of the values in
                             the given column
    Output:
        (pd.DataFrame): Pandas dataframe with n groups of categorized values.
                        Labeling scheme of each group is 'g1', 'g2', ..., 'gn'.
    '''
    labels = ['g' + str(x+1) for x in range(n)]
    return pd.qcut(df[col], n, labels=labels)

#6) Build pipeline for training and testing machine learning models
def build_apply_model(X_train, Y_train, X_test, Y_test, MODELS, GRID, TYPE, cv):
    '''
    Build and train classifiers on machine learning model.
        Inputs: train data, test data, model name, grid, model type
        Output: results of classifiers including model type, parameteres,
                and accuracy score along with y_predict values
    '''

    start = datetime.datetime.now()
    results = []
    if TYPE =='classifier':
        scoring  ='accuracy'
    if TYPE =='regression':
        scoring = 'r2'

    for model_key in MODELS.keys():
        print("Training model:", model_key)            
        gs = GridSearchCV(estimator=MODELS[model_key],
                          param_grid=GRID[model_key],
                          scoring=scoring,
                          cv=cv)
        gs = gs.fit(X_train, Y_train)
        best_score = gs.best_score_
        best_params = gs.best_params_
        print('best score', "|", best_score, 'best params', "|", best_params)
        best_model = gs.best_estimator_
        best_model.fit(X_train, Y_train)
        Y_pred = best_model.predict(X_test)

        if TYPE =='regression':
            r2 =  r2_score(Y_test, Y_pred)
            MAE = mean_absolute_error(Y_test, Y_pred)
            MSE = mean_squared_error(Y_test, Y_pred)
            results.append([model_key, best_params, r2, MAE, MSE, best_model])
        elif TYPE == 'classifier':
            accuracy = evaluate_classifiers(Y_test, Y_pred)
            precision = get_precision(Y_test, Y_pred)
            recall = get_recall(Y_test, Y_pred)
            results.append([model_key, best_params, accuracy, precision, recall, best_model])

        else:
            print('Choose type : "regression" or "classifier"')
            break

    result = pd.DataFrame(results)
    if TYPE =='regression':
        result = result.rename(columns={0: 'Model', 1: 
                               'Parameters', 2:'R2_score', 3:'MAE', 4:'MSE',
                               5:'best_model'})
    elif TYPE == 'classifier':
        result = result.rename(columns={0:'Model', 1:'Parameters', 2:'Accuracy Score',
                                        3:'Precision Score', 4:'Recall Score' ,5:'best_model'})

    stop = datetime.datetime.now()
    print("Time Elapsed:", stop - start)
    return result


#7) Evaluate classifiers
def evaluate_classifiers(y_true, y_pred):
    '''
    Evaluate classifiers with accuracy score.
        Inputs: true y-value, predicted y-value
        Output: accuracy score
    '''
    return accuracy_score(y_true, y_pred)

def get_precision(y_true, y_pred):
    '''
    Evaluate classifiers with precision score.
        Inputs: true y-value, predicted y-value
        Output: accuracy score
    '''
    return precision_score(y_true, y_pred, average='macro')

def get_recall(y_true, y_pred):
    '''
    Evaluate classifiers with recall score.
        Inputs: true y-value, predicted y-value
        Output: accuracy score
    '''
    return recall_score(y_true, y_pred, average='macro')


#8) Summarize model results
def summarize_best_model_result(df):
    '''
    Summarize the model with best results.
        Inputs: dataframe
        Output: information about the best model
    '''
    for colname in df.columns:
        print(colname, ': ', df[colname].head(1).values[0])

def obtain_match_rate(table, col_list, true_col):
    '''
    Obtain match rate of a column to another.
        Inputs: dataframe, list of columns for comparisons, true column as a 
                basis for comparisons to be made against
        Outputs: match rate between columns
    '''
    for col in col_list:
        val = table[table[col] == table[true_col]].count()[col] / len(table)
        print(col, ': ', round(val, 2))
