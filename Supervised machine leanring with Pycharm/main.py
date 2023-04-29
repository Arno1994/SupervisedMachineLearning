# The dataset used in the project is for the detection of fraudulent credit card transactions

# Import libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn import preprocessing


# Define functions for processing the data from a csv file
def describe_data(dataf):
    # Show first five rows
    dataf.head()

    # Check the amount of data points and variables
    print('Number of rows: {} Number of columns: {}\n'.format(dataf.shape[0], dataf.shape[1]))

    # Set the decimal precision to 2 decimal places
    pd.set_option('display.float_format', lambda x: '%.2f' % x)
    # The describe function gives us insight into the statistical properties of the columns
    stats = dataf.describe()
    selected_stats = stats.loc[["mean", "std", "min", "max"]].transpose()  # select relevant rows
    print(selected_stats)  # Show the results
    print('')

    # Count missing values
    missing = dataf.isnull().sum()
    relevant_missing = pd.DataFrame(missing, columns=["missing"])
    # create a list of row names with values greater than zero
    missing_rows = relevant_missing.index[relevant_missing.gt(0).any(axis=1)].tolist()
    if len(missing_rows) > 0:
        # print the list of row names with missing values
        print("Rows with missing values:", missing_rows)
    else:
        print('No missing values detected in data')
        print('')

    # Check the data types of each column to see if there are any strings between the numbers
    # get the data types of the columns
    data_types = dataf.info()
    # get the column names with object data type
    object_columns = dataf.select_dtypes(include=['object']).columns.tolist()
    # count the number of columns with object data type
    object_columns_count = len(object_columns)
    if object_columns_count > 0:
        # print the column names with object data type
        print("Columns with object data type:", object_columns)
        # print the number of columns with object data type
        print("Number of columns with object data type:", object_columns_count)
    else:
        print('No objects detected in the dataset')


def plot_histograms(dataf):
    # Plot histograms
    histograms = dataf.hist(color='black', grid=False, figsize=(15, 10))
    plt.tight_layout()
    plt.show()


def correlation(dataf):
    # Create a new dataframe by dropping all columns deemed unnecessary
    corr_df = dataf.drop(["Time", "Amount"], axis=1).corr()
    labels = corr_df.columns
    correlations = corr_df.corr()
    mask_ut = np.triu(np.ones(corr_df.shape)).astype(bool)
    sns.heatmap(corr_df, mask=mask_ut, cmap="coolwarm")
    plt.show()


def supervised_learn(dataf: object) -> object:
    # Split the data into a training and test set.
    X = dataf.iloc[:, 0: 30].values
    y = dataf.iloc[:, 30].values
    X = X.reshape(-1, 30)
    X = preprocessing.scale(X)  # scale the data so that it is easier to fit

    # Split the testing and training data by 75% and 25%
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=0)

    # Use sklearn’s logistic regression function to fit a model
    log_reg = LogisticRegression()
    log_reg.fit(X_train, y_train)

    # make predictions on test data
    y_pred = log_reg.predict(X_test).reshape(-1, 1)

    # Use the Class encoders as the classes
    classes = dataf['Class'].unique()

    # Make the confusion matrix
    conf_mat = confusion_matrix(y_test, y_pred)
    # Make a dataframe for the confusion matrix
    cm_df = pd.DataFrame(conf_mat, columns=classes, index=classes)
    print(cm_df)
    print("")

    #  Make the confusion array a 1D array that contains the input elements.
    # Assign the elements to respective variables
    TP, FN, FP, TN = conf_mat.ravel()
    print(f'''TP = {TP}
    FN = {FN}
    FP = {FP}
    TN = {TN}''')
    print("")

    # Calculate Accuracy
    accuracy = round((TP + TN) / (TP + FN + TN + FP),3)

    # Calculate Precision
    precision = round(TP / (TP + FP),3)

    # Calculate Recall
    recall = round(TP / (TP + FN),3)

    print(f'''Accuracy = {accuracy}
Precision = {precision}
Recall = {recall}''')

# Import the data to a dataframe
df = pd.read_csv("creditcard.csv")

describe_data(df)
# From the data description in this case it can be seen that the data provided had no missing value or objects
# in the data.
plot_histograms(df)
# The histogram plots shows that the variables are normally distributed
correlation(df)
# The generated heatmap shows that there is no correlation seen between any variable this could prove challenging
# for Principal Component Analysis. All variable only seem to have a correlation with the class variable
supervised_learn(df)

# Supervised machine learning was used in this project due to having an outcome (shown in the Class column)
# The confusion matrix generated shows that majority of cases the machine was able to properly predict a non-fraudulent
# transaction. However, there are larger mistakes seen with fraudulent transactions (coded as 1). This could be a result
# of the lower amount of fraudulent transaction data the machine has available to use in the project.

# With regard to the Accuracy the model has a 0.999 accuracy which shows the model will be able to predict the right
# answer most of the time
# For the precision a 0.999 result is seen which says that the model will correctly predict positive instances
# The Recall result of 1.0 shows that the model can identify positive instances very well.

# However in this case the amount of non-fraudulent transactions in the data is significantly more than the fraudulent
# transactions which is skewing the result of the model. Either the amount of non-fraudulent transactions need to be
# scaled down or more fraudulent transactions need to be added

####################################################################################################################
# Make a new dataframe from the previous one containing equal amount of fraudulent and non-fraudulent transactions
# Split the original dataframe into two separate dataframes - one for fraudulent transactions
# and one for non-fraudulent transactions
fraudulent_df = df[df['Class'] == 1]
non_fraudulent_df = df[df['Class'] == 0]

# Reduce amount of non-fraudulent transactions
non_fraudulent_sample = non_fraudulent_df.sample(n=120, random_state=42)
# Concatenate the two dataframes into a new dataframe with equal numbers of fraudulent and non-fraudulent transactions
new_df = pd.concat([fraudulent_df, non_fraudulent_sample])

# Retry the supervised machine learning model
supervised_learn(new_df)

# As can be seen when the amount of each data type was equalised there was a reduction in model accuracy, recall
# and precision. However, the model can still be seen to be quite relaible in the predictions it can make.