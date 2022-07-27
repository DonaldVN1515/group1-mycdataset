# https://github.com/CodexploreRepo/data-science/blob/main/Code/P01_Pre_Processing/data_preprocessing_template.ipynb
# https://www.youtube.com/watch?v=VsXKtjddXWY&t=937s&ab_channel=CodeXplore

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

dataset = pd.read_csv('MYCShopData.csv')

print(dataset)
# print(X)
# print(y)

# Data Imputation (Missing Data Replacement)

# for i in range(len(dataset.columns)):
#     missing_data = dataset[dataset.columns[i]].isna().sum()
#     perc = missing_data / len(dataset) * 100
#     print('>%d,  missing entries: %d, percentage %.2f' % (i, missing_data, perc))
    
    
    
# plt.figure(figsize = (4,4)) #is to create a figure object with a given size
# sns.heatmap(dataset.isna(), cbar=False, cmap='viridis', yticklabels=False)


# #convert the dataframe into a numpy array by calling values on my dataframe (not necessary), but a habit I prefer
# X= dataset.iloc[:, :-1].values
# y = dataset.iloc[:, -1].values


# from sklearn.impute import SimpleImputer

# #Create an instance of Class SimpleImputer: np.nan is the empty value in the dataset
# imputer = SimpleImputer(missing_values=np.nan, strategy='mean')

# #Replace missing value from numerical Col 1 'Age', Col 2 'Salary'
# #fit on the dataset to calculate the statistic for each column
# imputer.fit(X[:, 1:3]) 

# #The fit imputer is then applied to the dataset 
# # to create a copy of the dataset with all the missing values 
# # for each column replaced with the calculated mean statistic.
# #transform will replace & return the new updated columns
# X[:, 1:3] = imputer.transform(X[:, 1:3])

# print(X)


# Encode Categorical Data

# Encode Independent variable (X)
# from sklearn.compose import ColumnTransformer
# from sklearn.preprocessing import OneHotEncoder
# #transformers: specify what kind of transformation, and which cols
# #Tuple ('encoder' encoding transformation, instance of Class OneHotEncoder, [col to transform])
# #remainder ="passthrough" > to keep the cols which not be transformed. Otherwise, the remaining cols will not be included 
# ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), [0])] , remainder="passthrough" )
# #fit and transform with input = X
# #np.array: need to convert output of fit_transform() from matrix to np.array
# X = np.array(ct.fit_transform(X))
# print(X)

# Encode Dependent Variable
# from sklearn.preprocessing import LabelEncoder
# le = LabelEncoder()
# #output of fit_transform of Label Encoder is already a Numpy Array
# y = le.fit_transform(y)
# print(y)



# Splitting the dataset (X = data, y = output) into the Training set and Test set

# from sklearn.model_selection import train_test_split
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 1)
# print(X_train)
# print(X_test)
# print(y_train)
# print(y_test)


# Feature Scaling

# from sklearn.preprocessing import StandardScaler
# sc = StandardScaler()
# X_train[:,3:] = sc.fit_transform(X_train[:,3:])
# #only use Transform to use the SAME scaler as the Training Set
# X_test[:,3:] = sc.transform(X_test[:,3:])
# print(X_train)

# print(X_test)


# 5. Training Machine Learning Model

## Models from Scikit-Learn: Search "scikit learn model map"
# from sklearn.linear_model import LogisticRegression
# logistic_clf = LogisticRegression()
# logistic_clf.fit(X_train, y_train)

# 5.1. Evaluate the model

# # Evaluate the model on the training set
# logistic_clf.score(X_train, y_train)

# # Evaluate the model on the test set
# logistic_clf.score(X_test, y_test)

# y_preds = logistic_clf.predict(X_test)
# y_preds

# y_test

# #Predict with a single input
# logistic_clf.predict([[0.0, 0.0, 1.0, -0.19159184384578545, -1.0781259408412425]])