import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

dataset = pd.read_csv('MYCShopData.csv')

# Print dataset MYC
# print(dataset)

# Check missing data

# dataset.info()

# for i in range(len(dataset.columns)):
#     missingData = dataset[dataset.columns[i]].isna().sum()
#     missingPercent = missingData / len(dataset) * 100
# print(f'Column {i}: has {missingPercent}% missing dataset')

# from sklearn.impute import SimpleImputer

# convert the dataframe into a numpy array by calling values on my dataframe (not necessary), but a habit I prefer
# X= dataset.iloc[:, :-1].values
# print("X: ",X)
# Y = dataset.iloc[:, -1].values
# print("Y: ",Y)


# Create an instance of Class SimpleImputer: np.nan is the empty value in the dataset
# imputer = SimpleImputer(missing_values=np.nan, strategy='mean')

# Replace missing value from numerical Col 1 'Age', Col 2 'Salary'
# fit on the dataset to calculate the statistic for each column
# x = dataset.iloc[:, :-1].values
# imputer.fit(x[:, 1:7])

# The fit imputer is then applied to the dataset
# to create a copy of the dataset with all the missing values
# for each column replaced with the calculated mean statistic.
# transform will replace & return the new updated columns
# x[:, 1:7] = imputer.transform(x[:, 1:7])

# print("Data Imputation: ", x)


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

# X = dataset.iloc[:, :-1].values
# Y = dataset.iloc[:, -1].values

# ct = ColumnTransformer(
#     transformers=[('encoder', OneHotEncoder(), [0])], remainder="passthrough")
# X = ct.fit_transform(X)

# lbl = LabelEncoder()
# Y = lbl.fit_transform(Y)

# xTrain, xTest, yTrain, yTest = train_test_split(
#     X, Y, test_size=0.2, random_state=1)
# print("xTrain: ", xTrain)
# print("xTest: ", xTest)
# print("yTrain: ", yTrain)
# print("yTest: ", yTest)


# Feature Scaling

# from sklearn.preprocessing import StandardScaler
# sc = StandardScaler()
# X_train[:,3:] = sc.fit_transform(X_train[:,3:])
# #only use Transform to use the SAME scaler as the Training Set
# X_test[:,3:] = sc.transform(X_test[:,3:])
# print(X_train)

# Feature Scaling


# X = dataset.iloc[:, :-1].values
# Y = dataset.iloc[:, -1].values

# ct = ColumnTransformer(
#     transformers=[('encoder', OneHotEncoder(), [0])], remainder="passthrough")
# X = ct.fit_transform(X)

# lbl = LabelEncoder()
# Y = lbl.fit_transform(Y)

# xTrain, xTest, yTrain, yTest = train_test_split(
#     X, Y, test_size=0.2, random_state=1)

# sc = StandardScaler(with_mean=False)
# xTrain[:, 3:] = sc.fit_transform(xTrain[:, 3:])
# xTest[:, 3:] = sc.transform(xTest[:, 3:])
# print("xTrain: ", xTrain)

# print("xTest: ", xTest)

# MYC - GROUP1 - BUSINESS INTELLIGENCE - 2022

# References:

# https://github.com/CodexploreRepo/data-science/blob/main/Code/P01_Pre_Processing/data_preprocessing_template.ipynb
# https://www.youtube.com/watch?v=VsXKtjddXWY&t=937s&ab_channel=CodeXplore