#Importing relevent libraries
import pandas as pd
import numpy as np

from sklearn.linear_model import LinearRegression

#Opening data
RAW_DATA = pd.read_csv("/Users/ellie/combinedhouse.csv", sep=',')
print("Shape of data: ", RAW_DATA.shape)
RAW_DATA.head()

#One hot encoding for categorical variables
DUMMIE_HOUSE_DATA = pd.get_dummies(data=RAW_DATA, columns=['MSZoning', 'Street', 'Alley',
                                                           'LotShape', 'LandContour',
                                                           'Utilities', 'LotConfig',
                                                           'LandSlope', 'Neighborhood',
                                                           'Condition1', 'Condition2',
                                                           'BldgType', 'HouseStyle',
                                                           'RoofStyle', 'RoofMatl',
                                                           'Exterior1st', 'Exterior2nd',
                                                           'MasVnrType', 'ExterQual',
                                                           'ExterCond', 'Foundation',
                                                           'BsmtQual', 'BsmtCond',
                                                           'BsmtExposure', 'BsmtFinType1',
                                                           'BsmtFinType2', 'Heating',
                                                           'HeatingQC', 'CentralAir',
                                                           'Electrical', 'KitchenQual',
                                                           'Functional', 'FireplaceQu',
                                                           'GarageType', 'GarageFinish',
                                                           'GarageQual', 'GarageCond',
                                                           'PavedDrive', 'PoolQC',
                                                           'Fence', 'MiscFeature',
                                                           'SaleType', 'SaleCondition'],
                                   drop_first=True)
print("Shape of data: ", DUMMIE_HOUSE_DATA.shape)
DUMMIE_HOUSE_DATA.head()

#Splitting the data into training and testing
#Replacing NaN in the training data with 0
N_TRAIN = 1460
HOUSE_DATA = DUMMIE_HOUSE_DATA[:N_TRAIN]
HOUSE_PREDICT = DUMMIE_HOUSE_DATA[N_TRAIN:]
HOUSE_DATA.fillna(0, inplace=True)

print("Missing values for house_data:", np.any(HOUSE_DATA.isnull().values))
print("Missing values for the test:", np.any(HOUSE_PREDICT.isnull().values))

#Assigning target to be predicted and features
#Again filling in NaN with 0 for features
Y_HOUSE = HOUSE_DATA['SalePrice']
X_HOUSE = HOUSE_DATA.drop(['Id', 'SalePrice'], axis=1)
X_PREDICT_HOUSE = HOUSE_PREDICT.drop(['Id', 'SalePrice'], axis=1)
X_PREDICT_HOUSE.fillna(0, inplace=True)
X_HOUSE.head()

#Impoting the linear regressino model
#Fitting it to the training data
LR_SK = LinearRegression().fit(X_HOUSE, Y_HOUSE)
print("Intercept =", LR_SK.intercept_)
print("R^2 =", LR_SK.score(X_HOUSE, Y_HOUSE))

#Predicting the sale prices of the testing data
#Creating a csv file with the predictions to submit to kaggle
Y_PRECIT_HOUSE = LR_SK.predict(X_PREDICT_HOUSE)
#np.savetxt("HousePredictionPrices4.csv", y_predict_house, delimiter=',')
