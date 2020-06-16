#Importing relevent libraries
import pandas as pd
import numpy as np
import sklearn as sk

#Opening data
raw_house_data = pd.read_csv("/Users/ellie/combinedhouse.csv", sep = ',')
print("Shape of data: ", raw_house_data.shape)
raw_house_data.head()

#One hot encoding for categorical variables
dummie_house_data = pd.get_dummies(data=raw_house_data, columns = ['MSZoning', 'Street', 'Alley', 'LotShape', 'LandContour', 'Utilities', 'LotConfig', 'LandSlope', 'Neighborhood', 'Condition1', 'Condition2', 'BldgType', 'HouseStyle', 'RoofStyle', 'RoofMatl', 'Exterior1st', 'Exterior2nd', 'MasVnrType', 'ExterQual', 'ExterCond', 'Foundation', 'BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2', 'Heating', 'HeatingQC', 'CentralAir', 'Electrical', 'KitchenQual', 'Functional', 'FireplaceQu', 'GarageType', 'GarageFinish', 'GarageQual', 'GarageCond', 'PavedDrive', 'PoolQC', 'Fence', 'MiscFeature', 'SaleType', 'SaleCondition'], drop_first=True)
print("Shape of data: ", dummie_house_data.shape)
dummie_house_data.head()

#Splitting the data into training and testing
#Replacing NaN in the training data with 0
N_train = 1460
house_data = dummie_house_data[:N_train]
house_predict = dummie_house_data[N_train:]
house_data.fillna(0, inplace=True)

print("Missing values for house_data:", np.any(house_data.isnull().values))
print("Missing values for the test:", np.any(house_predict.isnull().values))

#Assigning target to be predicted and the features to base prediction off of
#Again filling in NaN with 0 for features
y_house = house_data['SalePrice']
X_house = house_data.drop(['Id', 'SalePrice'], axis=1)
X_predict_house = house_predict.drop(['Id', 'SalePrice'], axis=1)
X_predict_house.fillna(0, inplace=True)
X_house.head()

#Impoting the linear regression model
#Fitting it to the training data
from sklearn.linear_model import LinearRegression
osl_sk = LinearRegression().fit(X_house, y_house)

print("Intercept =", osl_sk.intercept_)
##print("Model coefficients =", osl_sk.coef_)
print("R^2 =", osl_sk.score(X_house, y_house))

#Predicting the sale prices of the testing data
#Creating a csv file with the predictions to submit to kaggle
y_predict_house = osl_sk.predict(X_predict_house)
np.savetxt("HousePredictionPrices4.csv", y_predict_house, delimiter=',')
