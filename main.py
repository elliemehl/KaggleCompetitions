import pandas as pd
import numpy as np
import argparse
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LinearRegression


def parse_arguments():
    parser = argparse.ArgumentParser(description='Enter inputs for the file you want to execute')
    parser.add_argument('problem', choices=['Housing', 'Titanic'], help='The problem to be solved.')
    parser.add_argument('input_file', type=str, help='Path to input file')
    parser.add_argument('test_size', type=float, help='Size of the testing file')
    parser.add_argument('output_file', type=str, help='Path to output file')
    args = parser.parse_args()
    return args


class HousePredictions:

    def __init__(self, dataset):
        self._data_frame = dataset
        self._linear_reg = LinearRegression()

    def clean(self):
        raw_data = pd.read_csv(self._data_frame, sep=',')
        house_dropped_data = raw_data.drop(['Alley', 'PoolQC', 'Fence', 'MiscFeature'], axis=1)
        cols = ['MasVnrArea', 'LotFrontage', 'BsmtFinSF1', 'BsmtUnfSF',
                'TotalBsmtSF', 'BsmtFullBath', 'BsmtHalfBath', 'GarageYrBlt',
                'GarageCars', 'GarageArea', 'BsmtFinSF2']
        house_dropped_data[cols] = house_dropped_data[cols].fillna(0)
        self._clean_data = pd.get_dummies(data=house_dropped_data,
                                         columns=['MSZoning', 'Street',
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
                                                'PavedDrive', 'SaleType',
                                                'SaleCondition'], drop_first=True)

    def split(self, test_size):
        self.X_train = np.array(self._clean_data.drop(['SalePrice', 'Id'], axis=1))
        self.y_train = np.array(self._clean_data['SalePrice'])
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X_train, self.y_train,
                                                                                test_size=test_size, shuffle=False)

    def fit(self):
        self.model = self._linear_reg.fit(self.X_train, self.y_train)

    def predict(self):
        result = self._linear_reg.predict(self.X_test)
        np.savetxt(args.output_file, result, delimiter=',',
                   header= 'SalePrice')


class TitanicPredictions:

    def __init__(self, dataset):
        self._data_frame = dataset
        self._decision_tree = DecisionTreeClassifier()

    def clean(self):
        titanic_raw = pd.read_csv(self._data_frame, sep=',')
        titanic_raw['Age'] = titanic_raw['Age'].fillna(titanic_raw['Age'].mean())
        titanic_raw['Fare'] = titanic_raw['Fare'].fillna(titanic_raw['Fare'].median())
        titanic_raw['Embarked'] = titanic_raw['Embarked'].fillna('S')

        titanic_drop = titanic_raw.drop(['PassengerId', 'SibSp', 'Name',
                                         'Parch', 'Ticket', 'Cabin'], axis=1)
        self._clean_data = pd.get_dummies(data=titanic_drop,
                                         columns=['Embarked', 'Sex'], drop_first=True)

    def split(self, test_size):
        self.X_train = np.array(self._clean_data.drop(['Survived'], axis=1))
        self.y_train = np.array(self._clean_data['Survived'])
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X_train, self.y_train, test_size=test_size,
                                                                            shuffle=False)

    def fit(self):
        self.model = self._decision_tree.fit(self.X_train, self.y_train)

    def predict(self):
        result = self._decision_tree.predict(self.X_test)
        np.savetxt(args.output_file, result, delimiter=',',
                   header='Survived')


if __name__ == '__main__':
    args = parse_arguments()
    if args.problem == "Housing":
        model_instance = HousePredictions(args.input_file)
    elif args.problem == "Titanic":
        model_instance = TitanicPredictions(args.input_file)
    else:
        raise ValueError("Invalid argument provided.")
    model_instance.clean()
    model_instance.split(args.test_size)
    model_instance.fit()
    model_instance.predict()
