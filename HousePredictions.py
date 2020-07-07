from Decision import data_file, save_file

def clean_data():
    raw_data = pd.read_csv(data_file, sep=',')
    house_dropped_data = raw_data.drop(['Alley', 'PoolQC', 'Fence', 'MiscFeature'], axis=1)
    cols = ['MasVnrArea', 'LotFrontage', 'BsmtFinSF1', 'BsmtUnfSF',
            'TotalBsmtSF', 'BsmtFullBath', 'BsmtHalfBath', 'GarageYrBlt',
            'GarageCars', 'GarageArea', 'BsmtFinSF2']
    house_dropped_data[cols] = house_dropped_data[cols].fillna(0)

    dummie_house_data = pd.get_dummies(data=house_dropped_data,
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

    dummie_house_data.to_csv('HousePredictionData.csv', index=False)


clean_data()


class HousePredictions:

    def __init__(self, dataset='HousePredictionData.csv'):
        self.data_frame = pd.read_csv(dataset)
        self.linear_reg = LinearRegression()

    def split(self, test_size):
        X = np.array(self.data_frame.drop(['SalePrice', 'Id'], axis=1))
        y = np.array(self.data_frame['SalePrice'])
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X, y, test_size=test_size,
                                                                                shuffle=False)

    def fit(self):
        self.model = self.linear_reg.fit(self.X_train, self.y_train)

    def predict(self):
        result = self.linear_reg.predict(self.X_test)
        return result


if __name__ == '__main__':
    model_instance = HousePredictions()
    model_instance.split(0.4995)
    model_instance.fit()
    np.savetxt(save_file, model_instance.predict(), delimiter=',',
               header='SalePrice')
