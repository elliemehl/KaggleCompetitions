from Decision import data_file, save_file

def clean_data():
    titanic_raw = pd.read_csv(data_file, sep=',')
    titanic_raw['Age'] = titanic_raw['Age'].fillna(titanic_raw['Age'].mean())
    titanic_raw['Fare'] = titanic_raw['Fare'].fillna(titanic_raw['Fare'].median())
    titanic_raw['Embarked'] = titanic_raw['Embarked'].fillna('S')

    titanic_drop = titanic_raw.drop(['PassengerId', 'SibSp', 'Name',
                                     'Parch', 'Ticket', 'Cabin'], axis=1)
    titanic_cleaned = pd.get_dummies(data=titanic_drop,
                                         columns=['Embarked', 'Sex'], drop_first=True)
    titanic_cleaned.to_csv('TitanicPredictionsData.csv', index=False)
clean_data()

class TitanicPredictions:

    def __init__(self, dataset='TitanicPredictionsData.csv'):
        self.data_frame = pd.read_csv(dataset)
        self.decision_tree = DecisionTreeClassifier()


    def split(self, test_size):
        X = np.array(self.data_frame.drop(['Survived'], axis=1))
        y = np.array(self.data_frame['Survived'])
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X, y, test_size=test_size,
                                                                            shuffle=False)

    def fit(self):
        self.model = self.decision_tree.fit(self.X_train, self.y_train)

    def predict(self):
        result = self.decision_tree.predict(self.X_test)
        return result

if __name__ == '__main__':
    model_instance = TitanicPredictions()
    model_instance.split(0.3193)
    model_instance.fit()
    np.savetxt(save_file, model_instance.predict(), delimiter=',',
                   header='Survived')
