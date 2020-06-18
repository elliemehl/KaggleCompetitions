#Importing relevent libraries
import pandas as pd

from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier

#Loading the data
#Filling blank spaces
#Dropping features
TITANIC_RAW_DATA = pd.read_csv("/Users/ellie/combinedtitanic.csv", sep=',')
TITANIC_RAW_DATA['Age'] = TITANIC_RAW_DATA['Age'].fillna(TITANIC_RAW_DATA['Age'].mean())
TITANIC_RAW_DATA['Fare'] = TITANIC_RAW_DATA['Fare'].fillna(TITANIC_RAW_DATA['Fare'].median())
TITANIC_RAW_DATA['Embarked'] = TITANIC_RAW_DATA['Embarked'].fillna('S')

TITANIC_DROPPED_DATA = TITANIC_RAW_DATA.drop(['PassengerId', 'SibSp', 'Name',
                                              'Parch', 'Ticket', 'Cabin'], axis=1)

#Splitting data into training and testing
N_TRAIN = 891
TITANIC_DATA = TITANIC_DROPPED_DATA[:N_TRAIN]
TITANIC_PREDICT = TITANIC_DROPPED_DATA[N_TRAIN:]

#Assigning target to be predicted and features of training data
Y_TITANIC = TITANIC_DATA['Survived']
INPUTS_TITANIC = TITANIC_DATA.drop(['Survived'], axis=1)

DUMMIES_TITANIC = pd.get_dummies(data=INPUTS_TITANIC, columns=['Embarked'], drop_first=True)
LE_FEATURES = LabelEncoder()
DUMMIES_TITANIC['Sex_n'] = LE_FEATURES.fit_transform(DUMMIES_TITANIC['Sex'])
X_TITANIC = DUMMIES_TITANIC.drop(['Sex'], axis=1)

#Assigning target to be predicted and features of testing data
DUMMIES_PREDICT = pd.get_dummies(data=TITANIC_PREDICT, columns=['Embarked'], drop_first=True)
LE_FEATURES_PREDICT = LabelEncoder()
DUMMIES_PREDICT['Sex_n'] = LE_FEATURES_PREDICT.fit_transform(DUMMIES_PREDICT['Sex'])
X_PREDICT = DUMMIES_PREDICT.drop(['Sex', 'Survived'], axis=1)

#Fitting the model
MODEL = DecisionTreeClassifier()
MODEL.fit(X_TITANIC, Y_TITANIC)

#Predicting values
Y_PREDICT_DT = MODEL.predict(X_PREDICT)
ACC_DT = round(MODEL.score(X_TITANIC, Y_TITANIC) * 100, 2)
print(ACC_DT)

#Creating the excel file to submit
SUBMISSION = pd.DataFrame({
        "Survived":Y_PREDICT_DT
    })

SUBMISSION.to_csv('titanic_survival3.csv', index=False)
