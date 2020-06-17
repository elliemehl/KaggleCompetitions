#Importing relevent libraries
import pandas as pd

from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier

#Loading the data
#Dropping features 
TITANIC_RAW_DATA = pd.read_csv("/Users/ellie/combinedtitanic.csv", sep=',')
TITANIC_DROPPED_DATA = TITANIC_RAW_DATA.drop(['PassengerId', 'Name', 'SibSp',
                                              'Parch', 'Ticket', 'Cabin',
                                              'Embarked'], axis=1)

#Splitting data into training and testing
N_TRAIN = 891
TITANIC_DATA = TITANIC_DROPPED_DATA[:N_TRAIN]
TITANIC_PREDICT = TITANIC_DROPPED_DATA[N_TRAIN:]
TITANIC_DATA.tail()

#Assigning target to be predicted and features of training data
#Replace age blanks with the mean age
Y_TITANIC = TITANIC_DATA['Survived']
INPUTS_TITANIC = TITANIC_DATA.drop(['Survived'], axis=1)
INPUTS_TITANIC['Age'] = INPUTS_TITANIC['Age'].fillna(INPUTS_TITANIC['Age'].mean())

#Assigning 0 and 1 to sexes
LE_SEX = LabelEncoder()
INPUTS_TITANIC['Sex_n'] = LE_SEX.fit_transform(INPUTS_TITANIC['Sex'])
X_TITANIC = INPUTS_TITANIC.drop(['Sex'], axis=1)
X_TITANIC.tail()

#Assigning target to be predicted and features of testing data
#Replacing the blank ages and fares with mean values
X_TITANIC_PREDICT = TITANIC_PREDICT.drop(['Survived'], axis=1)
TITANIC_PREDICT['Age'] = TITANIC_PREDICT['Age'].fillna(TITANIC_PREDICT['Age'].mean())
TITANIC_PREDICT['Fare'] = TITANIC_PREDICT['Fare'].fillna(TITANIC_PREDICT['Fare'].mean())

#Assigning 0 and 1 to sexes
LE_SEX_PREDICT = LabelEncoder()
TITANIC_PREDICT['Sex_n'] = LE_SEX_PREDICT.fit_transform(TITANIC_PREDICT['Sex'])
X_PREDICT = TITANIC_PREDICT.drop(['Sex', 'Survived'], axis=1)

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

SUBMISSION.to_csv('titanic_survival.csv', index=False)
