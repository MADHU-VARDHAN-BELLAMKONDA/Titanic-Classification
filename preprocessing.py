import pandas as pd
from sklearn.preprocessing import LabelEncoder

def preprocess_data(train_data):
    train_data['Age'] = train_data['Age'].fillna(train_data['Age'].median())
    train_data['Embarked'] = train_data['Embarked'].fillna(train_data['Embarked'].mode()[0])
    train_data['Fare'] = train_data['Fare'].fillna(train_data['Fare'].median())
    train_data = train_data.drop('Cabin', axis=1)
    labelencoder = LabelEncoder()
    train_data['Sex'] = labelencoder.fit_transform(train_data['Sex'])
    train_data['Embarked'] = labelencoder.fit_transform(train_data['Embarked'])
    train_data['FamilySize'] = train_data['SibSp'] + train_data['Parch'] + 1
    train_data['IsAlone'] = 1
    train_data.loc[train_data['FamilySize'] > 1, 'IsAlone'] = 0
    return train_data
