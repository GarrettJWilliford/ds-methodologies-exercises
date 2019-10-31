import pandas as pd
import dbtools as db
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import numpy as np
import dbtools as db
import warnings


#Stratify

warnings.filterwarnings('ignore')


def get_titanic_data(command = 'SELECT * FROM passengers', database = 'titanic_db'):
    return db.get_db_url(comm = command, database = database)


def get_iris_data(command = """SELECT measurement_id,  sepal_length,  sepal_width
,petal_length, petal_width,  species.species_name FROM measurements
JOIN species USING(species_id);""", database = 'iris_db'):
    return db.get_db_url(comm = command, database = database)


def prep_iris(iris):
    iris.drop(columns = ['measurement_id'], inplace = True)
    iris.rename(columns={"species_name": "species"}, inplace = True)
    label_encoder = LabelEncoder()
    iris['species'] = label_encoder.fit_transform(iris['species'])
    return iris


def prep_titanic(data):
    label_encoder = LabelEncoder()
    scaler = MinMaxScaler()
    imputer = SimpleImputer(strategy = 'constant')
    data.drop(columns = ['deck'], inplace = True)
    data['embarked'] = data['embarked'].dropna()
    data['embark_town'] = imputer.fit_transform(data[['embark_town']])
    #data['age'] = imputer.fit_transform(data[['age']])
    data['embarked'] = label_encoder.fit_transform(imputer.fit_transform(data[['embarked']]))
    print(scaler.fit(data[['age', 'fare']]))
    return data



titanic = get_titanic_data()
titanic = prep_titanic(titanic)
print(titanic)



a = input('')
error_me = me_error
