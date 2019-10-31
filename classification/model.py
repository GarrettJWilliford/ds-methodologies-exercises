from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix
from sklearn.impute import SimpleImputer
from sklearn.tree import export_graphviz
from graphviz import Graph
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import warnings
import dbtools as db
import graphviz
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import MinMaxScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import export_graphviz
from graphviz import Graph
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import warnings
import os
import dbtools as db
import pandas as pd
warnings.filterwarnings("ignore")




def get_titanic_data(command = 'SELECT * FROM passengers', database = 'titanic_db'):
    return db.get_db_url(comm = command, database = database)

def prep_titanic(data):
    label_encoder = LabelEncoder()
    scaler = MinMaxScaler()
    imputer = SimpleImputer(missing_values = np.nan, strategy = 'constant')
    data.drop(columns = ['deck'], inplace = True)
    data['embarked'] = data['embarked'].dropna()
    data['embark_town'] = imputer.fit_transform(data[['embark_town']])
    #data['age'] = imputer.fit_transform(data[['age']])
    data['embarked'] = label_encoder.fit_transform(imputer.fit_transform(data[['embarked']]))
    print(scaler.fit(data[['age', 'fare']]))
    return data


data = prep_titanic(get_titanic_data())
data.dropna(inplace = True)
print(data)


X = data[['pclass','age','fare','sibsp','parch']]
y = data[['survived']]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = .30, random_state = 123)

print(X_train.head())

logit = LogisticRegression(C=1, class_weight={1:2}, random_state = 123, solver='saga').fit(X_train, y_train)


print('Coefficient: \n', logit.coef_)
print('Intercept: \n', logit.intercept_)

y_pred = logit.predict(X_train)
y_pred_proba = logit.predict_proba(X_train)

print('Accuracy of Logistic Regression classifier on training set: {:.2f}'
     .format(logit.score(X_train, y_train)))



print(confusion_matrix(y_train, y_pred))

print(classification_report(y_train, y_pred))
dot_data = export_graphviz(machine_learn, out_file=None) 
graph = graphviz.Source(dot_data) 

graph.render('iris_decision_tree', view=True)

def results_train(logit, y_pred, y_pred_proba, x_train, y_train):
    print('\n\n<<<<<<<<<<<<<<<<|RESULTS|>>>>>>>>>>>>>>>\n')
    try:
        print('Accuracy of Logistic Regression classifier on training set: {:.2f} \n'
              .format(logit.score(x_train, y_train)))
    except:
        pass
    try:
        print('Accuracy of Logistic Regression classifier on test set: {:.2f} \n'
              .format(logit.score(x_test, y_test)))
    except:
        pass
    try:
        print('Coefficient: \n', logit.coef_)
        print('Intercept: \n', logit.intercept_)
    except:
        pass
    print('')
    print('-----------|CONFUSION_MATRIX|------------')
    try:
        print(confusion_matrix(y_train, y_pred))
    except:
        print('<<|UNKOWN|>>')
    print('-----------------|REPORT|-----------------')
    try:
        print(classification_report(y_train, y_pred))
    except:
        print('<<|UNKNOWN|>>')
    print('----------------------------------------')



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




machine_learn = DecisionTreeClassifier(criterion = 'gini', max_depth = 3, \
                                       random_state = 123)


data = prep_iris(get_iris_data())

train = data.drop(columns = ['species'])
test = data['species']


x_train, x_test, y_train, y_test = train_test_split(train, test, test_size = .3, random_state = 123)


machine_learn.fit(x_train, y_train)



y_pred = machine_learn.predict(x_train)
y_pred_proba = machine_learn.predict_proba(x_train)


final = pd.DataFrame()
print('!!!!')
final['species'] = y_train
final['y_pred'] = y_pred

print(final)

results_train(machine_learn, y_pred, y_pred_proba, x_train, y_train)



dot_data = export_graphviz(machine_learn, out_file=None) 
graph = graphviz.Source(dot_data) 

graph.render('iris_decision_tree', view=True)


def get_titanic_data(command = 'SELECT * FROM passengers', database = 'titanic_db'):
    return db.get_db_url(comm = command, database = database)

def prep_titanic(data):
    label_encoder = LabelEncoder()
    scaler = MinMaxScaler()
    imputer = SimpleImputer(missing_values = np.nan, strategy = 'constant')
    data.drop(columns = ['deck'], inplace = True)
    data['embarked'] = data['embarked'].dropna()
    data['embark_town'] = imputer.fit_transform(data[['embark_town']])
    #data['age'] = imputer.fit_transform(data[['age']])
    data['class'] = label_encoder.fit_transform(imputer.fit_transform(data[['class']]))
    data['sex'] = label_encoder.fit_transform(imputer.fit_transform(data[['sex']]))
    data['embarked'] = label_encoder.fit_transform(imputer.fit_transform(data[['embarked']]))
    print(scaler.fit(data[['age', 'fare']]))
    return data







data = prep_titanic(get_titanic_data())
data = data.drop(columns = ['embark_town', 'fare', 'class'])
data = data.dropna()



train = data.drop(columns = ['survived'])
test =data['survived']


x1, x2, y1, y2 = train_test_split(train, test, test_size = .3, random_state = 123)

forest_learn = RandomForestClassifier(bootstrap=True, class_weight=None, criterion='gini',
            max_depth=3, max_features='auto', max_leaf_nodes=None,
            min_impurity_decrease=0.0, min_impurity_split=None,
            min_samples_leaf=3, min_samples_split=2,
            min_weight_fraction_leaf=0.0, n_estimators=100, n_jobs=None,
            oob_score=False, random_state=123, verbose=0, warm_start=False).fit(x1, y1)


y_pred = forest_learn.predict(x2)
y_pred_proba = forest_learn.predict_proba(x2)



results_train(forest_learn, y_pred, y_pred_proba, x2, y2)
