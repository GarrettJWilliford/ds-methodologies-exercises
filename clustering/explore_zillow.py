import pydataset
from scipy import stats
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LinearRegression

from sklearn.ensemble import RandomForestClassifier





data = pydataset.data('faithful')
print(data)
print(stats.pearsonr(data['eruptions'],  data['waiting']))


ax = sns.scatterplot(x="eruptions", y="waiting",
                      data=data)

#plt.show()

print(data.info())

y = data[['eruptions']]
x = data[['waiting']]

print(x)



x_train, x_test, y_train, y_test = train_test_split(x, y, random_state = 123)

print(len(x_train))
print(len(y_train))

logit = LinearRegression().fit(x_train, y_train)


print('!!')
y_pred = logit.predict(x_train)
print('!!')
#y_pred_proba = logit.predict_proba(x_train)
#print(confusion_matrix(y_train, y_pred))

new_data = pd.DataFrame()
new_data['waiting'] = x_train['waiting']
new_data['eruption_actual'] = data['eruptions']
#new_data['eruptions_predicted']= pd.DataFrame(y_pred)[0]
print(len(new_data))
print(len(y_pred))



#pd.merge(left=new_data,right=y_pred,on = 'waiting', how='outer')

print(new_data)
ax = sns.scatterplot(x="eruptions_predicted", y = 'eruption_actual',
                      data=new_data)


#plt.show()




import pydataset
from sklearn.cluster import KMeans
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import dbtools as db




def iris_cluster():
    data = pydataset.data('iris')
    data.columns = [c.lower().replace('.', '_') for c in data]
    x = data[['sepal_length', 'petal_length', 'petal_width']]
    kmeans = KMeans(n_clusters = 4)
    kmeans.fit(x)
    centers = pd.DataFrame(kmeans.cluster_centers_, columns=x.columns)
    print(kmeans.cluster_centers_)
    fig = plt.figure(figsize=(12, 9))
    ax = Axes3D(fig)
    ax.scatter(data.sepal_length, data.petal_length, data.petal_width, c=kmeans.labels_)
    ax.scatter(centers.sepal_length, centers.petal_length, centers.petal_width, c='pink', s=10000, alpha=.4)
    ax.set(xlabel='sepal_length', ylabel='petal_length', zlabel='petal_width')
    plt.show()




def mall_cluster():
    data = db.get_db_url(comm = 'SELECT * FROM customers', database = 'mall_customers')
    data['gender'] = data['gender'].apply(lambda x: 0 if x == 'Female' else 1)
    print(data)
    x = data[['annual_income',  'spending_score']]
    kmeans = KMeans(n_clusters = 5)
    kmeans.fit(x)
    print(KMeans(n_cluster = 5).fit(data[['age']]).cluster_centers_)
    print(kmeans.cluster_centers_)
    fig = plt.figure(figsize=(12, 9))
    ax = Axes3D(fig)
    centers = pd.DataFrame(kmeans.cluster_centers_, columns=x.columns)

    ax.scatter(data.age, data.annual_income, data.spending_score, c=kmeans.labels_)
    ax.scatter(centers.age, centers.annual_income, centers.spending_score, c='pink', s=10000, alpha=.4)
    ax.set(xlabel='age', ylabel='annual_income', zlabel='spending_score')
    plt.show()
    


def tips_cluster():
    data = pydataset.data('tips')
    print(data.info())
    x = data[['total_bill','tip', 'size']]
    kmeans = KMeans(n_clusters = 3)
    kmeans.fit(x)
    print(kmeans.cluster_centers_)
    centers = pd.DataFrame(kmeans.cluster_centers_, columns=x.columns)
    fig = plt.figure(figsize=(12, 9))
    ax = Axes3D(fig)
    ax.scatter(data.total_bill, data.tip, data.size, c=kmeans.labels_)
    ax.scatter(centers.total_bill, centers.tip, centers.size, c='pink', s=10000, alpha=.4)
    ax.set(xlabel='total_bill', ylabel='tip', zlabel='size')
    plt.show()
    

    #print(data)


def iris_cluster_two():
    data = pydataset.data('iris')
    data.columns = [c.lower().replace('.', '_') for c in data]
    x = data[['sepal_length', 'petal_length', 'petal_width']]
    kmeans = DBSCAN(eps = 3, min_samples = 2)
    kmeans.fit(x)
    centers = pd.DataFrame(kmeans.cluster_centers_, columns=x.columns)
    print(kmeans.cluster_centers_)
    fig = plt.figure(figsize=(12, 9))
    ax = Axes3D(fig)
    ax.scatter(data.sepal_length, data.petal_length, data.petal_width, c=kmeans.labels_)
    ax.scatter(centers.sepal_length, centers.petal_length, centers.petal_width, c='pink', s=10000, alpha=.4)
    ax.set(xlabel='sepal_length', ylabel='petal_length', zlabel='petal_width')
    plt.show()




def mall_cluster_two():
    data = db.get_db_url(comm = 'SELECT * FROM customers', database = 'mall_customers')
    data['gender'] = data['gender'].apply(lambda x: 0 if x == 'Female' else 1)
    print(data)
    x = data[['annual_income',  'spending_score']]
    kmeans = KMeans(n_clusters = 5)
    kmeans.fit(x)
    print(KMeans(n_cluster = 5).fit(data[['age']]).cluster_centers_)
    print(kmeans.cluster_centers_)
    fig = plt.figure(figsize=(12, 9))
    ax = Axes3D(fig)
    centers = pd.DataFrame(kmeans.cluster_centers_, columns=x.columns)

    ax.scatter(data.age, data.annual_income, data.spending_score, c=kmeans.labels_)
    ax.scatter(centers.age, centers.annual_income, centers.spending_score, c='pink', s=10000, alpha=.4)
    ax.set(xlabel='age', ylabel='annual_income', zlabel='spending_score')
    plt.show()
    


def tips_cluster_two():
    data = pydataset.data('tips')
    print(data.info())
    x = data[['total_bill','tip', 'size']]
    kmeans = KMeans(n_clusters = 3)
    kmeans.fit(x)
    print(kmeans.cluster_centers_)
    centers = pd.DataFrame(kmeans.cluster_centers_, columns=x.columns)
    fig = plt.figure(figsize=(12, 9))
    ax = Axes3D(fig)
    ax.scatter(data.total_bill, data.tip, data.size, c=kmeans.labels_)
    ax.scatter(centers.total_bill, centers.tip, centers.size, c='pink', s=10000, alpha=.4)
    ax.set(xlabel='total_bill', ylabel='tip', zlabel='size')
    plt.show()
    



iris_cluster_two()





