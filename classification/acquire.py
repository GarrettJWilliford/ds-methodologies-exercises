import dbtools as db
import pandas as pd





def get_titanic_data(command = 'SELECT * FROM passengers', database = 'titanic_db'):
    return db.get_db_url(comm = command, database = database)


def get_iris_data(command = """SELECT measurement_id,  sepal_length,  sepal_width
,petal_length, petal_width,  species.species_name FROM measurements
JOIN species USING(species_id);""", database = 'iris_db'):
    return db.get_db_url(comm = command, database = database)





a = input('<<<<<>>>>>')

#error_me = me_error


df_iris = pd.DataFrame(db.get_db_url('SELECT * FROM measurements', \
                                     database = 'iris_db'))

print(df_iris.head(3))
print(df_iris.shape)
print(df_iris.columns)
print(df_iris.info())
print(df_iris.describe())
ints = df_iris.select_dtypes(include=['int64', 'float64'])
for i in ints:
    print(i)
    print(str(df_iris[i].max() - df_iris[i].min()))


print('<<<<<>>>>>')

df_excel = pd.DataFrame(pd.read_excel('mytable_customer_details.xlsx'))
df_excel_sample = df_excel.head(100)
print(len(df_excel))
print(df_excel.head(5))
print(df_excel.select_dtypes(include=['object']))

print('<<<<<>>>>>')

df_google = pd.DataFrame(pd.read_excel('train.xlsx'))
print(df_google.head(3))
print(df_google.shape())
print(df_google.columns)
print(df_google.info())
print(df_google.describe())
print(df_google.unique())
print('<<<<<>>>>>')


