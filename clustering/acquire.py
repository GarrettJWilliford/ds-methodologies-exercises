import dbtools as db
import pandas as pd
#import env
import pandas as pd

def get_db_url(comm = '!', database = '!'):
    db=MySQLdb.connect(host='157.230.209.171', user = env.user, \
    passwd = env.password, db=database)
    return psql.read_sql(comm, con=db)



def remove_columns(df, cols_to_remove):  
    df = df.drop(columns=cols_to_remove)
    return df


def handle_missing_values(df, prop_required_column = .5, prop_required_row = .75):
    threshold = int(round(prop_required_column*len(df.index),0))
    df.dropna(axis=1, thresh=threshold, inplace=True)
    threshold = int(round(prop_required_row*len(df.columns),0))
    df.dropna(axis=0, thresh=threshold, inplace=True)
    return df


def data_prep(df, cols_to_remove=[], prop_required_column=.5, prop_required_row=.75):
    df = remove_columns(df, cols_to_remove)
    df = handle_missing_values(df, prop_required_column, prop_required_row)
    return df








def data_percent(data):
    return_data = pd.DataFrame()
    return_data['num_rows_missing'] = data.isnull().sum()
    return_data['pct_rows_missing'] = (data.isnull().sum()) / len(data)
    return return_data




def database_percent(data):
    return_data = pd.DataFrame()
    return_data['num_cols_missing'] = data.isnull().sum(axis = 1)
    return_data['pct_cols_missing'] = (data.isnull().sum(axis = 1)) / len(data)
    return_data['num_rows'] = '!'
    return return_data
    




zillow_data = db.get_db_url( comm = """Select * From properties_2017
Join (SELECT
p_17.parcelid,
logerror,
transactiondate
FROM predictions_2017 p_17
JOIN 
(SELECT
  parcelid, Max(transactiondate) as tdate
FROM
  predictions_2017
 
Group By parcelid )as sq1
ON (sq1.parcelid=p_17.parcelid and sq1.tdate = p_17.transactiondate )) sq2
USING (parcelid)
WHERE (latitude IS NOT NULL AND longitude IS NOT NULL)
AND properties_2017.propertylandusetypeid NOT IN (31, 47,246, 247, 248, 267, 290, 291)
LIMIT 10000;""", database = 'zillow')




print(zillow_data)
print('---------------|DATABASE_INFO|---------------')
print(zillow_data.info())
print(zillow_data.describe())
print(zillow_data.shape)


percent_missing = database_percent(zillow_data)
print(zillow_data)
print(database_percent(zillow_data))
a = input('>><<')
error




