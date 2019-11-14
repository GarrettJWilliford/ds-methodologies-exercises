import requests
import bs4
import os
import pandas as pd
from pprint import pprint
import csv





def api_template(request, order, df = True):
    response = requests.get(request)
    response = response.json()
    if not df:
        return response
    return pd.DataFrame(response['payload'])




base = 'https://python.zach.lol'



def api_items():
    base = 'https://python.zach.lol'
    response = requests.get(base + '/api/v1/items')
    response = response.json()
    df = pd.DataFrame(response['payload']['items'])
    df.to_csv(r'items.csv')
    

def api_stores():
    base = 'https://python.zach.lol'
    response = requests.get(base + '/api/v1/stores')
    response = response.json()
    df = pd.DataFrame(response['payload']['stores'])
    df.to_csv(r'stores.csv')
    


def api_sales():
    base = 'https://python.zach.lol'
    og_data = pd.DataFrame()
    for i in range(1, 184):
        print('iteration ' + str(i))
        response = requests.get(base + '/api/v1/sales?page=' + str(i))
        response = response.json()
        og_data = og_data.append(pd.DataFrame(response['payload']['sales']))        
    og_data.to_csv(r'sales.csv')
    

def combine():
    sales_data = pd.read_csv('sales.csv')
    item_data = pd.read_csv('items.csv')
    store_data = pd.read_csv('stores.csv')
    sales_data['store_id'] = sales_data['store']
    sales_data['item_id'] = sales_data['item']
    sales_data.drop(columns = ['store', 'item'])
    data = pd.merge(sales_data, store_data, on='store_id', how = 'left')
    data = pd.merge(data, item_data, on='item_id', how = 'left')
    data.to_csv(r'mega_dataframe.csv')
    return data



def german_api():
    response = requests.get('https://raw.githubusercontent.com/jenfly/opsd/master/opsd_germany_daily.csv')
    return response
