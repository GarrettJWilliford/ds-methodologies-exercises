import re
import pandas as pd
from selenium import webdriver
from selenium.webdriver.firefox.options import Options
import pickle
import re
import seaborn as sns
import matplotlib.pyplot as plt

#97.105.19.58


def driver_init(headless = True):
    if headless:
        fop = Options()
        fop.add_argument('--headless')
        fop.add_argument('--window_size1920x1080')
        return webdriver.Firefox(options = fop)
    return webdriver.Firefox()



def ip_look(driver, ip):
    driver.get('https://www.iplocation.net')
    driver.find_element_by_name('query').send_keys(ip)
    driver.find_element_by_name('submit').click()
    td = driver.find_elements_by_tag_name('td')
    return [t.get_attribute('innerHTML') for t in td][2:5]
        


def prep_ip_dict(data, driver):
    known_ips = {}
    for i in data['ip']:
        if i in known_ips.keys():
            continue
        ip_data = ip_look(driver, i)
        known_ips.update({i : ip_data})
        pickle.dump(known_ips, open('ip_data.p', 'wb'))
        print('----------' + i + '----------')
        print(ip_data)
    driver.quit()
    return known_ips








def redundant(data, driver):
    known_ips = pickle.load(open('ip_data.p', 'rb'))





def acquire():
    return pd.read_csv(r'/Users/garrettwilliford/Downloads/anonymized-curriculum-access.txt', \
                   sep=" ", header=None, names = ['date', 'timestamp', 'url', 'number', 'other_number', 'ip'])






def prep(data):
    data['timestamp'] = data['date'] + ' ' + data['timestamp']
    data['timestamp'] = pd.to_datetime(data['timestamp'])
    data['about_html'] = data['url'].str.contains('html')
    data['about_java'] = data['url'].str.contains('java-i')
    data['about_javascript'] = data['url'].str.contains('javascript')
    return data



def explore():
    index = pd.read_csv('crap.csv')
    data = acquire()
    data['date'] = pd.to_datetime(data['date'])
    data = data.merge(index, on = 'ip')
    print('-----------------------------------------------')
    print(data[data['province'] == 'California']['company'].value_counts())
    print('-----------------------------------------------')
    print(data[data['province'] == 'Oklahoma']['company'].value_counts())
    print('-----------------------------------------------')
    print(data[data['province'] == 'New Jersey']['number'].value_counts())
    print('-----------------------------------------------')
    subset = pd.DataFrame(data.groupby('province')['number'].nunique())
    print(subset)
    a = input('<<|>>')
    subset = subset.reset_index()
    subset = subset[subset['province'] != 'Texas']
    plot = sns.barplot(subset['province'], subset['number'])
    plot.set_xticklabels(plot.get_xticklabels(), rotation=90)
    plt.show()




