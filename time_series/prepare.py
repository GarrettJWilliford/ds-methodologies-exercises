def data_wrangle():
    return pd.read_csv('mega_dataframe.csv')


def time_data_wrangle():
    return pd.read_csv('time_dataframe.csv')




def datetime_convert(data):
    data['sale_date'] = pd.to_datetime(data['sale_date'], unit='ns')
    data['month'] = data['sale_date'].dt.month
    data['day_of_week'] = data['sale_date'].dt.weekday
    data['sale_total'] = data['item_price'] * data['sale_amount']
    data['sales_diff'] = data['sale_total'].diff(periods = 1)
    data = data.drop(columns = ['Unnamed: 0.1','Unnamed: 0.1.1','Unnamed: 0.1.1.1','Unnamed: 0_x'])
    print(data)
    #data.to_csv(r'time_dataframe.csv')
    return data


def dist_data(data, data_list = ['sale_amount', 'sale_data', 'item_data']):
    for d in data_list:
        sns.distplot(data[d])
        plt.show()


def data_index(data, index_key):
    return data.set_index(index_key)


def scraper_data(data, data_list, dist_graphs = False):
    data = pd.read_csv('time_dataframe.csv')
    if dist_graphs:
        for d in dist_graphs:
            sns.distplot(data[d])
            plt.show()
    