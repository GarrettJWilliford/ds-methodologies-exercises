from dbtools import get_db_url
import env
import MySQLdb

def get_db_url():
    db=MySQLdb.connect(host='157.230.209.171', user = env.user, \
    passwd = env.password, db=database)
    return psql.read_sql(comm, con=db)





cust = get_db_url(comm = """SELECT customer_id, monthly_charges, tenure, total_charges
                    FROM customers WHERE contract_type_id = 3
                    ORDER BY total_charges DESC;""", \
                    database = 'telco_churn')




def wrangle_telco():
    cust = get_db_url(comm = """SELECT customer_id, monthly_charges, tenure, total_charges
                    FROM customers WHERE contract_type_id = 3
                    ORDER BY total_charges DESC;""", \
                    database = 'telco_churn')
    cust['total_charges'].apply(lambda x: x.strip())
    cust['total_charges'] = cust['total_charges'].apply(lambda x: float(x) if x[0].isdigit() else 0)
    return telco