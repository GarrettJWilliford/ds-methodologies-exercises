import pydataset
from statsmodels.formula.api import ols
import dbtools
import pandas as pd
import seaborn as sns
import math
import matplotlib.pyplot as plt


bill = pydataset.data('tips')

x = pd.DataFrame(bill['total_bill'])
y = bill['tip']

ols_model = ols('y ~ x', data=bill).fit()
bill['yhat'] = ols_model.predict(x)
print(bill)

print(ols_model)
def plot_residuals(x, y, df):
    sns.residplot(df[x], df[y])
    plt.show()
    



def regression_errors(y, yhat):
    residual = yhat - y
    residual2 = residual ** 2
    SSE =  sum(residual2)
    MSE = SSE/len(yhat)
    RMSE = math.sqrt(MSE)
    ESS = sum((yhat - y.mean())**2)
    TSS = ESS + SSE
    R2 = (ESS / TSS)
    print('SSE: ', SSE)
    print('MSE: ', MSE)
    print('RMSE: ', RMSE)
    print('TSS: ', TSS)
    print('R2: ', R2)
    return SSE
    


def baseline_mean_errors(y):
    yhat = y.mean()
    residual = yhat - y
    residual2 = residual ** 2
    BASELINE_SSE =  sum(residual2)
    MSE = SSE/len(y)
    RMSE = math.sqrt(MSE)
    print('BASELINE_SSE: ', BASELINE_SSE)
    print('BASELINE_MSE: ', MSE)
    print('BASELINE_RMSE ', RMSE)
    return BASELINE_SSE
    

def better_than_baseline(SSE, BASELINE_SSE):
    if SSE > BASELINE_SSE:
        return True
    return False


#def model_significance(ols_model):
    
    



regr = ols('bill.total_bill ~ bill.tip', data=bill).fit()

#dbtools.db_info(bill)


bill['yhat'] = regr.predict(pd.DataFrame(x))


print(bill)



    



SSE = regression_errors(y, bill['yhat'])
BASELINE_SSE = baseline_mean_errors(y)
print(better_than_baseline(SSE, BASELINE_SSE))
plot_residuals('total_bill', 'tip', bill)




