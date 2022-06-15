import numpy as np
import pandas as pd
import pandas_datareader as data
import matplotlib.pyplot as plt



start_date = "2021-01-1"
end_date = "2021-12-31"

test = data.DataReader(['TEL2-B.ST', 'SSAB-B.ST'], data_source='yahoo', start=start_date, end=end_date)


# Closing Price

test = test['Adj Close']

# Log of percentage change
tele2 = test['TEL2-B.ST'].pct_change().apply(lambda x: np.log(1+x))

var_tele2 = tele2.var()
print(f"Variance Tele2: {var_tele2}")

ssab = test['SSAB-B.ST'].pct_change().apply(lambda x: np.log(1+x))

var_ssab = ssab.var()
print(f"Varianc SSAB: {var_ssab}")

# Volatility
tele2_vol = np.sqrt(var_tele2 * 250)
ssab_vol = np.sqrt(var_ssab * 250)
print(tele2_vol, ssab_vol)

# Log of Percentage change
test1 = test.pct_change().apply(lambda x: np.log(1+x))
test1.head()

# Covariance
cov = test1['TEL2-B.ST'].cov(test1['SSAB-B.ST'])
print(f"Covariance: {cov}")

# Correlation
corr = test1['TEL2-B.ST'].corr(test1['SSAB-B.ST'])
print(f"Correlation: {corr}")

test2 = test.pct_change().apply(lambda x: np.log(1+x))
test2.head()

# Weights

weights = [0.5, 0.5]
exp_return = test2.mean()
print(exp_return)

# Total expected return
e_r = (exp_return*weights).sum()
print(e_r)
