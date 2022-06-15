import numpy as np
import pandas as pd
import pandas_datareader as data
import matplotlib.pyplot as plt

df = data.DataReader(['TEL2-B.ST', 'SBB-D.ST', 'ELUX-B.ST', 'HUFV-A.ST', 'AXFO.ST', 'AZA.ST'], 'yahoo', start='2011/01/01', end='2021/12/31')


df = df['Adj Close']

# Log of percentage change
cov_matrix = df.pct_change().apply(lambda x: np.log(1+x)).cov()
# print(cov_matrix)

corr_matrix = df.pct_change().apply(lambda x: np.log(1+x)).corr()
# print(corr_matrix)

w = {'TEL2-B.ST': 0.2, 'SBB-D.ST': 0.2, 'HUFV-A.ST': 0.2, 'AXFO.ST': 0.2, 'ELUX-B.ST': 0.1, 'AZA.ST': 0.1}
port_var = cov_matrix.mul(w, axis=0).mul(w, axis=1).sum().sum()
# print(port_var)

ind_er = df.resample('Y').last().pct_change().mean()
# print(ind_er)

w = [0.2, 0.2, 0.2, 0.2, 0.1, 0.1]
port_er = (w*ind_er).sum()
# print(port_er)

ann_sd = df.pct_change().apply(lambda x: np.log(1+x)).std().apply(lambda x: x*np.sqrt(250))


assets = pd.concat([ind_er, ann_sd], axis=1) # Creating a table for visualising returns and volatility of assets
assets.columns = ['Returns', 'Volatility']


p_ret = [] # Define an empty array for portfolio returns
p_vol = [] # Define an empty array for portfolio volatility
p_weights = [] # Define an empty array for asset weights

num_assets = len(df.columns)
num_portfolios = 10000

for portfolio in range(num_portfolios):
    weights = np.random.random(num_assets)
    weights = weights/np.sum(weights)
    p_weights.append(weights)
    returns = np.dot(weights, ind_er) # Returns are the product of individual expected returns of asset and its
                                      # weights
    p_ret.append(returns)
    var = cov_matrix.mul(weights, axis=0).mul(weights, axis=1).sum().sum()# Portfolio Variance
    sd = np.sqrt(var) # Daily standard deviation
    ann_sd = sd*np.sqrt(250) # Annual standard deviation = volatility
    p_vol.append(ann_sd)

data = {'Returns':p_ret, 'Volatility':p_vol}

for counter, symbol in enumerate(df.columns.tolist()):
    #print(counter, symbol)
    data[symbol+' weight'] = [w[counter] for w in p_weights]

portfolios = pd.DataFrame(data)
portfolios.head() # Dataframe of the 10000 portfolios created

# Plot efficient frontier
portfolios.plot.scatter(x='Volatility', y='Returns', marker='o', s=10, alpha=0.3, grid=True, figsize=[10,10])
# plt.show()

min_vol_port = portfolios.iloc[portfolios['Volatility'].idxmin()]
print(min_vol_port)

# plotting the minimum volatility portfolio
plt.subplots(figsize=[10,10])
plt.scatter(portfolios['Volatility'], portfolios['Returns'],marker='o', s=10, alpha=0.3)
plt.scatter(min_vol_port[1], min_vol_port[0], color='r', marker='*', s=500)
# plt.show()

# Finding the optimal portfolio
rf = 0.01 # risk factor
optimal_risky_port = portfolios.iloc[((portfolios['Returns']-rf)/portfolios['Volatility']).idxmax()]
print(optimal_risky_port)

# Plotting optimal portfolio
plt.subplots(figsize=(10, 10))
plt.scatter(portfolios['Volatility'], portfolios['Returns'],marker='o', s=10, alpha=0.3)
plt.scatter(min_vol_port[1], min_vol_port[0], color='r', marker='*', s=500)
plt.scatter(optimal_risky_port[1], optimal_risky_port[0], color='g', marker='*', s=500)
plt.show()