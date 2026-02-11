import numpy as np

def backtest(close_prices, signal, transaction_cost=0.0001):
    returns = np.log(close_prices).diff()
    
    strategy_returns = signal.shift(1) * returns
    
    # coût de transaction
    trades = signal.diff().abs()
    costs = trades * transaction_cost
    
    strategy_returns = strategy_returns - costs
    
    equity = strategy_returns.cumsum().apply(np.exp)
    
    return equity, strategy_returns
