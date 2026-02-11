def max_drawdown(equity):
    roll_max = equity.cummax()
    drawdown = equity / roll_max - 1
    return drawdown.min()

def sharpe_ratio(returns):
    return returns.mean() / returns.std() * np.sqrt(252*24*4)
