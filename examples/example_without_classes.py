"""1D Ornstein-Uhlenbeck processes"""

import nsim


lam = -1.0
sigma = 0.3
y0 = 0.0

def f(y, t):
    return lam * y

def G(y, t):
    return sigma



sims = nsim.quicksim(f, G, y0, T=60, repeat=4)
sims.plot()

