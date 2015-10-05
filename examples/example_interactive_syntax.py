"""This shows the simpler syntax for interactive use.  

simulating Ito SDE   dy = lam y dt + sigma (0.2 + y) dW
"""

import nsim

lam = -1.0
sigma = 0.3
y0 = 0.0

def f(y, t):
    return lam * y

def G(y, t):
    return sigma * (0.2 + y)


Model = nsim.newmodel(f, G, y0)

s = nsim.Simulation(Model, T=400.0)
s.output.plot()
