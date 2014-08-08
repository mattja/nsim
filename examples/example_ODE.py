"""ODE example:    d^2y/dt^2 = (0.5 sin(0.3 t) - 0.1) dy/dt - 3 y

writing as a two-dimensional system:

                   dy1/dt = y2
                   dy2/dt = (0.5 sin(0.3 t) - 0.1) y2 - 3 y1
"""
import nsim
import numpy as np



class ExampleODE(nsim.ODEModel):

    y0 = np.array([ 1.0, 
                    1.0 ])

    def f(self, y, t):
        return np.array([ y[1], 
                          (0.5*np.sin(0.3*t) - 0.1)*y[1] - 3*y[0] ])
                            


s = nsim.Simulation(ExampleODE, T=120.0)
s.timeseries.plot()
