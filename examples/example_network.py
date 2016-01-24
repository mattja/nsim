"""A simple network model.

This example couples together 4 Jansen-Rit models using a 
directed connectivity graph with different connection strengths.

This example also shows how a model can define its own custom coupling function

(In this particular model, for each connection from a source to a target the
target model's input variable v[5] is driven by a sigmoid function S of the
difference between the source model's two outputs: S(v[1] - v[2]), weighted by
the connection strength. This coupling is defined in nsim.models.JansenRit.)
"""

import nsim
import numpy as np

nodes = [nsim.models.JansenRit() for i in range(4)] # four nodes in the network

network = np.array([[  0,   0.2, 0.6,   0],  # weighted directed graph
                    [0.2,     0,   0,   0],
                    [0.2,     0,   0, 0.3],
                    [  0,   0.2, 0.3,   0]])

model = nsim.NetworkModel(nodes, network)
sim = nsim.Simulation(model, T=60.0)
sim.timeseries.plot(title='time series at each node, with network coupling')


# try the same simulation without any coupling between them:
network2 = np.zeros((4, 4))
model2 = nsim.NetworkModel(nodes, network2)
sim2 = nsim.Simulation(model2, T=60.0)
sim2.timeseries.plot(title='time series at each node, with no coupling')
