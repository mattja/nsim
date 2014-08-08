"""Example using a built-in model: Jansen-Rit neural mass model"""

import nsim

model = nsim.models.JansenRit()
# customize some parameters
model.g = 2000.0
model.ke = 120.0


sims = nsim.RepeatedSim(model, T=60.0, repeat=4)

sims.timeseries.plot()
sims[3].output.psd()
