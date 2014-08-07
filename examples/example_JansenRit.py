"""Jansen-Rit example"""

import nsim


sims = nsim.RepeatedSim(nsim.models.JansenRit, T=60, repeat=4)

sims.plot()
sims.psd()
