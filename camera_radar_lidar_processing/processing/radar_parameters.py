import numpy as np

c = 3e8
fc = 24.125e9
lam = c / fc
BW = 554e6
N = 256
clk = 38461538
delay = 2214

delta_v = (lam * clk * 3.6) / (2 * N * (12 * (N + 4) + delay))
Vmax = delta_v * (N // 2)
range_bins = np.arange(N) * (c / (2 * BW))
velocity_bins = np.arange(N) * delta_v - Vmax