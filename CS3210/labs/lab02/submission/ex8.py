import matplotlib.pyplot as plt
import numpy as np

COL = "column-major"
ROW = "row-major"

COL_TIME_ELAPSED = [6.35398, 3.2725, 1.7487,
                    1.8128, 1.8220, 1.79526, 1.8048, 1.77243, 1.76215]
ROW_TIME_ELAPSED = [4.80298, 2.47475, 1.29968,
                    1.25159, 1.25631, 1.25475, 1.23945, 1.24016, 1.2745]

COL_L1D_CACHE_MISSES = [3.94, 3.95,
                        3.98, 3.20, 3.90, 3.94, 3.89, 3.81, 3.65]
ROW_L1D_CACHE_MISSES = [0.21, 0.21,
                        0.21, 0.16, 0.21, 0.22, 0.21, 0.20, 0.21]

fig, ax = plt.subplots()

ax.set_xlabel("Number of threads")
ax.set_ylabel("Time elapsed (sec)")
# ax.set_ylabel("L1D cache misses")

ax.set_xticks([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12], [
              0, 1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048])

xi = [1, 2, 3, 4, 5, 6, 7, 8, 9]

ax.plot(xi, COL_TIME_ELAPSED, marker="o", label=COL)
ax.plot(xi, ROW_TIME_ELAPSED, marker="o", label=ROW)

# ax.set_yticks([0.4, 0.8, 1.2, 1.6, 2.0, 2.4, 2.8, 3.2, 3.6, 4.0, 4.4, 4.8],
#               ["0.4%", "0.8%", "1.2%", "1.6%", "2.0%", "2.4%", "2.8%", "3.2%", "3.6%", "4.0%", "4.4%", "4.8%"])

# ax.plot(xi, COL_L1D_CACHE_MISSES, marker="o", label=COL)
# ax.plot(xi, ROW_L1D_CACHE_MISSES, marker="o", label=ROW)

ax.legend()

fig.savefig("ex8-time.png", dpi=300)
# fig.savefig("ex8-cache.png", dpi=300)
