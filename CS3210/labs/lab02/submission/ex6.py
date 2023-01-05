import matplotlib.pyplot as plt
import numpy as np

PARTITION_1 = "i7-7700"
PARTITION_2 = "i7-9700"
PARTITION_3 = "xs-4114"
PARTITION_4 = "dxs-4114"

PARTITION_1_TIME_ELAPSED = [6.31594, 3.2662, 1.6918, 1.7134,
                            1.77322, 1.76810, 1.76311, 1.7613, 1.76004]

# PARTITION_2_TIME_ELAPSED = [5.8303, 2.9566, 1.52167,
#                             0.78267, 0.8665, 0.8720, 0.8656, 0.8609, 0.8636]

PARTITION_3_TIME_ELAPSED = [9.7940, 5.07155, 2.6831, 1.4424,
                            1.30846, 1.1928, 1.16233, 1.14833, 1.15243]

# PARTITION_4_TIME_ELAPSED = [9.8074, 5.0045, 2.6034, 1.42404,
#                             1.15581, 0.78723, 0.67014, 0.65221, 0.64099]

PARTITION_1_IPC = [2.41, 2.39, 2.39, 1.20, 1.16, 1.16, 1.16, 1.16, 1.16]

# PARTITION_2_IPC = [2.38, 2.38, 2.37, 2.34, 2.38, 2.39, 2.39, 2.39, 2.39]

PARTITION_3_IPC = [2.22, 2.21, 2.21, 2.20, 1.50, 1.22, 1.21, 1.21, 1.21]

# PARTITION_4_IPC = [2.21, 2.20, 2.19, 2.17, 1.70, 1.30, 1.24, 1.22, 1.21]

PARTITION_1_MFLOPS = [340.046, 330.025,
                      321.577, 161.116, 156.298, 155.749, 156.201, 156.423, 156.792]

# PARTITION_2_MFLOPS = [368.345, 364.912,
#                       356.676, 353.542, 335.684, 325.128, 325.572, 324.905, 323.802]

PARTITION_3_MFLOPS = [219.295, 213.012,
                      202.472, 191.661, 124.663, 101.294, 100.277, 100.610, 99.998]

# PARTITION_4_MFLOPS = []

fig, ax = plt.subplots()

ax.set_xlabel("Number of threads")
ax.set_ylabel("Time elapsed (sec)")
# ax.set_ylabel("IPC")
# ax.set_ylabel("MFLOPS")

ax.axvline(x=4, color="gray", linestyle="--")
ax.axvline(x=5.4, color="gray", linestyle="--")

ax.annotate("Physical cores in i7-7700", xy=(4.1, 8),
            xytext=(4.1, 8), color="gray")
ax.annotate("Physical cores in xs-4114", xy=(5.5, 8),
            xytext=(5.5, 4), color="gray")

# ax.annotate("Physical cores in i7-7700", xy=(4.1, 2.0),
#             xytext=(4.1, 2.0), color="gray")
# ax.annotate("Physical cores in xs-4114", xy=(5.5, 1.8),
#             xytext=(5.5, 1.8), color="gray")

# ax.annotate("Physical cores in i7-7700", xy=(4.1, 280),
#             xytext=(4.1, 280), color="gray")
# ax.annotate("Physical cores in xs-4114", xy=(4.1, 240),
#             xytext=(5.5, 240), color="gray")

ax.set_xticks([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12], [
              0, 1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048])

xi = [1, 2, 3, 4, 5, 6, 7, 8, 9]

ax.plot(xi, PARTITION_1_TIME_ELAPSED, marker="o", label=PARTITION_1)
# ax.plot(xi, PARTITION_2_TIME_ELAPSED, marker="o", label=PARTITION_2)
ax.plot(xi, PARTITION_3_TIME_ELAPSED, marker="o", label=PARTITION_3)
# ax.plot(xi, PARTITION_4_IPC, marker="o", label=PARTITION_4)

# ax.plot(xi, PARTITION_1_IPC, marker="o", label=PARTITION_1)
# ax.plot(xi, PARTITION_2_IPC, marker="o", label=PARTITION_2)
# ax.plot(xi, PARTITION_3_IPC, marker="o", label=PARTITION_3)
# ax.plot(xi, PARTITION_4_IPC, marker="o", label=PARTITION_4)

# ax.plot(xi, PARTITION_1_MFLOPS, marker="o", label=PARTITION_1)
# ax.plot(xi, PARTITION_2_MFLOPS, marker="o", label=PARTITION_2)
# ax.plot(xi, PARTITION_3_MFLOPS, marker="o", label=PARTITION_3)
# ax.plot(xi, PARTITION_4_MFLOPS, marker="o", label=PARTITION_4)

ax.legend()

fig.savefig("ex6-time.png", dpi=300)
# fig.savefig("ex6-ipc.png", dpi=300)
# fig.savefig("ex6-mflops.png", dpi=300)
