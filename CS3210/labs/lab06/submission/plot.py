from __future__ import division
import matplotlib.pyplot as plt
import numpy as np

labels = ['2', '4', '8', '16']

omp_i7 = [2.04, 1.12, 1.10, 1.13]
omp_xs = [3.09, 1.65, 0.88, 0.80]

mpi_i7 = [13.71, 19.24, 13.85, 13.03]
mpi_xs = [14.75, 17.94, 14.18, 12.75]

seq_i7 = 3.68
seq_xs = 5.63

omp_i7_s = [round(seq_i7/i, 2) for i in omp_i7]
omp_xs_s = [round(seq_xs/i, 2) for i in omp_xs]

mpi_i7_s = [round(seq_i7/i, 2) for i in mpi_i7]
mpi_xs_s = [round(seq_xs/i, 2) for i in mpi_xs]

omp_i7_f = [round((pow(2, i) - omp_i7_s[i-1]) / (omp_i7_s[i-1] * (pow(2, i) - 1)), 2) for i in range(1, 5)]
omp_xs_f = [round((pow(2, i) - omp_xs_s[i-1]) / (omp_xs_s[i-1] * (pow(2, i) - 1)), 2) for i in range(1, 5)]

omp_i7_f_adj = [round((min(pow(2, i), 4) - omp_i7_s[i-1]) / (omp_i7_s[i-1] * (min(pow(2, i), 4) - 1)), 2) for i in range(1, 5)]
omp_xs_f_adj = [round((min(pow(2, i), 10) - omp_xs_s[i-1]) / (omp_xs_s[i-1] * (min(pow(2, i), 10) - 1)), 2) for i in range(1, 5)]

x = np.arange(len(labels))
width = 0.1

fig, ax = plt.subplots()

# rects1 = ax.bar(x - width/2, omp_i7, width, label='i7-7700')
# rects2 = ax.bar(x + width/2, omp_xs, width, label='xs-4114')

# rects1 = ax.bar(x - width/2, [round(seq_i7/i, 2) for i in omp_i7], width, label='i7-7700')
# rects2 = ax.bar(x + width/2, [round(seq_xs/i, 2) for i in omp_xs], width, label='xs-4114')

# rects1 = ax.bar(x - width/2, mpi_i7, width, label='i7-7700')
# rects2 = ax.bar(x + width/2, mpi_xs, width, label='xs-4114')

# rects1 = ax.bar(x - width/2, [round(seq_i7/i, 2) for i in mpi_i7], width, label='i7-7700')
# rects2 = ax.bar(x + width/2, [round(seq_xs/i, 2) for i in mpi_xs], width, label='xs-4114')

# rects1 = ax.bar(x - width/2, omp_i7_f, width, label='i7-7700')
# rects2 = ax.bar(x + width/2, omp_xs_f, width, label='xs-4114')

# rects1 = ax.bar(x - width/2, omp_i7_f_adj, width, label='i7-7700')
# rects2 = ax.bar(x + width/2, omp_xs_f_adj, width, label='xs-4114')

rects1 = ax.bar(1 - width/2, [seq_i7], width, label='i7-7700')
rects2 = ax.bar(1 + width/2, [seq_xs], width, label='xs-4114')

# ax.set_xlabel('Number of threads')
ax.set_ylabel('Time elapsed (sec)')
# ax.set_ylabel('Sequential portion')
# ax.set_ylabel('Speedup')

ax.set_xticks([1], [' '])
ax.legend()

ax.bar_label(rects1, padding=3)
ax.bar_label(rects2, padding=3)

# fig.savefig("omp.png", dpi=300)
# fig.savefig("omp_s.png", dpi=300)
# fig.savefig("omp_f.png", dpi=300)
# fig.savefig("omp_f_adj.png", dpi=300)
# fig.savefig("mpi.png", dpi=300)
# fig.savefig("mpi_s.png", dpi=300)
# fig.savefig("mpi_f.png", dpi=300)
fig.savefig("seq.png", dpi=300)
