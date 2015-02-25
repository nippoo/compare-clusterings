from correlations import *
import kwiklib
import matplotlib.pyplot as plt

post = kwiklib.Experiment("/home/nippoo/neurodata/Achilles/postsleep/achilles_postsleep_firsthour.kwik")
pre = kwiklib.Experiment("/home/nippoo/neurodata/Achilles/presleep/achilles_presleep_firsthour.kwik")

sim_m = normalize(compare_clusterings(pre, post))

plt.pcolor(sim_m, cmap='jet')
plt.title('Similarity between post- and pre-sleep clusterings')

# set the limits of the plot to the limits of the data
plt.axis([0, sim_m.shape[0], 0, sim_m.shape[1]])
plt.colorbar()

plt.pcolor(sim_m)

plt.savefig('sim.png')
