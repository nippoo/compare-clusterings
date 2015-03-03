from correlations import *
import kwiklib
import matplotlib.pyplot as plt

post = kwiklib.Experiment("/home/nippoo/neurodata/Achilles/postsleep/achilles_postsleep_firsthour.kwik")
pre = kwiklib.Experiment("/home/nippoo/neurodata/Achilles/presleep/achilles_presleep_firsthour.kwik")

sm, clusters_1, cluster_groups_1, clusters_2, cluster_groups_2 = compare_clusterings(pre, post)

selected_cl_1 = (cluster_groups_1 == 2)
selected_cl_2 = (cluster_groups_2 == 2)

cl_numbers_1 = np.unique(clusters_1)[selected_cl_1]
cl_numbers_2 = np.unique(clusters_2)[selected_cl_2]

sm = sm[(selected_cl_1),:]
sm = sm[:,(selected_cl_2)]

# sm = normalize(sm)
sm = -np.log(sm)

x = np.arange(sm.shape[1])[None, :]
com = (x * sm).mean(axis=1)

sortidx = np.argsort(com)

sm = sm[sortidx,:]

ylen, xlen = sm.shape

plt.xticks(np.arange(xlen), cl_numbers_2)
plt.yticks(np.arange(ylen), cl_numbers_1[sortidx])
plt.xlabel("postsleep clusters")
plt.ylabel("presleep clusters")

for label in plt.axes().get_xticklabels():
    label.set_horizontalalignment('left')

for label in plt.axes().get_yticklabels():
    label.set_verticalalignment('bottom')

plt.axis([0, xlen, 0, ylen])
plt.pcolor(sm, cmap='gray')

plt.title('Similarity between post- and pre-sleep clusterings')

# set the limits of the plot to the limits of the data
plt.colorbar()

plt.savefig('sim_goodnn.png')
