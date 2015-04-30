from correlations import *
import kwiklib
import matplotlib
import matplotlib.pyplot as plt

post = kwiklib.Experiment("/home/nippoo/neurodata/Achilles/postsleep/achilles_postsleep_firsthour.kwik")
pre = kwiklib.Experiment("/home/nippoo/neurodata/Achilles/presleep/achilles_presleep_firsthour.kwik")

sm, clusters_1, cluster_groups_1, clusters_2, cluster_groups_2 = compare_clusterings(pre, post)

invsm, invclusters_1, invcluster_groups_1, invclusters_2, invcluster_groups_2 = compare_clusterings(post, pre)

selected_cl_1 = (cluster_groups_1 == 2)
selected_cl_2 = (cluster_groups_2 == 2)

cl_numbers_1 = np.unique(clusters_1)[selected_cl_1]
cl_numbers_2 = np.unique(clusters_2)[selected_cl_2]

sm = sm[(selected_cl_1),:]
sm = sm[:,(selected_cl_2)]

invsm = normalize(invsm)
invsm = invsm[:, (selected_cl_1)]
invsm = invsm[(selected_cl_2), :]
invsm = invsm.T

sm = normalize(sm)
# sm = -np.log(sm)

x = np.arange(sm.shape[1])[None, :]
com = (x * sm).mean(axis=1)

sortidx = np.argsort(com)

sm = sm[sortidx,:]
invsm = invsm[sortidx, :]

ylen, xlen = sm.shape

print invsm.shape
print sm.shape

def patch(x, y, hatch, color, lw=0, alpha=1.):
    return matplotlib.patches.Rectangle((x-0.5, y-0.5), 1, 1,
                                        hatch=hatch, fill=False, color=color, lw=lw, alpha=alpha)

plt.xticks(np.arange(xlen), cl_numbers_2)
plt.yticks(np.arange(ylen), cl_numbers_1[sortidx])
plt.xlabel("postsleep clusters")
plt.ylabel("presleep clusters")

plt.axis([-0.5, xlen-0.5, -0.5, ylen-0.5])
# plt.imshow(sm, cmap='Reds', interpolation='nearest')

plt.imshow(np.zeros(sm.shape), cmap='Greys', interpolation='nearest')
plt.title('Similarity between post- and pre-sleep clusterings')

ax = plt.gca()

for i in range(sm.shape[0]): # rows
    for j in range(sm.shape[1]):

        ax.add_patch(patch(j, i, '///', "red", alpha=sm[i, j]))
        ax.add_patch(patch(j, i, '\\\\\\', "blue", alpha=invsm[i, j]))


    # ax.add_patch(patch(np.argmax(sm[i, :]), i, '//', "red"))
    # ax.add_patch(patch(np.argmax(invsm[i, :]), i, '\\\\', "blue"))

    # if np.argmax(sm[i, :]) == np.argmax(invsm[i, :]):
    #     ax.add_patch(patch(np.argmax(sm[i, :]), i, '', "green", lw="3"))
    if np.argmax(sm[i, :]) == np.argmax(invsm[i, :]):
        ax.add_patch(patch(np.argmax(sm[i, :]), i, '', "green", lw="3"))


# set the limits of the plot to the limits of the data
# plt.colorbar()

plt.savefig('mostsim.png')
