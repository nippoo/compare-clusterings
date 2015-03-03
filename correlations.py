"""Computes the similarity matrix between clusters of two different datasets with potentially differing numbers of clusters, or two clusterings from a given dataset."""

# -----------------------------------------------------------------------------
# Imports
# -----------------------------------------------------------------------------

import numpy as np

def compare_clusterings(exp1, exp2):
    clusters_1, masks_1, features_1, cluster_groups_1 = load_clusters_masks_features(exp1)
    clusters_2, masks_2, features_2, cluster_groups_2 = load_clusters_masks_features(exp2)

    cm = compute_matrix(clusters_1, masks_1, features_1,
                          clusters_2, masks_2, features_2)

    sm = get_similarity_matrix(cm, clusters_1, clusters_2)

    return sm, clusters_1, cluster_groups_1, clusters_2, cluster_groups_2


def load_clusters_masks_features(exp, channel_group=1, clustering='main'):

        spikes_data = exp.channel_groups[channel_group].spikes
        spikes_selected, fm = spikes_data.load_features_masks(fraction=.1)
        clusters = getattr(spikes_data.clusters, clustering)[:][spikes_selected]

        fm = np.atleast_3d(fm)
        features = fm[:, :, 0]

        if features.shape[1] <= 1:
            return []

        if fm.shape[2] > 1:
            masks = fm[:, :, 1]
        else:
            masks = None


        clusters_data = getattr(exp.channel_groups[channel_group].clusters, clustering)
        cluster_groups_data = getattr(exp.channel_groups[channel_group].cluster_groups, clustering)
        clusters_all = sorted(clusters_data.keys())
        cluster_groups = np.array([clusters_data[cl].cluster_group or 0
                                   for cl in clusters_all])

        return clusters, masks, features, cluster_groups

def compute_clustering_statistics(masks, features):
    """Compute global Gaussian statistics for a given clustering from the features and masks."""

    nspikes, ndims = features.shape
    masks = masks
    features = features

    # precompute the mean and variances of the masked points for each
    # feature
    # this contains 1 when the corresponding point is masked
    masked = np.zeros_like(masks)
    masked[masks == 0] = 1
    nmasked = np.sum(masked, axis=0)
    nu = np.sum(features * masked, axis=0) / nmasked
    # Handle nmasked == 0.
    nu[np.isnan(nu)] = 0
    nu = nu.reshape((1, -1))
    sigma2 = np.sum(((features - nu) * masked) ** 2, axis=0) / nmasked
    sigma2[np.isnan(sigma2)] = 0
    # sigma2 = sigma2.reshape((1, -1))
    # WARNING: make sure what is inside diag is a 1D array, otherwise
    # it will take the diag of a 2D (1, n) matrix instead of generating a
    # (n, n) diagonal matrix...
    D = np.diag(sigma2.ravel())
    # expected features
    y = features * masks + (1 - masks) * nu
    z = masks * features**2 + (1 - masks) * (nu ** 2 + sigma2)
    eta = z - y ** 2

    return y, sigma2, D, eta

def compute_cluster_statistics(spikes_in_clusters, masks, features, eta, y, D):
    """Compute the statistics of all clusters."""

    unmask_threshold = 10

    nspikes, ndims = features.shape
    nclusters = len(spikes_in_clusters)
    LogP = np.zeros((nspikes, nclusters))
    stats = {}

    for c in spikes_in_clusters:
        # "my" refers to "my cluster"
        myspikes = spikes_in_clusters[c]
        myfeatures = np.take(y, myspikes, axis=0).astype(np.float64)
        nmyspikes = len(myfeatures)
        mymasks = np.take(masks, myspikes, axis=0)
        mymean = np.mean(myfeatures, axis=0).reshape((1, -1))
        # Boolean vector of size (nchannels,): which channels are unmasked?
        unmask = ((mymasks>0).sum(axis=0) > unmask_threshold)
        mask = ~unmask
        nunmask = np.sum(unmask)
        if nmyspikes <= 1 or nunmask == 0:
            mymean = np.zeros((1, myfeatures.shape[1]))
            covmat = 1e-3 * np.eye(nunmask)  # optim: nactivefeatures
            stats[c] = (mymean, covmat,
                        (1e-3)**ndims, nmyspikes,
                        np.zeros(ndims, dtype=np.bool)  # unmask
                        )
            continue

        # optimization: covmat only for submatrix of active features
        covmat = np.cov(myfeatures[:, unmask], rowvar=0) # stats for cluster c

        # Variation Bayesian approximation
        priorpoint = 1
        covmat *= (nmyspikes - 1)  # get rid of the normalization factor
        covmat += D[unmask, unmask] * priorpoint  # D = np.diag(sigma2.ravel())
        covmat /= (nmyspikes + priorpoint - 1)

        # the eta just for the current cluster
        etac = np.take(eta, myspikes, axis=0)
        # optimization: etac just for active features
        etac = etac[:, unmask]
        d = np.mean(etac, axis=0)

        # Handle nmasked == 0
        d[np.isnan(d)] = 0

        # add diagonal
        covmat += np.diag(d)

        # Compute the det of the covmat
        _sign, logdet = np.linalg.slogdet(covmat)
        if _sign < 0:
            raise ValueError("The correlation matrix of cluster %d has a negative determinant" % c)

        stats[int(c)] = (mymean, covmat, logdet, nmyspikes, unmask)

    return stats

def compute_matrix(clusters_1, masks_1, features_1, clusters_2, masks_2, features_2):
    """Compare two clusterings from two different experiments to create a correlation matrix between every pair of clusters.

    A dictionary pairs => value is returned.

    Compute all rows and columns corresponding to clusters_to_update."""

    y_1, sigma2_1, D_1, eta_1 = compute_clustering_statistics(masks_1, features_1)
    y_2, sigma2_2, D_2, eta_2 = compute_clustering_statistics(masks_2, features_2)

    nspikes_1, ndims_1 = features_1.shape
    clusters_unique_1 = np.unique(clusters_1)

    nspikes_2, ndims_2 = features_2.shape
    clusters_unique_2 = np.unique(clusters_2)

    # Indices of spikes in each cluster
    spikes_in_clusters_1 = dict([(clu, np.nonzero(clusters_1 == clu)[0])
                               for clu in clusters_unique_1])
    spikes_in_clusters_2 = dict([(clu, np.nonzero(clusters_2 == clu)[0])
                               for clu in clusters_unique_2])

    stats_1 = compute_cluster_statistics(spikes_in_clusters_1, masks_1, features_1, eta_1, y_1, D_1)
    stats_2 = compute_cluster_statistics(spikes_in_clusters_2, masks_2, features_2, eta_2, y_2, D_2)

    # New matrix (clu0, clu1) => new value
    C = {}

    def _compute_coeff(ci, cj, stats_i, stats_j, sigma2_j, ndims_j, nspikes_j, inv):

        if ci not in stats_i or cj not in stats_j:
            C[ci, cj] = 0.
            return

        mui, Ci, logdeti, npointsi, unmaski = stats_i[ci]
        muj, Cj, logdetj, npointsj, unmaskj = stats_j[cj]

        if npointsi <= 1 or npointsj <= 1:
            C[ci, cj] = 0.
            return

        dmu = (muj - mui).reshape((-1, 1))

        unmasked = unmaskj
        masked = ~unmasked
        dmu_unmasked = dmu[unmasked]

        # pij is the probability that mui belongs to Cj:
        #    $$p_{ij} = w_j * N(\mu_i | \mu_j; C_j)$$
        # where wj is the relative size of cluster j
        # pii is the probability that mui belongs to Ci
        try:
            bj = np.linalg.solve(Cj, dmu_unmasked)
        except np.linalg.LinAlgError:
            bj = np.linalg.lstsq(Cj, dmu_unmasked)[0]

        var = np.sum(dmu[masked] ** 2 / sigma2_j[masked])
        logpij = (np.log(2*np.pi) * (-ndims_j/2.) +
                 -.5 * (logdetj + np.sum(np.log(sigma2_j[masked]))) +
                 -.5 * (np.dot(bj.T, dmu_unmasked) + var))

        # nspikes is the total number of spikes.
        wj = float(npointsj) / nspikes_j

        C[ci, cj, inv] = wj * np.exp(logpij)[0,0]

    for ci in clusters_unique_1:
        for cj in clusters_unique_2:
            _compute_coeff(ci, cj, stats_1, stats_2, sigma2_2, ndims_2, nspikes_2, False)
            _compute_coeff(cj, ci, stats_2, stats_1, sigma2_1, ndims_1, nspikes_1, True)

    return C

def get_similarity_matrix(dic, clusters_x, clusters_y):
    """Return a correlation matrix from a dictionary. Normalization happens
    here."""
    clu0, clu1, inv = zip(*dic.keys())
    clusters_x = sorted(np.unique(clusters_x))
    clusters_y = sorted(np.unique(clusters_y))
    clumax_x = max(clusters_x) + 1
    clumax_y = max(clusters_y) + 1
    matrix = np.zeros((len(clusters_x), len(clusters_y)))

    # Relative clusters: cluster absolute => cluster relative
    clusters_rel_x = np.zeros(clumax_x, dtype=np.int32)
    clusters_rel_y = np.zeros(clumax_y, dtype=np.int32)
    clusters_rel_x[clusters_x] = np.arange(len(clusters_x))
    clusters_rel_y[clusters_y] = np.arange(len(clusters_y))

    for (clu0, clu1, inv), value in dic.iteritems():
        if inv:
            matrix[clusters_rel_y[clu0], clusters_rel_x[clu1]] = value
        else:
            matrix[clusters_rel_x[clu0], clusters_rel_y[clu1]] = value

    return matrix

def normalize(matrix, direction='row'):

    if direction == 'row':
        s = matrix.sum(axis=1)
    else:
        s = matrix.sum(axis=0)

    # Non-null rows.
    indices = (s != 0)

    # Row normalization.
    if direction == 'row':
        matrix[indices, :] *= (1. / s[indices].reshape((-1, 1)))

    # Column normalization.
    else:
        matrix[:, indices] *= (1. / s[indices].reshape((1, -1)))

    return matrix
