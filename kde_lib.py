import time
import numpy as np
from sklearn.neighbors import KernelDensity
from scipy.spatial.distance import euclidean
from sklearn.metrics.pairwise import rbf_kernel, euclidean_distances
from cvxopt import matrix, solvers
from sklearn.model_selection import GridSearchCV, LeaveOneOut, KFold


def gaussian_kernel(X, h, d):
    """
    Apply gaussian kernel to the input distance X
    """
    K = np.exp(-X / (2 * (h**2))) / ((2 * np.pi * (h**2))**(d / 2))
    return K


def rho(x, typ='hampel', a=0, b=0, c=0):
    if typ == 'huber':
        in1 = (x <= a)
        in2 = (x > a)
        in1_t = x[in1]**2 / 2
        in2_t = x[in2] * a - a**2 / 2
        L = np.sum(in1_t) + np.sum(in2_t)
    if typ == 'hampel':
        in1 = (x < a)
        in2 = np.logical_and(a <= x, x < b)
        in3 = np.logical_and(b <= x, x < c)
        in4 = (c <= x)
        in1_t = (x[in1]**2) / 2
        in2_t = a * x[in2] - a**2 / 2
        in3_t = (a * (x[in3] - c)**2 / (2 * (b - c))) + a * (b + c - a) / 2
        in4_t = np.ones(x[in4].shape) * a * (b + c - a) / 2
        L = np.sum(in1_t) + np.sum(in2_t) + np.sum(in3_t) + np.sum(in4_t)
    if typ == 'square':
        t = x**2
    if typ == 'abs':
        t = np.abs(x)
        L = np.sum(t)

    return L / x.shape[0]


def loss(x, typ='hampel', a=0, b=0, c=0):
    return rho(x, typ=typ, a=a, b=b, c=c) / x.shape[0]


def psi(x, typ='hampel', a=0, b=0, c=0):
    if typ == 'huber':
        return np.minimum(x, a)
    if typ == 'hampel':
        in1 = (x < a)
        in2 = np.logical_and(a <= x, x < b)
        in3 = np.logical_and(b <= x, x < c)
        in4 = (c <= x)
        in1_t = x[in1]
        in2_t = np.ones(x[in2].shape) * a
        in3_t = a * (c - x[in3]) / (c - b)
        in4_t = np.zeros(x[in4].shape)
        return np.concatenate((in1_t, in2_t, in3_t, in4_t)).reshape((-1, x.shape[1]))
    if typ == 'square':
        return 2 * x
    if typ == 'abs':
        return 1


def phi(x, typ='hampel', a=0, b=0, c=0):
    x[x == 0] = 10e-6
    return psi(x, typ=typ, a=a, b=b, c=c) / x


def irls(Km, type_rho, n, a, b, c, alpha=10e-8, max_it=100):
    """
    Iterative reweighted least-square
    """
    # init weights
    w = np.ones((n, 1)) / n
    # first pass
    t1 = np.diag(Km).reshape((-1, 1))  #  (-1, dimension)
    t2 = - 2 * np.dot(Km, w)
    t3 = np.dot(np.dot(w.T, Km), w)
    t = t1 + t2 + t3
    norm = np.sqrt(t)
    J = loss(norm, typ=type_rho, a=a, b=b, c=c)
    stop = 0
    count = 0
    losses = [J]
    while not stop:
        count += 1
        # print("i: {}  loss: {}".format(count, J))
        J_old = J
        # update weights
        w = phi(norm, typ=type_rho, a=a, b=b, c=c)
        w = w / np.sum(w)
        t1 = np.diag(Km).reshape((-1, 1))  #  (-1, dimension)
        t2 = - 2 * np.dot(Km, w)
        t3 = np.dot(np.dot(w.T, Km), w)
        t = t1 + t2 + t3
        norm = np.sqrt(t)
        # update loss
        J = loss(norm, typ=type_rho, a=a, b=b, c=c)
        losses.append(J)
        if ((np.abs(J - J_old) < (J_old * alpha)) or (count == max_it)):
            print("Stop at {} iterations".format(count))
            stop = 1
    return w, norm, losses


def kde(X_data, X_plot, h, kernel='gaussian', return_model=False):
    kde_fit = KernelDensity(kernel=kernel, bandwidth=h).fit(X_data)
    if return_model:
        return np.exp(kde_fit.score_samples(X_plot)), kde_fit
    else:
        return np.exp(kde_fit.score_samples(X_plot))


def area_density(z, grid):
    if grid is None:
        print('\nWARNING: no grid ==> return area = 1')
        return 1
    shapes = [el.shape[0] for el in grid]
    area = np.trapz(z.reshape(shapes), grid[0], axis=0)
    for i_grid, ax in enumerate(grid[1:]):
        area = np.trapz(area, ax, axis=0)
    return area


def area_MC_mom(X, model_momkde, n_mc=100000, distribution='kde', h=1):
    """
    Parameters
    -----
    distribution : 'uniform', 'kde'
    h : if 'kde', need to specifiy the bandwidth h

    Returns
    -----
    """
    cube_lows = []
    cube_highs = []
    dim = X.shape[1]
    if distribution == 'uniform':
        for d in range(dim):
            x_min, x_max = X[:, d].min(), X[:, d].max()
            offset = np.abs(x_max - x_min) * 0.5
            cube_lows.append(x_min - offset)
            cube_highs.append(x_max + offset)
        x_mc = np.random.uniform(cube_lows, cube_highs, size=(n_mc, X.shape[1]))
        p_mc = 1 / np.product(np.array(cube_highs) - np.array(cube_lows))
    elif distribution == 'kde':
        kde = KernelDensity(h)
        kde.fit(X)
        x_mc = kde.sample(n_mc)
        p_mc = np.exp(kde.score_samples(x_mc))

    res = []
    for kde_k in model_momkde:
        res.append(np.exp(kde_k.score_samples(x_mc)))
    res = np.median(res, axis=0)
    res = res / p_mc
    area = np.mean(res)
    # print('area MC: {}'.format(area))
    return area


def mom_kde(X_data,
            X_plot,
            h,
            outliers_fraction,
            grid,
            K='auto',
            kernel='gaussian',
            median='pointwise',
            norm=True,
            return_model=False,
            h_std=False):
    """
    Returns
    -----
    (if return_model=True) KDE_K: the list of all kdes fitted on the blocks.
        Warning : the KDE_K is not normed to area=1, only z is normed.
    """
    # t0 = time.time()
    n_samples = X_data.shape[0]
    if K == 'auto':
        K = int(2 * n_samples * outliers_fraction) + 1
    # print("N blocks: ", K)
    # print("N samples per block: ", int(n_samples / K))
    KDE_K = []
    X_shuffle = np.array(X_data)
    np.random.shuffle(X_shuffle)
    z = []
    for k in range(K):
        values = X_shuffle[k * int(n_samples / K):(k + 1) * int(n_samples / K), :]
        if h_std:
            std = np.std(values)
            h0 = h * std
        else:
            h0 = h
        kde = KernelDensity(bandwidth=h0)
        kde.fit(values)
        KDE_K.append(kde)
        z.append(np.exp(kde.score_samples(X_plot)))
    if median == 'pointwise':
        z = np.median(z, axis=0)
    elif median == 'geometric':
        distance = euclidean
        z = min(map(lambda p1: (p1, sum(map(lambda p2: distance(p1, p2), z))), z), key=lambda x: x[1])[0]
    else:
        raise ValueError('Wrong value for argument median: ' + median)
    # t1 = time.time() - t0
    # print("time fit MOM-KDE : ", (str(np.round(t1, 2)) + 's'))
    if norm:
        if grid is None:
            print('no grid specified, computing area with MC')
            area = area_MC_mom(X_data, KDE_K)
        else:
            area = area_density(z, grid)
        print("area mom (before normalization) :{}".format(area))
        z = z / area
    if return_model:
        return z, KDE_K
    else:
        return z


def rkde(X_data, X_plot, h, type_rho='hampel', return_model=False):
    t0 = time.time()
    # kernel matrix
    n_samples, d = X_data.shape
    gamma = 1. / (2 * (h**2))
    Km = rbf_kernel(X_data, X_data, gamma=gamma) * (2 * np.pi * h**2)**(-d / 2.)
    # find a, b, c via iterative reweighted least square
    a = b = c = 0
    alpha = 10e-8
    max_it = 100
    # first it. reweighted least ssquare with rho = absolute function
    w, norm, losses = irls(Km, 'abs', n_samples, a, b, c, alpha, max_it)
    a = np.median(norm)
    b = np.percentile(norm, 75)
    c = np.percentile(norm, 95)
    # find weights via second iterative reweighted least square with input rho
    w, norm, losses = irls(Km, type_rho, n_samples, a, b, c, alpha, max_it)
    # kernel evaluated on plot data
    gamma = 1. / (2 * (h**2))
    K_plot = rbf_kernel(X_plot, X_data, gamma=gamma) * (2 * np.pi * h**2)**(-d / 2.)
    # final density
    z = np.dot(K_plot, w)
    t1 = time.time() - t0
    print("time fit RKDE : ", (str(np.round(t1, 2)) + 's'))
    if return_model:
        return z, w
    else:
        return z


def spkde(X_data, X_plot, h, outliers_fraction, return_model=False):
    t0 = time.time()
    d = X_data.shape[1]
    beta = 1. / (1 - outliers_fraction)
    gamma = 1. / (2 * (h**2))
    G = rbf_kernel(X_data, X_data, gamma=gamma) * (2 * np.pi * h**2)**(-d / 2.)

    P = matrix(G)
    q = matrix(-beta / X_data.shape[0] * np.sum(G, axis=0))
    G = matrix(-np.identity(X_data.shape[0]))
    h_solver = matrix(np.zeros(X_data.shape[0]))
    A = matrix(np.ones((1, X_data.shape[0])))
    b = matrix(1.0)
    sol = solvers.qp(P, q, G, h_solver, A, b)
    a = np.array(sol['x']).reshape((-1, ))
    t1 = time.time() - t0
    print("time fit SPKDE : ", (str(np.round(t1, 2)) + 's'))
    # final density
    GG = rbf_kernel(X_data, X_plot, gamma=gamma) * (2 * np.pi * h**2)**(-d / 2.)
    z = np.zeros((X_plot.shape[0]))
    for j in range(X_plot.shape[0]):
        for i in range(len(a)):
            z[j] += a[i] * GG[i, j]
    if return_model:
        return z, a
    else:
        return z


def bandwidth_select(X_data, htype=4):
    n = X_data.shape[0]
    d = X_data.shape[1]
    sigma = np.logspace(-1.5, 0.5, 100)  # grid for method 2 et 3

    if htype == 1:
        h = .25
        return h
    elif htype == 2:
        # least square cross validation
        losses = []
        distance = euclidean_distances(X_data)
        # distance[distance == 0] = np.inf
        Jmin = np.inf
        for i in range(len(sigma)):
            h_temp = sigma[i]
            K1 = (4 * np.pi * h_temp**2)**(-d / 2) * np.exp(-distance / (4 * h_temp**2))
            K2 = (2 * np.pi * h_temp**2)**(-d / 2) * (np.exp(-distance / (2 * h_temp**2)) - np.identity(n))
            J = np.sum(K1) / (n**2) - 2 / (n * (n - 1)) * np.sum(K2)
            # print(J)
            losses.append(J)
            if J < Jmin:
                h = h_temp.copy()
                Jmin = J.copy()
        return h, sigma, losses
    elif htype == 3:
        # log-likelihood cross validation
        losses = []
        distance = euclidean_distances(X_data)
        distance[distance == 0] = np.inf
        Jmax = -np.inf
        for i in range(len(sigma)):
            h_temp = sigma[i]
            d = 2
            Kk = (2 * np.pi * h_temp**2)**(-d / 2) * (np.exp(-distance / (2 * h_temp**2)) - np.identity(n))
            J = 1 / n * np.sum(np.log(np.sum(Kk) / (n - 1)))
            losses.append(J)
            if J > Jmax:
                h = h_temp.copy()
                Jmax = J.copy()
        return h, sigma, losses
    elif htype == 4:
        # Jaakola heuristic
        distance = euclidean_distances(X_data)
        distance[distance == 0] = np.inf
        h = np.median(np.min(distance, axis=1))
        return h
    else:
        raise ValueError('Unknown value for htype: {}'.format(htype))


def bandwidth_cv(X_data):
    n = X_data.shape[0]
    d = X_data.shape[1]
    sigma = np.logspace(-1.5, 0.5, 100)  # grid for method 2 et 3
    losses = []
    distance = euclidean_distances(X_data)
    distance_minusi = []
    for i in range(n):
        X_minusi = np.delete(X_data, i, axis=0)
        distance_minusi.append(euclidean_distances(X_minusi))
    # distance[distance == 0] = np.inf
    Jmin = np.inf
    for i in range(len(sigma)):
        h_temp = sigma[i]
        K = gaussian_kernel(distance, h_temp, d)
        K_minusi = []
        for i in range(n):
            K_minusi.append(gaussian_kernel(distance_minusi[i], h_temp, d))
        A = (1 / (n**2)) * np.sum((np.sum(K, axis=0))**2)
        B = (1 / (n - 1)) * np.sum(K_minusi)
        J = A - (2 / n) * B
        print('h_temp: {}, J: {}'.format(h_temp, J))
        losses.append(J)
        if J < Jmin:
            h = h_temp.copy()
            Jmin = J.copy()
    return h, sigma, losses


def bandwidth_cvgrid(X_data, loo=False, kfold=5):
    """
    Compute the best bandwidth along a grid search.

    Parameters
    -----
    X_data : input data

    Returns
    -----
    h : the best bandwidth
    sigma : the search grid
    losses : the scores along the grid
    """
    print("Finding best bandwidth...")
    sigma = np.logspace(-1.5, 0.5, 80)  # grid for method 2 et 3
    if loo:
        grid = GridSearchCV(KernelDensity(kernel='gaussian'),
                            {'bandwidth': sigma},
                            cv=LeaveOneOut())
    else:
        grid = GridSearchCV(KernelDensity(kernel='gaussian'),
                            {'bandwidth': sigma},
                            cv=KFold(n_splits=kfold))
    grid.fit(X_data)
    h = grid.best_params_['bandwidth']
    losses = grid.cv_results_['mean_test_score']
    # print('best h: ', h)
    return h, sigma, losses
