from libs import kde_lib
from libs import data
from libs.exp_lib import Density_model
import numpy as np


# =======================================================
#   Parameters
# =======================================================
algos = [
    'kde',
    'mom-kde',
    'rkde',
    'spkde',
]

# Synth params
dim = 1
n_samples = 500
outlier_scheme = 'uniform'
# outlier_scheme = 'reg_gaussian'
# outlier_scheme = 'thin_gaussian'
# outlier_scheme = 'adversarial'

dataset = ('D_' + str(dim) + '_' + outlier_scheme)

outlierprop_range = [0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5]
grid_points = 500
kernel = 'gaussian'

n_exp = 10

WRITE_SCORE = 1
scores_file = "outputs/scores/scores_2020_06_10.csv"

# =======================================================
#   Processing
# =======================================================
for i_exp in range(n_exp):
    print('EXP: {} / {}'.format(i_exp + 1, n_exp))
    for i_outlierprop, outlier_prop in enumerate(outlierprop_range):
        print('\nOutlier prop: {} ({} / {})'.format(outlier_prop, i_outlierprop + 1, len(outlierprop_range)))
        epsilon = outlier_prop / (1 - outlier_prop)
        X, y = data.generate_situations(n_samples,
                                        outlier_prop,
                                        outlier_scheme=outlier_scheme,
                                        dim=dim)
        # Grid on which to evaluate densities
        grid, X_plot = data.make_grid(X, grid_points)
        n_outliers = np.sum(y == 0)
        # Find bandwidth
        X_inlier = X[y == 1]
        h_cv, _, _ = kde_lib.bandwidth_cvgrid(X_inlier)
        print("h CV: ", h_cv)
        true_dens = data.true_density_situations(X_plot)
        # set range for k (number blocks)
        if epsilon == 0:
            k_range = [1]
        else:
            if epsilon < (1 / 3):
                k_max = 2 * n_outliers + 1
            else:
                k_max = X.shape[0] / 2
            k_range = np.linspace(1, k_max, 20).astype(int)
        # Processing all algos
        for algo in algos:
            print('\nAlgo: ', algo)
            kde_model = Density_model(algo, dataset, outlier_prop, kernel, h_cv)
            # if mom, run on several k
            if algo == 'mom-kde':
                k_range_run = k_range
            else:
                k_range_run = [1]
            for k in k_range_run:
                print('k: ', k)
                kde_model.fit(X, X_plot, grid, k=k, norm_mom=True)
                kde_model.compute_score(true_dens)
                if epsilon != 0:
                    # density on observation
                    kde_model.estimate_density(X)
                    kde_model.compute_anomaly_roc(y)
                if WRITE_SCORE:
                    kde_model.write_score(scores_file)
