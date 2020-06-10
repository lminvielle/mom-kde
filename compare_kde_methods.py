import kde_lib
from exp_lib import Density_model
import numpy as np
import data


# =======================================================
#   Parameters
# =======================================================
algos = [
    'kde',
    'mom-kde',
    # 'mom-geom-kde',
    'rkde',
    'spkde',
]

datasets = [
    # 'banana',

    # 'titanic',

    # 'german',
    # 'sk-breast-cancer',
    # 'iris_0',
    # 'iris_1',
    # 'iris_2',
    # 'wine_0',
    # 'wine_1',
    # 'wine_2',
    # 'digits_0',
    # 'digits_1',
    # 'digits_2',
    # 'digits_3',
    # 'digits_4',
    # 'digits_5',
    'digits_6',
    'digits_7',
    'digits_8',
    'digits_9',
    # 'digits_1_0',
    # 'digits_0_1',
]

n_exp = 10
# epsilon_range = [0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3]  # n_outlier = epsilon * n_inlier
# epsilon_range = [0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9]
outlierprop_range = [0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5]
# grid_points = 100
# h_range_orig = np.linspace(0.05, 1, 30)
h_range_orig = np.array([])
kernel = 'gaussian'

WRITE_SCORE = 1
# scores_file = "outputs/scores/scores_2020_05_25.csv"  # only bandwidthcv
# scores_file = "outputs/scores/scores_2020_05_26.csv"  # more data sets, high epsilon range
# scores_file = "outputs/scores/scores_2020_05_26_2.csv"  # see effect of scaling
# scores_file = "outputs/scores/scores_2020_05_26_3.csv"  # all data hcv
# scores_file = "outputs/scores/scores_2020_05_26_4.csv"  # all data all h
# scores_file = "outputs/scores/scores_2020_05_28_2.csv"  # all data hcv, several times, with outlier proportion (instead of epsilon)
# scores_file = "outputs/scores/scores_2020_06_02_2.csv"  # all data, several times, with outlier proportion (instead of epsilon), h_range, IRIS
# scores_file = "outputs/scores/scores_2020_06_04.csv"  # all data, several times, with outlier proportion (instead of epsilon), h_range, BREAST_CANCER
# scores_file = "outputs/scores/scores_2020_06_04_digits.csv"  # all data, several times, with outlier proportion (instead of epsilon), h_range
# scores_file = "outputs/scores/scores_2020_06_04_german.csv"  # all data, several times, with outlier proportion (instead of epsilon), h_range
scores_file = "outputs/scores/scores_2020_06_10.csv"  # all data hcv, several times, with outlier proportion (instead of epsilon), all digits

# =======================================================
#   Processing
# =======================================================

for i_exp in range(n_exp):
    print('EXP: {} / {}'.format(i_exp + 1, n_exp))
    for dataset in datasets:
        print('Dataset: ', dataset)
        X0, y0 = data.load_data_outlier(dataset)
        # Grid on which to evaluate densities
        # grid, X_plot = data.make_grid(X0, grid_points)

        # Find bandwidth for true density
        h_cv, _, _ = kde_lib.bandwidth_cvgrid(X0)
        # h_cv = 0.15
        print("h CV: ", h_cv)
        # add h_cv to h range
        h_range = np.r_[h_range_orig, h_cv]

        # for epsilon in epsilon_range:
        for i_outlierprop, outlier_prop in enumerate(outlierprop_range):
            # print('Epsilon: ', epsilon)
            epsilon = outlier_prop / (1 - outlier_prop)
            # print('Outlier prop: {} (Epsilon: {})'.format(outlier_prop, epsilon))
            print('\nOutlier prop: {} ({} / {})'.format(outlier_prop, i_outlierprop + 1, len(outlierprop_range)))
            # balance the inlier / outlier according to epsilon
            if epsilon != 0:
                X, y = data.balance_outlier(X0, y0, e=epsilon)
            else:
                X = X0.copy()
                y = y0.copy()
            n_outliers = np.sum(y == 0)
            # evaluate on observations
            X_plot = X
            # compute true density
            X_inlier = X0[y0 == 1]
            true_dens = kde_lib.kde(X_inlier, X_plot, h_cv, 'gaussian')
            # set range for k (number blocks) according to outliers
            if epsilon == 0:
                k_range = [1]
            else:
                if epsilon < 1 / 3:
                    k_max = 2 * n_outliers + 1
                else:
                    k_max = X.shape[0] / 2
                k_range = np.linspace(1, k_max, 20).astype(int)

            for i_h, h in enumerate(h_range):
                print('\nBandwidth: {:.3f} ({} / {})'.format(h, i_h + 1, h_range.shape[0]))
                if i_h == h_range.shape[0] - 1:
                    is_hcv = True
                else:
                    is_hcv = False
                # Processing all algos
                for algo in algos:
                    print('Algo: ', algo)
                    model = Density_model(algo, dataset, outlier_prop, kernel, h, is_hcv)
                    # if mom, run on several k
                    if algo == 'mom-kde':
                        k_range_run = k_range
                    else:
                        k_range_run = [1]
                    for k in k_range_run:
                        print('k: ', k)
                        # WARNING NO NORM
                        try:
                            model.fit(X, X_plot, grid=None, k=k, norm_mom=False)
                        except ValueError:
                            print('\n WARNING!! Fit impossible (h too low ?)')
                            if WRITE_SCORE:
                                model.write_score(scores_file)
                            continue
                        # model.compute_score(true_dens)
                        if epsilon != 0:
                            model.compute_anomaly_roc(y)
                        if WRITE_SCORE:
                            model.write_score(scores_file)
