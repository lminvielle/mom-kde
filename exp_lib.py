import os
import time
import pandas as pd
import numpy as np
# import matplotlib.pyplot as plt
import sklearn.metrics as sk_metrics
from sklearn.metrics.pairwise import rbf_kernel

import metrics
import kde_lib


class Density_model:

    def __init__(self, name, dataset, outlier_prop, kernel, h, is_hcv):
        self.algo = name
        self.kernel = kernel
        self.bandwidth = h
        self.is_bandwidth_cv = is_hcv
        self.density = None
        self.n_block = None
        self.dataset = dataset
        # self.epsilon = None
        self.outliers_fraction = outlier_prop
        self.kullback_f0_f = None
        self.kullback_f_f0 = None
        self.jensen = None
        self.time = None
        self.auc_anomaly = None
        self.X_data = None

    def fit(self, X, X_plot, grid, k='auto', norm_mom=True, hstd_mom=False):
        # self.outliers_fraction = self.epsilon / (1 + self.epsilon)
        t0 = time.time()
        if self.algo == 'kde':
            self.density, self.model = kde_lib.kde(X,
                                                   X_plot,
                                                   self.bandwidth,
                                                   self.kernel,
                                                   return_model=True)
        elif self.algo == 'mom-kde':
            self.n_block = k
            self.density, self.model = kde_lib.mom_kde(X,
                                                       X_plot,
                                                       self.bandwidth,
                                                       self.outliers_fraction,
                                                       grid,
                                                       K=k,
                                                       h_std=hstd_mom,
                                                       median='pointwise',
                                                       norm=norm_mom,
                                                       return_model=True)
        elif self.algo == 'mom-geom-kde':
            self.n_block = k
            self.density, self.model = kde_lib.mom_kde(X,
                                                       X_plot,
                                                       self.bandwidth,
                                                       self.outliers_fraction,
                                                       grid,
                                                       K=k,
                                                       median='geometric',
                                                       return_model=True)
        elif self.algo == 'rkde':
            self.X_data = X
            self.density, self.model = kde_lib.rkde(X,
                                                    X_plot,
                                                    self.bandwidth,
                                                    type_rho='hampel',
                                                    return_model=True)
        elif self.algo == 'spkde':
            self.X_data = X
            self.density, self.model = kde_lib.spkde(X,
                                                     X_plot,
                                                     self.bandwidth,
                                                     self.outliers_fraction,
                                                     return_model=True)
        else:
            raise ValueError('Wrong name of algo')
        t1 = time.time() - t0
        self.time = t1

    def compute_score(self, true_dens):
        if self.density is None:
            raise ValueError('Cannot compute score, density not estimated')
        self.kullback_f0_f = metrics.kl(true_dens.reshape((-1, 1)), self.density.reshape((-1, 1)))
        self.kullback_f_f0 = metrics.kl(self.density.reshape((-1, 1)), true_dens.reshape((-1, 1)))
        self.jensen = metrics.js(self.density.reshape((-1, 1)), true_dens.reshape((-1, 1)))

    def compute_anomaly_roc(self, y, plot_roc=False):
        fpr, tpr, thresholds = sk_metrics.roc_curve(y, self.density)
        self.auc_anomaly = sk_metrics.auc(fpr, tpr)
        # if plot_roc:
        # fig, ax = plt.subplots()
        # ax.plt(fpr, tpr, label='AUC = {:.2f}'.format(self.auc_anomaly))
        # plt.show()

    def estimate_density(self, X):
        model = self.model
        if self.algo == 'kde':
            # model : kde scikit-learn
            self.density = np.exp(model.score_samples(X))
        elif self.algo == 'mom-kde':
            # model : list of kdes scikit-learn
            z = []
            for k in range(len(model)):
                kde_k = model[k]
                z.append(np.exp(kde_k.score_samples(X)))
            self.density = np.median(z, axis=0)
        elif self.algo == 'rkde':
            # model : weights vector w
            n_samples, d = self.X_data.shape
            m = X.shape[0]
            K_plot = np.zeros((m, n_samples))
            for i_d in range(d):
                temp_xpos = X[:, i_d].reshape((-1, 1))
                temp_x = self.X_data[:, i_d].reshape((-1, 1))
                K_plot = K_plot + (np.dot(np.ones((m, 1)), temp_x.T) - np.dot(temp_xpos, np.ones((1, n_samples))))**2
            K_plot = kde_lib.gaussian_kernel(K_plot, self.bandwidth, d)
            z = np.dot(K_plot, model)
            self.density = z
        elif self.algo == 'spkde':
            # model : weights vector a
            d = self.X_data.shape[1]
            gamma = 1. / (2 * (self.bandwidth**2))
            GG = rbf_kernel(self.X_data, X, gamma=gamma) * (2 * np.pi * self.bandwidth**2)**(-d / 2.)
            z = np.zeros((X.shape[0]))
            for j in range(X.shape[0]):
                for i in range(len(model)):
                    z[j] += model[i] * GG[i, j]
            self.density = z
        else:
            print('no algo specified')

    def write_score(self, file_path):
        new_score_df = pd.DataFrame([[
            self.algo,
            self.dataset,
            self.bandwidth,
            self.is_bandwidth_cv,
            # self.epsilon,
            self.outliers_fraction,
            self.n_block,
            self.kullback_f0_f,
            self.kullback_f_f0,
            self.jensen,
            self.auc_anomaly,
            self.time
        ]])
        header_list = [
            "algo",
            "dataset",
            "bandwidth",
            "is_bandwidth_cv",
            # "epsilon",
            "outlier_prop",
            "n_block",
            "kullback_f0_f",
            "kullback_f_f0",
            "jensen",
            "auc_anomaly",
            "time",
        ]
        write_header = False
        if not os.path.isfile(file_path):
            write_header = True
        with open(file_path, 'a') as f:
            if write_header:
                new_score_df.to_csv(f, header=header_list, index=False)
            else:
                new_score_df.to_csv(f, header=False, index=False)
        f.close()
        return
