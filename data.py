import os
import numpy as np
from sklearn.utils import resample
from sklearn.datasets import load_iris, load_wine, load_breast_cancer, load_digits, make_moons
from sklearn.preprocessing import StandardScaler
from scipy.stats import norm as sci_norm
from scipy.stats import uniform as sci_uniform
from scipy.stats import multivariate_normal as sci_multinorm

current_file = os.path.dirname(os.path.abspath(__file__))
path_data = os.path.join(current_file, '../BDD/toy_data/')

datasets_asc = [
    'banana',
    'breast-cancer',
    'flare-solar',
    'german',
    'heart',
    'image',
    'ringnorm',
    'splice',
    'thyroid',
    'titanic',
    'twonorm',
    'waveform',
]


def load_data(dataset):
    full_path = path_data + '/' + dataset + '/' + dataset
    if dataset in datasets_asc:
        X1 = np.loadtxt(full_path + '_train_data_1.asc')
        X2 = np.loadtxt(full_path + '_test_data_1.asc')
        X = np.vstack((X1, X2))
        label1 = np.loadtxt(full_path + '_train_labels_1.asc')
        label2 = np.loadtxt(full_path + '_test_labels_1.asc')
        y = np.hstack((label1, label2))
    elif dataset == 'iris':
        data = load_iris()
        X = data.data
        y = data.target
    elif dataset == 'wine':
        data = load_wine()
        X = data.data
        y = data.target
    elif dataset == 'sk-breast-cancer':
        data = load_breast_cancer()
        X = data.data
        y = data.target
    elif dataset == 'digits':
        data = load_digits()
        X = data.data
        y = data.target
    else:
        raise ValueError('Unknown dataset')

    print("Loaded {} data: {} samples, {} dimensions, {} labels".format(dataset, X.shape[0], X.shape[1], len(set(y))))
    print("classes = {}".format(set(y)))
    return X, y


def set_inlierclass(dataset):
    if dataset in datasets_asc:
        cl = -1
    elif dataset == 'sk-breast-cancer':
        cl = 1
    elif dataset == 'iris_0':
        cl = 0
    elif dataset == 'iris_1':
        cl = 1
    elif dataset == 'iris_2':
        cl = 2
    elif dataset == 'wine_0':
        cl = 0
    elif dataset == 'wine_1':
        cl = 1
    elif dataset == 'wine_2':
        cl = 2
    elif dataset == 'digits_0':
        cl = 0
    elif dataset == 'digits_1':
        cl = 1
    elif dataset == 'digits_2':
        cl = 2
    elif dataset == 'digits_3':
        cl = 3
    elif dataset == 'digits_4':
        cl = 4
    elif dataset == 'digits_5':
        cl = 5
    elif dataset == 'digits_6':
        cl = 6
    elif dataset == 'digits_7':
        cl = 7
    elif dataset == 'digits_8':
        cl = 8
    elif dataset == 'digits_9':
        cl = 9
    elif dataset == 'digits_0_1':
        cl = 1
    else:
        raise ValueError('Unknown dataset: ' + dataset)
    return cl


def set_io_class(dataset):
    """
    Set inlier and outlier class for a given dataset
    """
    cl_i = 'all'
    if dataset in datasets_asc:
        cl_o = -1
    elif dataset == 'sk-breast-cancer':
        cl_o = 1
    elif dataset == 'iris_0':
        cl_o = 0
    elif dataset == 'iris_1':
        cl_o = 1
    elif dataset == 'iris_2':
        cl_o = 2
    elif dataset == 'wine_0':
        cl_o = 0
    elif dataset == 'wine_1':
        cl_o = 1
    elif dataset == 'wine_2':
        cl_o = 2
    elif dataset == 'digits_0':
        cl_o = 0
    elif dataset == 'digits_1':
        cl_o = 1
    elif dataset == 'digits_2':
        cl_o = 2
    elif dataset == 'digits_3':
        cl_o = 3
    elif dataset == 'digits_4':
        cl_o = 4
    elif dataset == 'digits_5':
        cl_o = 5
    elif dataset == 'digits_6':
        cl_o = 6
    elif dataset == 'digits_7':
        cl_o = 7
    elif dataset == 'digits_8':
        cl_o = 8
    elif dataset == 'digits_9':
        cl_o = 9
    elif dataset == 'digits_0_1':
        cl_i = 0
        cl_o = 1
    elif dataset == 'digits_1_0':
        cl_i = 1
        cl_o = 0
    else:
        raise ValueError('Unknown dataset: ' + dataset)
    return cl_i, cl_o


def set_datasetname(dataset):
    if 'iris' in dataset:
        return 'iris'
    elif 'wine' in dataset:
        return 'wine'
    elif 'digits' in dataset:
        return 'digits'
    else:
        return dataset


def load_data_outlier(dataset, scale=True):
    """
    Load real data

    Returns
    -----
    X : X is ordered as [inliers, outliers] so that user can call all inliers using
    X[:n_inlier, :]
    y : y is in [1, 0] with 1 for inliers and 0 for outliers
    """
    dataset_orig = set_datasetname(dataset)
    X, y = load_data(dataset_orig)
    if scale:
        scaler = StandardScaler()
        X = scaler.fit_transform(X)
    # if (outlier_class != 'auto'):
        # if (outlier_class not in set(y)):
        # raise ValueError('outlier_class not found in dataset labels')
    # else:
    inlier_class, outlier_class = set_io_class(dataset)
    if (inlier_class != 'all'):
        # inlier are a specific class
        if (inlier_class not in set(y)):
            raise ValueError('inlier_class not found in dataset labels')
        else:
            X_inlier = X[y == inlier_class]
    else:
        # inlier are every other classes than outlier
        X_inlier = X[y != outlier_class]

    X = np.r_[X_inlier, X[y == outlier_class]]

    # out labels: inliers=1, outliers=0
    y_out = np.zeros(X.shape[0])
    y_out[:X_inlier.shape[0]] = 1

    return X, y_out


def downsample(X, n_samples):
    """
    Downsampling X to n_samples
    """
    X = resample(X, replace=False, n_samples=n_samples)
    return X


def balance_outlier(X, y, outlier_label=0, e=0.1):
    """
    Resample X according to epsilon such that n_outlier = e * n_inlier

    Parameters
    -----
    X : data
    y : labels must be in [1, 0], with 1 being inliers and 0 being outliers
    """
    X_inlier = X[y == 1]
    X_outlier = X[y == 0]
    n_inlier_in = np.sum(y == 1)
    n_outlier_in = np.sum(y == 0)
    # if possible to downsample outliers then n_outlier equals:
    n_outlier_out = int(e * n_inlier_in)
    # if otherwise need to downsample inliers then n_inlier equals:
    n_inlier_out = int(n_outlier_in / e)
    if X_outlier.shape[0] < n_outlier_out:
        print("downsample inliers")
        # downsample inliers
        X_inlier = downsample(X_inlier, n_inlier_out)
    else:
        print("downsample outliers")
        # downsample outlier
        X_outlier = downsample(X_outlier, n_outlier_out)
    y = np.zeros(X_inlier.shape[0] + X_outlier.shape[0])
    y[:X_inlier.shape[0]] = 1
    return np.r_[X_inlier, X_outlier], y


def make_grid(X, n):
    """
    Create a grid according to the input data X

    Parameters
    -----
    X: input data, shape (samples, dimensions)
    n: number of points per dimension
    """
    grid = []
    for i in range(X.shape[1]):
        x_min, x_max = X[:, i].min(), X[:, i].max()
        offset = np.abs(x_max - x_min) * 0.1
        grid.append(np.linspace(x_min - offset, x_max + offset, n))
    mesh = np.meshgrid(*grid)
    X_flat = np.vstack([el.ravel() for el in mesh]).T
    return grid, X_flat


def generate(param_gauss, n_samples, outliers_fraction, inlier_type='gaussian', outlier_type='gaussian',
             outlier_param=[5, 0.2], dim=1):
    '''
    param_gauss = [mean_1, mean_2, var_1, var_2]

    outlier_type in ['gaussian', 'dirac', 'uniform']

    outlier_param - gaussian: [mean_outlier,var_outlier]

                  - dirac: [loc]

                  - uniform: [min,max]

    '''
    n_inliers = int((1. - outliers_fraction) * n_samples)
    n_outliers = int(outliers_fraction * n_samples)
    # if (dim == 2) & (inlier_type == 'moons'):
    # X = make_moons(n_samples=n_samples)[0]
    # else:
    X1 = np.random.normal(param_gauss[0], param_gauss[2], size=(n_inliers // 2, dim))
    X2 = np.random.normal(param_gauss[1], param_gauss[3], size=(n_inliers // 2, dim))
    # X2 = np.random.uniform(param_gauss[1], param_gauss[3], size=(n_inliers // 2, dim))
    X = np.r_[X1, X2]
    n_inliers = X.shape[0]

    # Add outliers
    # Gaussian outliers
    if outlier_type == 'gaussian':
        X = np.r_[X, np.random.normal(loc=outlier_param[0], scale=outlier_param[1],
                                      size=(n_outliers, dim))]
    # Dirac
    elif outlier_type == 'dirac':
        X = np.r_[X, np.random.normal(loc=outlier_param[0], scale=0,
                                      size=(n_outliers, dim))]
    # Uniform outliers
    elif outlier_type == 'uniform':
        X = np.r_[X, np.random.uniform(low=outlier_param[0], high=outlier_param[1],
                                       size=(n_outliers, dim))]
    elif outlier_type == 'mix':
        X = np.r_[X, np.random.uniform(low=outlier_param[0], high=outlier_param[1],
                                       size=(n_outliers // 2, dim))]
        X = np.r_[X, np.random.normal(loc=outlier_param[2], scale=outlier_param[3],
                                      size=(n_outliers // 2, dim))]
    else:
        print('Wrong anomaly type')
    y = np.zeros(X.shape[0])
    y[:n_inliers] = 1
    return X, y


def true_density(gauss_param, X_plot):
    coeff = 0.5
    d = X_plot.shape[1]
    if d == 1:
        # true_dens = coeff * sci_norm(gauss_param[0], gauss_param[2]).pdf(X_plot) + coeff * sci_norm(gauss_param[1], gauss_param[3]).pdf(X_plot)
        dens_1 = sci_norm(gauss_param[0], gauss_param[2]).pdf(X_plot)
        dens_2 = sci_norm(gauss_param[1], gauss_param[3]).pdf(X_plot)
        # dens_2 = sci_uniform(gauss_param[1], gauss_param[3]).pdf(X_plot)
        true_dens = coeff * dens_1 + coeff * dens_2
    if d > 1:
        true_dens = coeff * sci_multinorm((gauss_param[0], gauss_param[0]), gauss_param[2]).pdf(X_plot) + coeff * sci_multinorm((gauss_param[1], gauss_param[1]), gauss_param[3]).pdf(X_plot)
    return true_dens


if __name__ == '__main__':
    # Real data
    datasets = [
        'banana',
        'breast-cancer',
        'flare-solar',
        'german',
        'heart',
        'image',
        'ringnorm',
        'splice',
        'thyroid',
        'titanic',
        'twonorm',
        'waveform',
        'sk-breast-cancer',
        'iris',
        'digits',
    ]

    for dataset in datasets:
        X, y = load_data(dataset)

    # dataset = 'iris'
    # dataset = 'wine'
    # dataset = 'sk-breast-cancer'
    # dataset = 'iris_0'
    dataset = 'digits_0_1'
    X, y = load_data_outlier(dataset)
    X, y = balance_outlier(X, y, e=0.8)

    # Plot data
    # import matplotlib.pyplot as plt
    # fig, ax = plt.subplots()
    # X_inlier = X[y == 1]
    # X_outlier = X[y == 0]
    # ax.scatter(X_inlier[:, 0], X_inlier[:, 1])
    # ax.scatter(X_outlier[:, 0], X_outlier[:, 1])
    # plt.show()

    # grid, X_plot = make_grid(X, 100)

    # X, y = balance_outlier(X, y, e=0.8)

    # Synthetic data
    # XX = generate([0, 6, 0.5, 0.5], n_samples=100, outliers_fraction=0.05,
    # outlier_type='gaussian', outlier_param=[3, 0.2], dim=1)

    # XX = generate([0, 6, 0.5, 0.5], n_samples=100, outliers_fraction=0.05,
    # outlier_type='dirac', outlier_param=[0], dim=1)

    # XX = generate([0, 6, 0.5, 0.5], n_samples=100, outliers_fraction=0.05,
    # outlier_type='uniform', outlier_param=[-5, 7], dim=1)
