import numpy as np


def norm_for_train(x, gamma, beta, params):
    running_mean = params['running_mean']
    running_var = params['running_var']
    mom = params['mom']
    eps = params['eps']
    x_mean = x.mean()
    x_var = x.var()

    running_mean = mom * running_mean + (1 - mom) * x_mean
    running_var = mom * running_var + (1 - mom) * x_var

    # 归一化
    x_norm = (x - running_mean) / np.sqrt(running_var + eps)

    res = gamma * x_norm + beta

    params['running_mean'] = running_mean
    params['running_var'] = running_var

    return res


def norm_for_test(x, gamma, beta, params):
    running_mean = params['running_mean']
    running_var = params['running_var']
    # mom = params['mom']

    eps = params['eps']
    # x_mean = x.mean()
    # x_var = x.var()

    # running_mean = mom * running_mean + (1- mom) * x_mean
    # running_var = mom * running_var + (1- mom) * x_var

    # 归一化
    x_norm = (x - running_mean) / np.sqrt(running_var + eps)

    res = gamma * x_norm + beta

    # prams['running_mean'] = running_mean
    # prams['running_var'] = running_var

    return res