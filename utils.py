import numpy as np
from agent import Type1Agent, Type0Agent
from scipy.stats import truncnorm


def get_random_agent(i, rec, c0, c1, var=1, var_g=1, P=0.5):
    tmp = np.random.rand()
    if tmp > P:
        agent = Type1Agent(i, rec, c=c1, var=var, var_g=var_g)
    else:
        agent = Type0Agent(i, rec, c=c0, var=var, var_g=var_g)
    return agent


def iv_regression(samples):
    y_lst, x_lst, z_lst = samples
    # print(len(y_lst))
    y_bar = np.mean(y_lst)
    x_bar = np.mean(x_lst)
    z_bar = np.mean(z_lst)
    # print("y-bar: {}, x-bar: {}, z-bar: {}".format(y_bar, x_bar, z_bar))
    num = 0
    denom = 0
    for i in range(len(y_lst)):
        num += (y_lst[i] - y_bar)*(z_lst[i] - z_bar)
        denom += (x_lst[i] - x_bar)*(z_lst[i] - z_bar)
    theta_hat = num/denom
    return theta_hat


def ols(samples):
    y_lst, x_lst, z_lst = samples
    y_bar = np.mean(y_lst)
    x_bar = np.mean(x_lst)
    num = 0
    denom = 0
    for i in range(len(y_lst)):
        num += (y_lst[i] - y_bar) * (x_lst[i] - x_bar)
        denom += (x_lst[i] - x_bar) ** 2
    theta_hat = num/denom
    return theta_hat


def combine_samples(old_sample, new_sample):
    if len(old_sample) > 0:
        y_lst, x_lst, z_lst = old_sample
    else:
        y_lst, x_lst, z_lst = [], [], []
    y_n, x_n, z_n = new_sample
    y = y_lst + y_n
    x = x_lst + x_n
    z = z_lst + z_n
    # print(len(y))
    return y, x, z


def calc_average(avg_lst, n=5):
    avg = [np.mean(avg_lst[i]) for i in range(len(avg_lst))]
    std = [np.std(avg_lst[i])/n for i in range(len(avg_lst))]
    max_error = [np.add(avg[i], std[i]) for i in range(len(avg_lst))]
    min_error = [np.subtract(avg[i], std[i]) for i in range(len(avg_lst))]
    # print(avg)
    # print(max_error)
    # print(min_error)
    return avg, max_error, min_error


def approx_bound(samples, delta=0.1):
    y_lst, x_lst, z_lst = samples
    x_bar = np.mean(x_lst)
    z_bar = np.mean(z_lst)
    n = len(y_lst)
    num = 2*np.sqrt(2*n*np.log(2/delta))
    denom = 0
    for i in range(n):
        denom += (x_lst[i] - x_bar)*(z_lst[i] - z_bar)
    denom = np.abs(denom)
    apprx = num/denom
    return apprx


def find_best_tau(type=0):
    tau_lst = np.arange(0.1, 1, 0.01)
    best = 0
    best_tau = 0
    if type == 0:
        agent = Type0Agent(-1, None)
    else:
        agent = Type1Agent(-1, None)
    alpha = (-1 - agent.get_mu()) / agent.get_std()
    beta = (1 - agent.get_mu()) / agent.get_std()
    dist = truncnorm(alpha, beta, loc=agent.get_mu(), scale=agent.get_std())
    for tau in tau_lst:
        if type == 0:
            tmp = tau*(1 - dist.cdf(tau))
        else:
            tmp = tau*(dist.cdf(-tau))
        if tmp > best:
            best = tmp
            best_tau = tau
    return best_tau, best

