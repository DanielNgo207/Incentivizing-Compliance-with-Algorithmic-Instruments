import numpy as np
from scipy.stats import norm, truncnorm
import matplotlib.pyplot as plt
import pickle
from algorithm import sampling_stage, racing_stage, initial_stage, racing_stage_threshold, racing_stage_only, faster_sampling_stage
from utils import iv_regression, ols, calc_average, combine_samples, find_best_tau
import math


def ex_post_regret(arm_lst, best_arm):
    regret = 0
    regret_lst = []
    for i in arm_lst:
        if i != best_arm:
            regret += np.abs(THETA*best_arm - THETA*i)
            regret_lst.append(regret)
        else:
            regret_lst.append(regret)
    # Include the initial stage
    regret += INIT_LEN/2 * THETA
    return regret, regret_lst


def accuracy_plot(theta_lst, theta_ols_lst):
    time_ax = range(len(theta_lst))
    line, = plt.plot(theta_lst, time_ax, '-g')
    line.set_label('IV-Regression')
    line2, = plt.plot(theta_ols_lst, time_ax, '-b')
    line2.set_label('OLS')
    plt.legend(loc='best')
    plt.show()


# Fix the gap, the prior means, truncated gaussian, fix everything except the variance in prior over theta, plot how the
# probability xi is increasing with larger variance, Check that the agents are still preferring their arm (always-taker
# and never-takers).
# Report the variance
def new_rho_experiment():
    global THETA
    global P
    global INIT_LEN
    global ALG1_LEN
    global T
    THETA = 0.5
    P = 0.5
    INIT_LEN = 1000
    ALG1_LEN = 10000  # hyperparameter
    T = 1000000
    c1_lst = [0, 0.25, 0.5, 0.75, 1]
    gap = 0.5   # hyperparameter
    # var_lst = np.arange(0.1, 1, .1)
    var_lst = np.arange(0.8, 2, 0.2)
    # c1_val = 0.5
    i = 0
    for c1_val in c1_lst:
        rho_lst = []
        for var_val in var_lst:
            rho, _ = initial_stage(500, 500, threshold=1000, c0=c1_val - gap, c1=c1_val, mu=-0.5, var_g=var_val)
            rho_lst.append(rho)
            print("C1: {}, C0: {}, G: {}, Rho: {}".format(c1_val, c1_val - gap, gap, rho))
        i += 1
        plt.plot(var_lst, rho_lst)
        plt.xlabel('variance in prior over g(u) with var: {}'.format(var_val))
        plt.ylabel('rho')
        plt.savefig('g_prior_var_rho_experiment_{}'.format(i))
        plt.show()


# Plot in number of phases of exploration instead of number of total rounds
def new_sampling_stage_experiment(idx):
    global THETA
    global P
    global INIT_LEN
    global T
    THETA = 0.5
    P = 0.5
    INIT_LEN = 1000
    T = 1000000
    rho, best = initial_stage(500, 500, threshold=1000, c0=0, c1=0.1, mu=-0.5)      # hyperparameter
    if rho == 0:
        rho = np.random.uniform(low=0.001, high=0.03)
    tau = 0.1
    num_run = 5
    # phases_lst = [500, 1000, 1500]
    # phases_lst = np.arange(500, 5500, 500)
    phases_lst = np.arange(200000, 2200000, 200000)
    avg_approx = [[] for i in range(len(phases_lst))]
    avg_ols = [[] for i in range(len(phases_lst))]
    threshold = racing_stage_threshold(tau)
    for run in range(num_run):
        sample_lst = []
        prev_phase = 0
        for i, num_phases in enumerate(phases_lst):
            # samples = sampling_stage(rho, best, phases=num_phases-prev_phase)
            samples = sampling_stage(rho, best, alg1_len=num_phases-prev_phase)
            prev_phase = num_phases
            sample_lst = combine_samples(sample_lst, samples)
            theta_hat = iv_regression(sample_lst)
            approximation_bound = np.abs(theta_hat - THETA)
            theta_ols = ols(samples)
            ols_bound = np.abs(theta_ols - THETA)
            print("Approximation bound: {}".format(approximation_bound))
            print("OLS bound: {}".format(ols_bound))
            avg_approx[i].append(approximation_bound)
            avg_ols[i].append(ols_bound)
            if approximation_bound <= threshold:
                print("Sampling Stage Length: {}, reach BIC in racing stage".format(num_phases))
            else:
                print("Sampling Stage Length: {}, does not reach BIC in racing stage".format(num_phases))
    # pickle
    filename = 'sampling_stage_iv_'+str(idx)
    outfile = open(filename, 'wb')
    pickle.dump(avg_approx, outfile)
    outfile.close()

    filename_ols = 'sampling_stage_ols_'+str(idx)
    outfile = open(filename_ols, 'wb')
    pickle.dump(avg_ols, outfile)
    outfile.close()

    filename_rho = 'rho_' + str(idx)
    outfile = open(filename_rho, 'wb')
    pickle.dump(rho, outfile)
    outfile.close()
    return phases_lst, avg_approx, avg_ols, rho


def plot(phases_lst, avg_approx, avg_ols, i=100, rho=0.003, tau=0.1, stage='racing'):
    avg, max_error, min_error = calc_average(avg_approx, n=5)
    avg_ols, max_error_ols, min_error_ols = calc_average(avg_ols, n=5)
    plt.plot(phases_lst, avg, label='IV-Regression treatment effect estimation error', color='blue')
    plt.plot(phases_lst, avg_ols, label='OLS treatment effect estimation error', color='red')
    plt.yscale('log')
    plt.fill_between(phases_lst, min_error, max_error, alpha=0.3, facecolor='blue')
    plt.fill_between(phases_lst, min_error_ols, max_error_ols, alpha=0.3, facecolor='red')
    if stage == 'racing':
        plt.xlabel('Number of rounds in Algorithm 2 with tau={:.4f}'.format(tau))
    else:
        plt.xlabel('Number of rounds in Algorithm 1 with rho={:.4f}'.format(rho))
    plt.ylabel('Approximation Error')
    plt.legend(loc='best')
    if stage == 'racing':
        plt.savefig('racing_stage_experiment_{}'.format(i), bbox_inches="tight")
    else:
        plt.savefig('new_sampling_stage_experiment_phase_tauval_{}'.format(i), bbox_inches="tight")

    plt.show()


# standalone racing stage algorithm
def racing_stage_experiment_prepartion(idx, length=None):
    global THETA
    global P
    global INIT_LEN
    global T
    THETA = 0.5
    P = 0.5
    INIT_LEN = 1000
    T = 1000000
    tau_0, constant_0 = find_best_tau(type=0)
    tau_1, constant_1 = find_best_tau(type=1)
    print("Tau for type 0: {}".format(tau_0))
    print("Tau for type 1: {}".format(tau_1))
    print("Constant for type 0: {}".format(constant_0))
    if length is None:
        sampling_stage_len = math.ceil(racing_stage_threshold(tau_0, type=0, rho=0.001))
        print("Sampling stage length: {}".format(sampling_stage_len))
    else:
        sampling_stage_len = length
        print("Sampling stage length: {}".format(sampling_stage_len))
    samples = faster_sampling_stage(0.001, 0, alg1_len=sampling_stage_len, threshold=constant_0)
    filename_sample = 'sampling_stage_sample_tau_' + str(idx)
    outfile = open(filename_sample, 'wb')
    pickle.dump((samples, tau_0, tau_1), outfile)
    outfile.close()
    print("Sampling stage done")

    return samples, tau_0, tau_1


def racing_stage_experiment(samples, tau_0, tau_1, idx, num_run=5):
    theta_hat = iv_regression(samples)
    theta_hat_ols = ols(samples)
    y_lst, x_lst, z_lst = samples
    n = len(y_lst)
    print("Length of sampling stage: {}".format(n))
    phases_lst = np.arange(200000, 2200000, 200000)
    avg_approx = [[] for i in range(len(phases_lst))]
    avg_ols = [[] for i in range(len(phases_lst))]

    for i in range(num_run):
        prev_phase = 0
        sample_lst = []
        for j, length in enumerate(phases_lst):
            # just running the racing stage without any while condition
            best, t, theta_lst, theta_ols_lst, y_lst, x_lst, z_lst = racing_stage_only(theta_hat, theta_hat_ols, threshold=None, length=length-prev_phase, tau_0=tau_0, tau_1=tau_1)

            prev_phase = length
            sample_lst = combine_samples(sample_lst, (y_lst, x_lst, z_lst))
            theta_hat = iv_regression(sample_lst)
            approximation_bound = np.abs(theta_hat - THETA)
            theta_ols = ols(samples)
            ols_bound = np.abs(theta_ols - THETA)
            print("Approximation bound: {}".format(approximation_bound))
            print("OLS bound: {}".format(ols_bound))
            avg_approx[j].append(approximation_bound)
            avg_ols[j].append(ols_bound)
            print("Best arm overall: {}".format(best))

        filename_rho = 'racing_stage_sample_{}_{}'.format(i, idx)
        outfile = open(filename_rho, 'wb')
        samples = (y_lst, x_lst, z_lst)
        pickle.dump(samples, outfile)
        outfile.close()
    return phases_lst, avg_approx, avg_ols, tau_0


def plot_racing_stage(idx=1):
    infile = open('racing_stage_complete_{}'.format(idx), 'rb')
    (y_lst, x_lst, z_lst) = pickle.load(infile)
    infile.close()
    infile = open('sampling_stage_sample_tau_' + str(idx), 'rb')
    _, tau_0, tau_1 = pickle.load(infile)
    infile.close()

    n = len(y_lst)
    theta_lst = []
    theta_ols_lst = []
    for i in range(20000, n, 20000):
        y_sample, x_sample, z_sample = y_lst[:i], x_lst[:i], z_lst[:i]
        theta_hat = iv_regression((y_sample, x_sample, z_sample))
        theta_ols = ols((y_sample, x_sample, z_sample))
        theta_lst.append(np.abs(theta_hat - THETA))
        theta_ols_lst.append(np.abs(theta_ols - THETA))

    x_axis = range(20000, n, 20000)
    plt.plot(x_axis, theta_lst, label='IV-Regression treatment effect estimation error', color='blue')
    plt.plot(x_axis, theta_ols_lst, label='OLS treatment effect estimation error', color='red')
    plt.xlabel('Number of rounds in Algorithm 2 with tau={:.4f}'.format(tau_0))
    plt.ylabel('Approximation Error')
    plt.legend(loc='best')
    plt.savefig('racing_stage_experiment_complete_{}'.format(idx), bbox_inches="tight")
    plt.show()


def plot_regret(idx=1):
    infile = open('racing_stage_complete_{}'.format(idx), 'rb')
    (y_lst, x_lst, z_lst) = pickle.load(infile)
    infile.close()
    regret_lst = []
    regret = 0
    for i in range(len(x_lst)):
        tmp = THETA*np.abs(x_lst[i] - 1)
        regret += tmp
        regret_lst.append(regret)
    x_axis = range(len(x_lst))
    plt.plot(x_axis, regret_lst, label='Regret for combined Algorithm 1 + Algorithm 2')
    plt.xlabel('Total number of rounds in the Algorithm 1 + Algorithm 2')
    plt.ylabel('Regret Accumulated')
    plt.legend(loc='best')
    plt.savefig('regret_{}'.format(idx), bbox_inches='tight')
    plt.show()


def main(idx=1, alg1_len=None):
    racing_stage_experiment_prepartion(idx, length=alg1_len)
    infile = open('sampling_stage_sample_tau_' + str(idx), 'rb')
    samples, tau_0, tau_1 = pickle.load(infile)
    infile.close()
    best, t, theta_lst, theta_ols_lst, y_lst, x_lst, z_lst = racing_stage(samples, tau_0=tau_0, tau_1=tau_1)
    print("Best arm overall: {}".format(best))
    print("End phase: {}".format(t))
    filename_rho = 'racing_stage_complete_{}'.format(idx)
    outfile = open(filename_rho, 'wb')
    samples = (y_lst, x_lst, z_lst)
    pickle.dump(samples, outfile)
    outfile.close()


if __name__ == "__main__":
    main()
