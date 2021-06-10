import numpy as np
from utils import get_random_agent, iv_regression, ols, approx_bound, find_best_tau
from agent import Type0Agent, Type1Agent
from scipy.stats import norm, truncnorm
import math


THETA = 0.5
INIT_LEN = 1000


def initial_stage(l0, l1, threshold=1000, c0=0, c1=0.5, mu=-0.5, var=1, var_g=1):
    count = 0
    avg_gap = 0
    # Calculate the gap
    delta = 0.1  # failure probability NEED TO RECHECK THIS
    G = c1 - c0
    gap = np.sqrt(2 * np.log(2 / delta) / l0) + np.sqrt(2 * np.log(2 / delta) / l1) + G + 0.5
    # Empirically estimate y_1_bar - y_0_bar
    for j in range(threshold):
        y_0_bar = 0
        y_1_bar = 0
        count_0 = 0
        count_1 = 0
        for i in range(l0 + l1):
            agent = get_random_agent(i, None, c0, c1, var=var, var_g=var_g)
            reward = THETA*agent.action() + agent.get_g()
            if agent.get_type() == 0:
                y_0_bar += reward
                count_0 += 1
            else:
                y_1_bar += reward
                count_1 += 1
        y_0_bar = y_0_bar/count_0
        y_1_bar = y_1_bar/count_1
        actual_gap = y_1_bar - y_0_bar
        avg_gap += actual_gap
        if actual_gap >= gap:
            count += 1
    # Calculate Prob[xi]
    prob_xi = count/threshold
    print("Prob[xi]: {}".format(prob_xi))
    rho = 1 + (4*mu)/(prob_xi - 4*mu)
    print("Rho: {}".format(rho))
    avg_gap = avg_gap/threshold
    if avg_gap > gap:
        best = 1
    else:
        best = 0
    return rho, best


def sampling_stage(rho, best, alg1_len=None, phases=None):
    y_lst = []
    x_lst = []
    z_lst = []
    if alg1_len is not None:
        for i in range(alg1_len):
            tmp2 = np.random.rand()
            if tmp2 < rho:  # explore
                rec = 1
            else:  # exploit
                rec = best
            agent = get_random_agent(INIT_LEN + i, rec, c1=0.1, c0=0)
            action = agent.action()
            reward = THETA*action + agent.get_g()
            y_lst.append(reward)
            x_lst.append(action)
            z_lst.append(rec)
    elif phases is not None:
        phase_length = int(np.ceil(1/rho))
        for i in range(phases):
            explore_agent = get_random_agent(INIT_LEN + i, rec=1, c0=0, c1=0.1)  # Check this
            explore_action = explore_agent.action()
            explore_reward = THETA*explore_action + explore_agent.get_g()
            y_lst.append(explore_reward)
            x_lst.append(explore_action)
            z_lst.append(1)
            for j in range(phase_length - 1):
                agent = get_random_agent(INIT_LEN + i*j + 1, rec=best, c1=0.1, c0=0)
                action = agent.action()
                reward = THETA * action + agent.get_g()
                y_lst.append(reward)
                x_lst.append(action)
                z_lst.append(best)

    return y_lst, x_lst, z_lst


# Run after running sampling stage
def racing_stage(samples, T=100000000, P=0.5, tau_0=0.1, tau_1=0.1):
    y_lst, x_lst, z_lst = samples
    theta_hat = iv_regression(samples)
    theta_hat_ols = ols(samples)
    threshold = approx_bound(samples, delta=0.1)
    print("Threshold: {}".format(threshold))
    alg1_len = len(y_lst)
    t = INIT_LEN + alg1_len
    theta_lst = [theta_hat]
    print("Theta hat: {}".format(theta_hat))
    theta_ols_lst = [theta_hat_ols]
    mu_0 = np.random.normal(0, 1)
    mu_1 = np.random.normal(0.5, 1)
    agent_1 = Type1Agent(t, 0, c=0.5)
    # constant_1 = agent_1.get_racing_stage_constant(tau_1)
    _, constant_1 = find_best_tau(type=1)
    print("Constant 1: {}".format(constant_1))
    while np.abs(theta_hat) <= threshold or t < INIT_LEN + 1.5*alg1_len:
    # while np.abs(theta_hat - THETA) <= constant_1 :
        if t % 2 == 1:
            rec = 0
        else:
            rec = 1

        # agent = get_random_agent(i, rec, c1=0.1, c0=0)
        tmp = np.random.rand()
        if tmp < 0.5:
            action = rec
            g_0 = np.random.normal(mu_0, 1)
            reward = THETA * action + g_0
        else:
            if np.abs(theta_hat - THETA) <= constant_1:
            # if t > INIT_LEN + 1.4539*alg1_len:
                action = rec
            else:
                action = 1
            g_1 = np.random.normal(mu_1, 1)
            reward = THETA * action + g_1
        y_lst.append(reward)
        x_lst.append(action)
        z_lst.append(rec)
        t += 1
        # update theta_hat
        if t % 10000 == 0 and t > 10000:
            new_theta_hat = iv_regression((y_lst, x_lst, z_lst))
            if new_theta_hat > theta_hat:
                theta_hat = new_theta_hat
            print("Theta-hat: {} at round {}".format(theta_hat, t))
            theta_hat_ols = ols((y_lst, x_lst, z_lst))
            theta_lst.append(theta_hat)
            theta_ols_lst.append(theta_hat_ols)
            # print("Theta hat: {}".format(theta_hat))
            # update threshold
            threshold = approx_bound((y_lst, x_lst, z_lst), delta=0.1)
            print("Threshold: {}".format(threshold))

    if theta_hat >= 0:
        best = 1
    else:
        best = 0
    print("Best arm found!!")
    for i in range(t, t+100000):
        rec = best
        tmp = np.random.rand()
        if tmp < 0.5:
            action = rec
            g_0 = np.random.normal(mu_0, 1)
            reward = THETA * action + g_0
        else:
            if np.abs(theta_hat - THETA) <= constant_1:
                action = rec
            else:
                action = 1
            g_1 = np.random.normal(mu_1, 1)
            reward = THETA * action + g_1
        y_lst.append(reward)
        x_lst.append(action)
        z_lst.append(rec)
        if i % 10000 == 0 and i > 10000:
            new_theta_hat = iv_regression((y_lst, x_lst, z_lst))
            if new_theta_hat > theta_hat:
                theta_hat = new_theta_hat
            print("Theta-hat: {} at round {}".format(theta_hat, i))
            theta_hat_ols = ols((y_lst, x_lst, z_lst))
            theta_lst.append(theta_hat)
            theta_ols_lst.append(theta_hat_ols)
            # print("Theta hat: {}".format(theta_hat))

    return best, t, theta_lst, theta_ols_lst, y_lst, x_lst, z_lst


# Run as a standalone algorithm
def racing_stage_only(theta_hat, theta_hat_ols, threshold, P=0.5, tau_0=0.1, tau_1=0.1, T=100000000, t=10000000, length=None):
    # threshold = racing_stage_threshold(tau)
    y_lst = []
    x_lst = []
    z_lst = []
    theta_lst = [theta_hat]
    theta_ols_lst = [theta_hat_ols]
    mu_0 = np.random.normal(0, 1)
    mu_1 = np.random.normal(0.5, 1)
    # print("Theta_hat: {}".format(theta_hat))
    if threshold is None and length is not None:
        agent_1 = Type1Agent(t, 0, c=0.5)
        constant_1 = agent_1.get_racing_stage_constant(tau_1)
        for i in range(length):
            if i % 2 == 1:
                rec = 0
            else:
                rec = 1
            tmp = np.random.rand()
            if tmp < 0.5:
                action = rec
                g_0 = np.random.normal(mu_0, 1)
                reward = THETA * action + g_0
            else:
                if i > constant_1:
                    action = rec
                else:
                    action = 1
                g_1 = np.random.normal(mu_1, 1)
                reward = THETA * action + g_1
            y_lst.append(reward)
            x_lst.append(action)
            z_lst.append(rec)
            # if i % 1000 == 0 and i > 1000:
                # theta_hat = iv_regression((y_lst, x_lst, z_lst))
                # theta_ols = ols((y_lst, x_lst, z_lst))
                # theta_lst.append(np.abs(theta_hat - THETA))
                # theta_ols_lst.append(np.abs(theta_ols - THETA))
                # print("t: {}, theta_hat: {}, theta_ols: {}".format(i, theta_hat, theta_ols))

        if theta_hat >= 0:
            best = 1
        else:
            best = 0
        return best, t, theta_lst, theta_ols_lst, y_lst, x_lst, z_lst
    for i in range(10000):
        if i % 2 == 1:
            rec = 0
        else:
            rec = 1

        tmp = np.random.rand()
        if tmp < 0.5:  # type 0
            action = rec
            g_0 = np.random.normal(mu_0, 1)
            reward = THETA * action + g_0
        else:
            action = 1
            g_1 = np.random.normal(mu_1, 1)
            reward = THETA * action + g_1

        y_lst.append(reward)
        x_lst.append(action)
        z_lst.append(rec)
    print("Going into racing stage")
    threshold = approx_bound((y_lst, x_lst, z_lst))
    print("Threshold: {}".format(threshold))
    print("Theta hat: {}".format(np.abs(theta_hat)))
    while np.abs(theta_hat) <= threshold:
    # while (t < 1000000):
        if t % 2 == 1:
            rec = 0
        else:
            rec = 1
        tmp = np.random.rand()
        if tmp < 0.5:  # type 0
            action = rec
            g_0 = np.random.normal(mu_0, 1)
            reward = THETA * action + g_0
        else:
            action = 1
            g_1 = np.random.normal(mu_1, 1)
            reward = THETA * action + g_1
        y_lst.append(reward)
        x_lst.append(action)
        z_lst.append(rec)
        t += 1
        if t % 1000 == 0:
            new_theta_hat = iv_regression((y_lst, x_lst, z_lst))
            if np.abs(new_theta_hat) > np.abs(theta_hat):
                theta_hat = new_theta_hat
            theta_ols = ols((y_lst, x_lst, z_lst))
            theta_lst.append(theta_hat)
            theta_ols_lst.append(theta_ols)
            threshold = approx_bound((y_lst, x_lst, z_lst), delta=0.1)
        if t % 1000 == 0:
            print("t: {}, new_theta_hat: {} theta_hat: {}, threshold: {}".format(t, new_theta_hat, theta_hat, threshold))
    print("Racing Stage End!!")
    if theta_hat >= 0:
        best = 1
    else:
        best = 0
    print("Best arm overall: {}".format(best))

    for i in range(t, t + 100000):
        rec = best
        agent = get_random_agent(i, rec, c0=0, c1=0.1)
        action = agent.action()
        reward = THETA * action + agent.get_g()
        y_lst.append(reward)
        x_lst.append(action)
        z_lst.append(rec)
        if i % 1000 == 0:
            new_theta_hat = iv_regression((y_lst, x_lst, z_lst))
            if np.abs(new_theta_hat) > np.abs(theta_hat):
                theta_hat = new_theta_hat
            theta_ols = ols((y_lst, x_lst, z_lst))
            theta_lst.append(theta_hat)
            theta_ols_lst.append(theta_ols)

    return best, t, theta_lst, theta_ols_lst, y_lst, x_lst, z_lst


def racing_stage_threshold(tau=0.1, type=0, rho=0.001):
    if type == 0:
        agent = Type0Agent(-1, rec=None)
    else:
        agent = Type1Agent(-1, rec=None)
    threshold = agent.get_racing_stage_constant(tau, rho=rho)
    print("Threshold: {}".format(threshold))
    return threshold


def faster_sampling_stage(rho, best, alg1_len, threshold=None):
    y_lst = []
    x_lst = []
    z_lst = []
    mu_0 = np.random.normal(0, 1)
    mu_1 = np.random.normal(0.5, 1)
    for i in range(alg1_len):
        tmp2 = np.random.rand()
        if tmp2 < rho:  # explore
            rec = 1
        else:  # exploit
            rec = best
        # agent = get_random_agent(INIT_LEN + i, rec, c1=0.1, c0=0)
        tmp = np.random.rand()
        if tmp < 0.5: # type 0
            action = rec
            g_0 = np.random.normal(mu_0, 1)
            reward = THETA*action + g_0
        else:
            action = 1
            g_1 = np.random.normal(mu_1, 1)
            reward = THETA*action + g_1
        y_lst.append(reward)
        x_lst.append(action)
        z_lst.append(rec)
        if threshold is not None and i >= 100000 and i % 100000 == 0:
            # bound = approx_bound((y_lst, x_lst, z_lst))
            bound = np.abs(iv_regression((y_lst, x_lst, z_lst)) - 0.5)
            print("Bound: {}, threshold: {}".format(bound, threshold))
            if bound <= threshold:
                print("Guaranteed BIC for racing stage")
                break
        if i % 100000 == 0:
            print("i: {}".format(i))
    return y_lst, x_lst, z_lst
