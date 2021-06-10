import numpy as np
from scipy.stats import truncnorm


INIT_LEN = 1000
ALG1_LEN = 100000000
T = 10000000000


class Type1Agent():
    def __init__(self, t, rec, alg1_start=INIT_LEN, alg2_start=INIT_LEN+ALG1_LEN, mu=0.9, c=0.5, var=1, var_g=1):
        # self.theta = np.random.normal(mu, 1)
        self.mu = mu
        self.var = var
        self.var_g = var_g
        self.std = np.sqrt(self.var)
        alpha = (-1 - self.mu)/self.std
        beta = (1 - self.mu)/self.std
        self.dist = truncnorm(alpha, beta, loc=self.mu, scale=self.std)
        self.theta = truncnorm.rvs(alpha, beta, loc=self.mu, scale=self.std)
        mu_g = np.random.normal(c, var_g)
        self.g = np.random.normal(mu_g, var_g)
        self.t = t
        self.alg1_start = alg1_start
        self.alg2_start = alg2_start
        self.rec = rec
        self.tau = self.find_best_tau()

    def action(self):
        if self.t < self.alg1_start is None:
            # free to pick arm
            return 1
        elif self.get_racing_stage_constant(self.tau) <= self.t:
            # racing stage
            return self.rec
        else:   # sampling stage
            return 1

    def get_g(self):
        return self.g

    def get_mu(self):
        return self.theta

    def get_type(self):
        return 1

    def get_std(self):
        return self.std

    def get_racing_stage_constant(self, tau, delta=0.1, p=0.5, rho=0.001):
        prob = 8*self.var_g*np.sqrt(2*np.log(5/delta))/(tau*(self.dist.cdf(-tau))*p*rho*(1 - rho))
        prob += (3 - rho)*np.sqrt(rho*np.log(5/delta)/(2*(1 - rho)))
        prob = prob**2
        return prob

    def find_best_tau(self):
        tau_lst = np.arange(0.1, 1, 0.01)
        best = 0
        best_tau = 0

        for tau in tau_lst:
            tmp = tau * (self.dist.cdf(-tau))
            if tmp > best:
                best = tmp
                best_tau = tau
        return best_tau


class Type0Agent():
    def __init__(self, t, rec, alg1_start=INIT_LEN, alg2_start=INIT_LEN+ALG1_LEN, mu=-0.5, c=0, var=1, var_g=1):
        # self.theta = np.random.normal(mu, 1)
        self.mu = mu
        self.var = var
        self.var_g = var_g
        self.std = np.sqrt(self.var)
        alpha = (-1 - self.mu) / self.std
        beta = (1 - self.mu) / self.std
        self.dist = truncnorm(alpha, beta, loc=self.mu, scale=self.std)
        self.theta = truncnorm.rvs(alpha, beta, loc=self.mu, scale=self.std)
        mu_g = np.random.normal(c, var_g)
        self.g = np.random.normal(mu_g, var_g)
        self.t = t
        self.alg1_start = alg1_start
        self.alg2_start = alg2_start
        self.rec = rec
        self.tau = self.find_best_tau()

    def action(self):
        if self.t < self.alg1_start or self.rec is None:
            # free to pick arm
            return 0
        elif self.get_racing_stage_constant(self.tau) <= self.t:
            # racing stage
            return self.rec
        else:   # sampling stage
            return self.rec

    def get_g(self):
        return self.g

    def get_mu(self):
        return self.theta

    def get_type(self):
        return 0

    def get_std(self):
        return self.std

    def get_racing_stage_constant(self, tau, delta=0.1, p=0.5, rho=0.001):
        prob = 8*self.var_g*np.sqrt(2*np.log(5/delta))/(tau*(1 - self.dist.cdf(tau))*p*rho*(1 - rho))
        prob += (3 - rho)*np.sqrt(rho*np.log(5/delta)/(2*(1 - rho)))
        prob = prob**2
        return prob

    def find_best_tau(self):
        tau_lst = np.arange(0.1, 1, 0.01)
        best = 0
        best_tau = 0

        for tau in tau_lst:
            tmp = tau * (1 - self.dist.cdf(tau))
            if tmp > best:
                best = tmp
                best_tau = tau
        return best_tau
