import numpy as np
from scipy.stats import truncnorm

def message_passing(s1_prior_mean, s1_prior_variance, s2_prior_mean, s2_prior_variance, beta, y_observed):
    # Script for computing p(s|y) using message passing with moment-matching
    # Adapted from the Solution to Exercise 7.1 of the Advanced Probabilistic Machine Learning course,
    # Uppsala University, Version: September 18, 2020

    def mutiplyGauss(m1, s1, m2, s2):
        # computes the Gaussian distribution N(m,s) being proportional to N(m1,s1)*N(m2,s2)
        # m1, s1: mean and variance of first Gaussian  # m2, s2: mean and variance of second Gaussian
        # m, s: mean and variance of the product Gaussian
        s = 1/(1/s1+1/s2)
        m = (m1/s1+m2/s2)*s
        return m, s

    def divideGauss(m1, s1, m2, s2):
        # computes the Gaussian distribution N(m,s) being proportional to N(m1,s1)/N(m2,s2)
        # m1, s1: mean and variance of the numerator Gaussian  # m2, s2: mean and variance of the denominator Gaussian
        # m, s: mean and variance of the quotient Gaussian
        m, s = mutiplyGauss(m1, s1, m2, -s2)
        return m, s

    def truncGaussMM(a, b, m0, s0):
        # computes the mean and variance of a truncated Gaussian distribution
        # a, b: The interval [a, b] on which the Gaussian is being truncated
        # m0,s0: mean and variance of the Gaussian which is to be truncated
        # m, s: mean and variance of the truncated Gaussian
        a_scaled, b_scaled = (a - m0) / np.sqrt(s0), (b - m0) / np.sqrt(s0)
        m = truncnorm.mean(a_scaled, b_scaled, loc=m0, scale=np.sqrt(s0))
        s = truncnorm.var(a_scaled, b_scaled, loc=m0, scale=np.sqrt(s0))
        return m, s

    # Message mu1 from factor f1 and then also from s1 to f3
    m1_mean = s1_prior_mean  # mean of message
    m1_variance = s1_prior_variance  # variance of message

    # Message mu2 from factor f1 and then also from s2 to f3
    m2_mean = s2_prior_mean  # mean of message
    m2_variance = s2_prior_variance  # variance of message

    # Message mu3 from factor f3 to node t (using corollary 2)
    m3_mean = m1_mean - m2_mean  # mean of message
    m3_variance = m1_variance + m2_variance + beta  # variance of message

    # Do moment matching of the marginal of t
    if y_observed == 1:
        a, b = 0, np.inf
    else:
        a, b = -np.inf, 0
    t_marginal_mean, t_marginal_variance = truncGaussMM(a, b, m3_mean, m3_variance)

    # Message mu6 from t to f3
    m6_mean, m6_variance = divideGauss(t_marginal_mean, t_marginal_variance, m3_mean, m3_variance)

    # Message mu7 from f3 to s1
    m7_mean = m2_mean + m6_mean
    m7_variance = m2_variance + m6_variance + beta

    # Message mu8 from f3 to s2
    m8_mean = m1_mean - m6_mean
    m8_variance = m1_variance + m6_variance + beta

    # Compute the marginal of s1
    s1_posterior_mean, s1_posterior_variance = mutiplyGauss(m1_mean, m1_variance, m7_mean, m7_variance)
    print(s1_posterior_mean)
    print(s1_posterior_variance)

    # Compute the marginal of s2
    s2_posterior_mean, s2_posterior_variance = mutiplyGauss(m2_mean, m2_variance, m8_mean, m8_variance)
    print(s2_posterior_mean)
    print(s2_posterior_variance)

    return s1_posterior_mean, s2_posterior_variance, s2_posterior_mean, s2_posterior_variance


