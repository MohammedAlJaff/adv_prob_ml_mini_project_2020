
import numpy as np
from scipy.stats import truncnorm
from scipy import stats
import q7_8lib
import q4_6lib
import matplotlib.pyplot as plt


if __name__ == "__main__":
    
      ## Q4.  ------------------------
    nr_of_samples = 50000

    player_1_mean = 100
    player_1_var = 10
    player_1_stats = [player_1_mean, player_1_var]

    player_2_mean = 100
    player_2_var = 10
    player_2_stats = [player_2_mean, player_2_var]

    match_outcome = 1 # player 1 wins
    
    #Q.4a
    burn_in_indx = 3000
    nr_samples_q4c = 15000

    s_obs, t_obs = q4_6lib.gibbs_sampler(L=nr_samples_q4c,
                                player_1_stats=player_1_stats,
                                player_2_stats=player_2_stats,
                                t_game=match_outcome)

    s_obs = s_obs[burn_in_indx:, :] # discard inital B samples due to burn in. 

    player_1_stats_estimate_4, player_2_stats_estimate = q4_6lib.player_stats_estimate_from_obs(s_obs=s_obs[burn_in_indx:burn_in_indx+10000, :])


    p1_mu_est_4 = player_1_stats_estimate_4[0]
    p1_var_est_4 = player_1_stats_estimate_4[1]
    p1_sigma_est_4 = np.sqrt(p1_var_est_4)

    ## Q7.  ------------------------
    ## Q8.  ------------------------
    print('PERFORMING Q8 COMPUTATIONS AND RESULTS')
    # Run message-passing with skill priors for team1 and team2, beta, and y as input parameters
    s1_posterior_mean, s1_posterior_variance, s2_posterior_mean, s2_posterior_variance = q7_8lib.message_passing(s1_prior_mean= 100, s1_prior_variance=10**2, s2_prior_mean= 100, s2_prior_variance=10**2, beta = 3, y_observed= 1)


    # Stats for p(s2|y=1) Gaussian from Gibbs
    p2_mu_est_4 = player_2_stats_estimate[0]
    p2_var_est_4 = player_2_stats_estimate[1]
    p2_sigma_est_4 = np.sqrt(p1_var_est_4)

    # Plot of histograms for both players from gibbs, both gaussians from gibbs and both gaussians from message-passing
    plt.figure(figsize=[10, 10])
    plt.hist(s_obs[burn_in_indx:, 0], bins=50, density=True)  # P(S1 \Y=1)
    plt.hist(s_obs[burn_in_indx:, 1], bins=50, density=True)  # P(S2 \Y=1)

    x1 = np.linspace(p1_mu_est_4 - 4 * p1_sigma_est_4, p1_mu_est_4 + 4 * p1_sigma_est_4, 100)
    x2 = np.linspace(p2_mu_est_4 - 4 * p2_sigma_est_4, p2_mu_est_4 + 4 * p2_sigma_est_4, 100)

    plt.plot(x1, stats.norm.pdf(x1, loc=p1_mu_est_4, scale=p1_sigma_est_4), label='$p(s_{1}|y=1)$ from gibbs')
    plt.plot(x2, stats.norm.pdf(x2, loc=p2_mu_est_4, scale=p2_sigma_est_4), label='$p(s_{2}|y=1)$ from gibbs')

    plt.plot(x1, stats.norm.pdf(x1, loc=s1_posterior_mean, scale=np.sqrt(s1_posterior_variance)),
             label='$p(s_{1}|y=1)$ from m-p')
    plt.plot(x2, stats.norm.pdf(x2, loc=s2_posterior_mean, scale=np.sqrt(s2_posterior_variance)),
             label='$p(s_{2}|y=1)$ from m-p')

    plt.title('comparing $p(s_{1})$ and $p(s_{2})$ from Gibbs sampling and message-passing')
    plt.xlabel('probability density')
    plt.xlabel('skills')
    plt.legend()
    plt.show()
