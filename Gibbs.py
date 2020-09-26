import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats
from scipy.stats import truncnorm
import time
import csv


def get_s_cond_t_params(player_1_sigma, player_2_sigma, player_1_mean, player_2_mean, t_i, beta):

    # Covariance calculations
    covariance_t_given_s = beta
    covariance_ss = np.array([[player_1_sigma, 0], [0, player_2_sigma]])
    A = np.array([1, -1]).reshape([1, 2])
    inv_covariance_ss = np.linalg.inv(covariance_ss)
    ACA = np.matmul(A.T, A) * (1 / covariance_t_given_s)
    covariance_s_cond_t = np.linalg.inv(inv_covariance_ss + ACA)

    # Mean calculations
    player_means = np.array([player_1_mean, player_2_mean]).reshape([2, 1])
    a = covariance_s_cond_t
    b = np.matmul(inv_covariance_ss, player_means)
    c = A.T * (1 / covariance_t_given_s) * t_i
    mean_s_cond_t = np.matmul(a, b + c)

    return mean_s_cond_t, covariance_s_cond_t


def P_s_cond_t(t_i, player_1_mean, player_1_sigma, player_2_mean, player_2_sigma, beta):

    mean_s_cond_t, cov_s_cond_t = get_s_cond_t_params(player_1_mean=player_1_mean, player_1_sigma=player_1_sigma,
                                                      player_2_mean=player_2_mean, player_2_sigma=player_2_sigma,
                                                      t_i=t_i, beta=beta)

    return np.random.multivariate_normal(mean=mean_s_cond_t.reshape(2), cov=cov_s_cond_t,
                                         check_valid='warn', tol=1e-8)


def P_t_cond_s(s_i, t_game, beta):
    s_diff = s_i[0] - s_i[1]
    #beta = (25/3/2)**2
    t_sigma = beta

    if t_game > 0:  # case for when y=1
        a, b = (0 - s_diff) / t_sigma, np.inf
        t = truncnorm.rvs(a, b) * t_sigma + s_diff
        return t
    elif t_game < 0:  # case for when y=-1
        a, b = -np.inf, (0 - s_diff) / t_sigma
        t = truncnorm.rvs(a, b) * t_sigma + s_diff
        return t
    else:
        print("ERROR, TIES PRESENTLY NOT ALLOWED")


# Plots histogram and fitted normal curve
def plot_hist_normal(mu, variance, nr_of_samples, color, **kwargs):
    samples = kwargs.get('samples', list)
    if kwargs:
        plt.hist(samples, bins=50, color=color)
    sigma = np.sqrt(variance)
    x = np.linspace(mu - 3 * sigma, mu + 3 * sigma, 100)
    plt.plot(x, nr_of_samples * stats.norm.pdf(x, mu, sigma), color='black')


def make_stats_dictionary(filename, stats_dictionary, printable, mean, variance):
    with open(filename) as csvfile:
        reader = csv.DictReader(csvfile)
        list = []
        for row in reader:
            # Create dictionary (=map) with keyword 'team' and value [mean, variance]
            stats_dictionary[row['team1']] = [mean, variance]
            # Add teams and result to list
            list.append([row['team1'], row['team2'], int(row['score1']) - int(row['score2'])])

            if printable == 1:  # Print the whole list
                print(f"{row['team1']} vs {row['team2']}: {row['score1']}-{row['score2']}")  # Access by column header

    return stats_dictionary, list


# Ranking by mean (should perhaps be improved to mean - 3 * sigma)
def ranking(stats_dictionary):
    # Conservative rank
    sorted_teams = sorted(stats_dictionary.items(), key=lambda x: x[1][0] - 3*np.sqrt(x[1][1]), reverse=True)
    # Rank by mean
    # sorted_teams = sorted(stats_dictionary.items(), key=lambda x: x[1][0], reverse=True)
    print("\nList of teams ranked by mean skill in descending order:\n")
    for i in sorted_teams:
        print(i[0], i[1])


def gibbs_sampler(L, player_1_stats, player_2_stats, t_game, beta, hists):
    player_1_mean, player_1_sigma = player_1_stats
    player_2_mean, player_2_sigma = player_2_stats

    t_obs = np.zeros(L+1)
    s_obs = np.zeros([L+1, 2])
    s_obs[0] = [player_1_mean, player_2_mean]  # Initialize with player means

    for i in range(L):

        t_obs[i+1] = P_t_cond_s(s_obs[i], t_game=t_game, beta=beta)
        s_obs[i+1] = P_s_cond_t(t_obs[i+1], player_1_mean=player_1_mean, player_1_sigma=player_1_sigma,
                                player_2_mean=player_2_mean, player_2_sigma=player_2_sigma, beta=beta)

        # plt.scatter(s_obs[:, 0], s_obs[:, 1])
        # plt.pause(0.1)
    #plt.show()

    # Plots of samples by iteration number
    # plt.plot(s_obs[:, 0]); plt.plot(s_obs[:, 1]); plt.plot(t_obs[:]); plt.show()

    # Fit Gaussians to the samples using estimated mean and variance
    """ NOTE: Now only using from index start_index and on!"""
    start_index = 500  # Burn-in
    nr_of_samples = L - start_index
    # [mean, variance] of samples after burn-in
    player_1_stats_estimate = [np.mean(s_obs[start_index:, 0]), np.var(s_obs[start_index:, 0])]
    player_2_stats_estimate = [np.mean(s_obs[start_index:, 1]), np.var(s_obs[start_index:, 1])]
    t_estimate = [np.mean(t_obs[start_index:]), np.var(s_obs[start_index:])] # [mean, variance] of samples

    # Histogram and normal curve (scaled) for posterior
    if hists:
        plot_hist_normal(player_1_stats_estimate[0], player_1_stats_estimate[1],
                         nr_of_samples, color='blue', samples=s_obs[start_index:, 0])
        plot_hist_normal(player_2_stats_estimate[0], player_2_stats_estimate[1],
                         nr_of_samples, color='green', samples=s_obs[start_index:, 1])
        # plot_hist_normal(t_estimate[0], t_estimate[1], L, color='orange', samples=t_obs[:])
        plt.show()
    # plot_hist_normal(player_1_stats[0], player_1_stats[1], L, color='blue') # Normal curve for prior

    # print("s_1|y=1: mean =", player_1_stats_estimate[0], "; cov =", player_1_stats_estimate[1])
    # print("s_2|y=1: mean =", player_2_stats_estimate[0], "; cov =", player_2_stats_estimate[1])

    return player_1_stats_estimate, player_2_stats_estimate


if __name__ == '__main__':

    MEAN = 25  # TrueSkill prior parameters before any games
    VARIANCE = (25 / 3) ** 2
    BETA = 3  # VARIANCE / 4
    # Make dictionary of team stats (mean and variance) & list of all games with result
    stats_dictionary, result_list = make_stats_dictionary('SerieA.csv', stats_dictionary={}, printable=0,
                                                          mean=MEAN, variance=VARIANCE)
    print(stats_dictionary)


    # Iterate game list using Gibbs-sampler to estimate posterior for skills, using them as new priors
    for i in range(0,30):  # len(result_list)):

        # result_list holds [team1, team2, result = score1 - score2]
        print(f"\nResult game {i}:   {result_list[i]}")
        team1 = result_list[i][0]
        team2 = result_list[i][1]
        result = result_list[i][2]
        print(result_list)
        # stats_dictionary with keyword 'teamname' and value [mean, variance]
        print(f"Stats before game: {stats_dictionary[team1]}, {stats_dictionary[team2]}")

        if result == 0:  # ignore tied games for now
            print("Game ignored due to tie")
            print(f"Stats after game:  {stats_dictionary[team1]}, {stats_dictionary[team2]}")

        else:
            player_1_stats_posterior, player_2_stats_posterior = gibbs_sampler(L=5000,
                                                                               player_1_stats=stats_dictionary[team1],
                                                                               player_2_stats=stats_dictionary[team2],
                                                                               t_game=result, beta=BETA, hists=False)
            # Update team stats so posterior makes new prior
            stats_dictionary[team1] = player_1_stats_posterior
            stats_dictionary[team2] = player_2_stats_posterior
            print(f"Stats after game:  {player_1_stats_posterior}, {player_2_stats_posterior}")

    print(f"\n{stats_dictionary}")
    # ranking(stats_dictionary)
