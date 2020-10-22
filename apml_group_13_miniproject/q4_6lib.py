# AUTHORS
# - Mohammed Al-Jaff
# - Joel Olofsson
# - Carl Öhrnell

# COURSE
# Advanced Probabilistic Machine Learning 
# 5hp, -ht20
# Uppsala University 

import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats
from scipy.stats import truncnorm
import time
import csv


def get_s_cond_t_params(player_1_sigma, player_2_sigma, player_1_mean, player_2_mean, t_i):
    beta = 3  # från uppgiften

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


def P_s_cond_t(t_i, player_1_mean, player_1_sigma, player_2_mean, player_2_sigma):
    # player_1_mean = 25; player_2_mean = 25; player_1_sigma = 8.3**2; player_2_sigma = 8.3**2

    mean_s_cond_t, cov_s_cond_t = get_s_cond_t_params(player_1_mean=player_1_mean, player_1_sigma=player_1_sigma,
                                                      player_2_mean=player_2_mean, player_2_sigma=player_2_sigma,
                                                      t_i=t_i)

    return np.random.multivariate_normal(mean=mean_s_cond_t.reshape(2), cov=cov_s_cond_t,
                                         check_valid='warn', tol=1e-8)

def P_t_cond_s(s_i, t_game):
    s_diff = s_i[0] - s_i[1]
    beta = 3
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


        
def gibbs_sampler(L, player_1_stats, player_2_stats, t_game):
    player_1_mean, player_1_sigma = player_1_stats
    player_2_mean, player_2_sigma = player_2_stats
    
    s_i = [player_1_mean, player_2_mean]

    t_obs = np.zeros(L)
    s_obs = np.zeros([L, 2])

    for i in range(L):
        t_i_plus_1 = P_t_cond_s(s_i, t_game=t_game)
        s_i_plus_1 = P_s_cond_t(t_i_plus_1, player_1_mean, player_1_sigma, player_2_mean, player_2_sigma)

        t_obs[i] = t_i_plus_1
        s_obs[i, :] = s_i_plus_1

        s_i = s_i_plus_1
        # plt.scatter(s_obs[:, 0], s_obs[:, 1])
        # plt.pause(0.1)
    # plt.show()

    # plt.plot(s_obs[:, 0]); plt.plot(s_obs[:, 1]); plt.show()
    return s_obs, t_obs



def player_stats_estimate_from_obs(s_obs):
    player_1_stats_estimate = [np.mean(s_obs[:, 0]), np.var(s_obs[:, 0])] # [mean, variance] of samples
    player_2_stats_estimate = [np.mean(s_obs[:, 1]), np.var(s_obs[:, 1])] # [mean, variance] of samples
    
    return player_1_stats_estimate, player_2_stats_estimate


    

def make_stats_dictionary(filename, stats_dictionary, printable):
    with open(filename) as csvfile:
        reader = csv.DictReader(csvfile)
        mean = 100; variance = (15)**2  # TrueSkill prior parameters before any games
        gamelist = []
        for row in reader:
            # Create dictionary (=map) with keyword 'team' and value [mean, variance]
            stats_dictionary[row['team1']] = [mean, variance]
            # Add teams and result to list
            gamelist.append([row['team1'], row['team2'], int(row['score1']) - int(row['score2'])])

            if printable == 1:  # Print the whole list
                print(f"{row['team1']} vs {row['team2']}: {row['score1']}-{row['score2']}")  # Access by column header
                      
    #stats_dictionary = random.shuffle(stats_dictionary, random)

    return stats_dictionary, gamelist



# Ranking by mean (should perhaps be improved to mean - 3 * sigma)
def ranking(stats_dictionary):
    sorted_teams = sorted(stats_dictionary.items(), key=lambda x: x[1], reverse=True)
    print("\nList of teams ranked by mean skill in descending order:\n")
    for i in sorted_teams:
        print(i[0], i[1])

        
def predict_winner(team1_string, team2_string, stats_dictionary):
    diff = stats_dictionary[team1_string][0] - stats_dictionary[team2_string][0] 
    #print('statdic team1 : ', stats_dictionary[team1_string][0])
    #print('statdic team2 : ', stats_dictionary[team2_string][0])
    print('\t diff: ', diff)
    
    if diff > 0:
        return 1
    elif diff < 0:
        return -1
    else:
        return 1
            
