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

import q4_6lib
import q7_8lib
import q9lib
import q10lib



if  __name__ == '__main__':
    print('Q4 COMPUTATIONS AND RESULTS....')

    ## Q4.  ------------------------
    nr_of_samples = 50000

    player_1_mean = 100
    player_1_var = 10
    player_1_stats = [player_1_mean, player_1_var]

    player_2_mean = 100
    player_2_var = 10
    player_2_stats = [player_2_mean, player_2_var]

    match_outcome = 1 # player 1 wins

    s_obs_1, t_obs_1 = q4_6lib.gibbs_sampler(L=nr_of_samples,
                                player_1_stats=player_1_stats,
                                player_2_stats=player_2_stats,
                                t_game=match_outcome)
    print('1 done')

    s_obs_2, t_obs_2 = q4_6lib.gibbs_sampler(L=nr_of_samples,
                                player_1_stats=player_1_stats,
                                player_2_stats=player_2_stats,
                                t_game=match_outcome)
    print('2 done')

    s_obs_3, t_obs_3 = q4_6lib.gibbs_sampler(L=nr_of_samples,
                                player_1_stats=player_1_stats,
                                player_2_stats=player_2_stats,
                                t_game=match_outcome)
    print('3 done')



    #plotting mean of all observed values up until the i'th obs
    mean_vector_1 = np.cumsum(s_obs_1[:,0])/(np.array(list(range(len(s_obs_1[:,0]))))+1)
    mean_vector_2 = np.cumsum(s_obs_2[:,0])/(np.array(list(range(len(s_obs_2[:,0]))))+1)
    mean_vector_3 = np.cumsum(s_obs_3[:,0])/(np.array(list(range(len(s_obs_3[:,0]))))+1)

    plt.figure(figsize=[10,5])
    plt.grid()
    plt.plot(mean_vector_1, color='blue')
    plt.plot(mean_vector_2, color='green')
    plt.plot(mean_vector_3, color='red')
    plt.title('Mean of Player 1 skill samples from Gibbs sampling procedure. 50000 samples')
    plt.xlabel('sample nr. ')
    plt.ylabel('mean skill estimate up to ith sample (arbitrary units)')
 


    #..........
    plt.figure(figsize=[10,5])
    plt.grid()
    #plotting mean of all observed values up until the i'th obs

    plt.plot(mean_vector_1[:20000], color='blue')
    plt.plot(mean_vector_2[:20000], color='green')
    plt.plot(mean_vector_3[:20000], color='red')

    plt.title("Mean up until ith gibbs sample/observation")
    plt.xlabel("'i'th gibbs sample/observation ")
    plt.ylabel("Mean value for player 1 estimate up unitl i'th")
  
      #..........

    # Q.4a
    burn_in_indx = 3000
    nr_samples_q4c = 15000

    s_obs, t_obs = q4_6lib.gibbs_sampler(L=nr_samples_q4c,
                                player_1_stats=player_1_stats,
                                player_2_stats=player_2_stats,
                                t_game=match_outcome)

    s_obs = s_obs[burn_in_indx:, :] # discard inital B samples due to burn in. 


    player_1_stats_estimate_1, player_2_stats_estimate = q4_6lib.player_stats_estimate_from_obs(s_obs=s_obs[burn_in_indx:burn_in_indx+100,:])
    
    player_1_stats_estimate_2, player_2_stats_estimate = q4_6lib.player_stats_estimate_from_obs(s_obs=s_obs[burn_in_indx:burn_in_indx+500, :])
    
    player_1_stats_estimate_3, player_2_stats_estimate = q4_6lib.player_stats_estimate_from_obs(s_obs=s_obs[burn_in_indx:burn_in_indx+2500, :])
    
    player_1_stats_estimate_4, player_2_stats_estimate = q4_6lib.player_stats_estimate_from_obs(s_obs=s_obs[burn_in_indx:burn_in_indx+10000, :])


    p1_mu_est_1 = player_1_stats_estimate_1[0]
    p1_var_est_1 = player_1_stats_estimate_1[1]
    p1_sigma_est_1 = np.sqrt(p1_var_est_1)

    p1_mu_est_2 = player_1_stats_estimate_2[0]
    p1_var_est_2 = player_1_stats_estimate_2[1]
    p1_sigma_est_2 = np.sqrt(p1_var_est_2)

    p1_mu_est_3 = player_1_stats_estimate_3[0]
    p1_var_est_3 = player_1_stats_estimate_3[1]
    p1_sigma_est_3 = np.sqrt(p1_var_est_3)

    p1_mu_est_4 = player_1_stats_estimate_4[0]
    p1_var_est_4 = player_1_stats_estimate_4[1]
    p1_sigma_est_4 = np.sqrt(p1_var_est_4)


    x1 = np.linspace(p1_mu_est_1 - 3 * p1_sigma_est_1, p1_mu_est_1 + 3 * p1_sigma_est_1, 100)
    x2 = np.linspace(p1_mu_est_2 - 3 * p1_sigma_est_2, p1_mu_est_2 + 3 * p1_sigma_est_2, 100)
    x3 = np.linspace(p1_mu_est_3 - 3 * p1_sigma_est_3, p1_mu_est_3 + 3 * p1_sigma_est_3, 100)
    x4 = np.linspace(p1_mu_est_4 - 3 * p1_sigma_est_4, p1_mu_est_4 + 3 * p1_sigma_est_4, 100)

    
    plt.figure(figsize=[20,5])
    plt.subplot(1,4, 1) 
    plt.hist(s_obs[burn_in_indx:burn_in_indx+100, 0], bins=50, density=True) #P(S1 \Y=1)
    plt.plot(x1, stats.norm.pdf(x1, loc = p1_mu_est_1, scale = p1_sigma_est_1))
    plt.xlabel('skill estimate')
    plt.ylabel('freq.')
    plt.title('100 samples after burn-in.')

    plt.subplot(1,4, 2) 
    plt.hist(s_obs[burn_in_indx:burn_in_indx+500, 0], bins=50, density=True) #P(S1 \Y=1)
    plt.plot(x2, stats.norm.pdf(x2, loc = p1_mu_est_2, scale = p1_sigma_est_2))
    plt.xlabel('skill estimate')
    plt.title('500 samples after burn-in.')

    plt.subplot(1,4, 3) 
    plt.hist(s_obs[burn_in_indx:burn_in_indx+3000, 0], bins=50, density=True) #P(S1 \Y=1)
    plt.plot(x3, stats.norm.pdf(x3, loc = p1_mu_est_3, scale = p1_sigma_est_3))
    plt.xlabel('skill estimate')
    plt.title('2500 samples after burn-in.')

    plt.subplot(1,4, 4) 
    plt.hist(s_obs[burn_in_indx:burn_in_indx+10000, 0], bins=50, density=True) #P(S1 \Y=1)
    plt.plot(x4, stats.norm.pdf(x4, loc = p1_mu_est_4, scale = p1_sigma_est_4))
    plt.xlabel('skill estimate')
    plt.ylabel('')
    plt.title('10000 samples after burn-in.')

    # .......
    plt.figure(figsize=[10,10])
    plt.hist(s_obs[burn_in_indx:burn_in_indx+25000, 0], bins=50, density=True) #P(S1 \Y=1)
    
    plt.plot(x3, stats.norm.pdf(x3, loc = p1_mu_est_3, scale = p1_sigma_est_3), label='$P(s_{1}|y=1) from gibbs$')

    x_p_s1 = np.linspace(player_1_mean - 3 * player_1_var, player_1_mean + 3 *player_1_var, 100)
    plt.plot(x_p_s1, stats.norm.pdf(x_p_s1, loc = player_1_mean, scale = np.sqrt(player_1_var)), label='"true"$P(s_{1})$')

    plt.title('comparing $P(s_{1})$ with $P(s_{1}|y=1)$')
    plt.xlabel('probability/likelihood value')
    plt.xlabel('player 1 skill estiate')
    plt.legend()




    ## Q5. and Q6 ------------------------

    # Make dictionary of team stats (mean and variance) & list of all games with result
    print('Q5 COMPUTATIONS')
    stats_dictionary, result_list = q4_6lib.make_stats_dictionary('SerieA.csv', stats_dictionary={}, printable=0)

    correct_predictions = 0
    nr_of_draws = 0

    # Randomize list
    x = np.random.choice(range(len(result_list)), size=len(result_list), replace=False)
    randomized_list = []
    for i in x:
        randomized_list.append(result_list[i])
    result_list = randomized_list[:250]

    result_list = result_list[:250]
    for i in range(len(result_list)):
        # result_list holds [team1, team2, result = score1 - score2]
        print(f"\nResult game {i}:   {result_list[i]}")
        team1 = result_list[i][0]
        team2 = result_list[i][1]
        
        print('team1: ', team1)
        print('team2: ', team2)
        result = result_list[i][2]
        
        # stats_dictionary with keyword 'teamname' and value [mean, variance]
        print(f"Stats before game: {stats_dictionary[team1]}, {stats_dictionary[team2]}")

        if result == 0:  # ignore tied games for now
            print("Game ignored due to tie")
            print(f"Stats after game:  {stats_dictionary[team1]}, {stats_dictionary[team2]}")
            nr_of_draws += 1
            print("Draws: ", nr_of_draws)
        else:
        
            # prediktera mat resultat och spara prediktion
            pred_result = q4_6lib.predict_winner(team1, team2, stats_dictionary)
            
            if np.sign(result) == np.sign(pred_result):
                    correct_predictions += 1
                    print("Correct pred: ", correct_predictions)
            
            print('true result: ', result)
            print('pred result: ', pred_result)
                    
            s_obs, t_obs =  q4_6lib.gibbs_sampler(L=5000,
                                        player_1_stats=stats_dictionary[team1], 
                                        player_2_stats=stats_dictionary[team2],
                                        t_game=result)
            
            player_1_stats_posterior, player_2_stats_posterior = q4_6lib.player_stats_estimate_from_obs(s_obs[3000:, :])
            
            
            # Update team stats so posterior makes new prior
            stats_dictionary[team1] = player_1_stats_posterior
            stats_dictionary[team2] = player_2_stats_posterior


            print(f"Stats AFTER game: {stats_dictionary[team1]}, {stats_dictionary[team2]}")

    print()
    print("correct_predictions ", correct_predictions)
    print("reslist: ", len(result_list))
    print("draws: ", nr_of_draws)
    print('performance: ', correct_predictions / (len(result_list)-nr_of_draws))
    q4_6lib.ranking(stats_dictionary)

   
    ## Q7.  ------------------------
    ## Q8.  ------------------------
    
    # Run message-passing with skill priors for team1 and team2, beta, and y as input parameters
    s1_posterior_mean, s1_posterior_variance, s2_posterior_mean, s2_posterior_variance = q7lib.message_passing(
        s1_prior_mean= 100, s1_prior_variance=1 0* *2, s2_prior_mean= 100, s2_prior_variance=1 0* *2,
        beta = 3, y_observed= 1)

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

    plt.plot(x1, stats.norm.pdf(x1, loc=p1_mu_est_4, scale=p1_sigma_est_4), label='$p(s_{1}|y=1) from gibbs$')
    plt.plot(x2, stats.norm.pdf(x2, loc=p2_mu_est_4, scale=p2_sigma_est_4), label='$p(s_{2}|y=1) from gibbs$')

    plt.plot(x1, stats.norm.pdf(x1, loc=s1_posterior_mean, scale=np.sqrt(s1_posterior_variance)),
             label='$p(s_{1}|y=1)$ from m-p')
    plt.plot(x2, stats.norm.pdf(x2, loc=s2_posterior_mean, scale=np.sqrt(s2_posterior_variance)),
             label='$p(s_{2}|y=1)$ from m-p')

    plt.title('comparing $p(s_{1})$ and $p(s_{2})$ from Gibbs sampling and message-passing')
    plt.xlabel('probability density')
    plt.xlabel('skills')
    plt.legend()
    plt.show()

   
    ## Q9.  ------------------------
   
   
    ## Q10. ------------------------
    plt.show()

