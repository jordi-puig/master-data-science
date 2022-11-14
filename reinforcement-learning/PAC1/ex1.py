import gym
import numpy as np
import blackjack_env_v2 as bj2

env=bj2.BlackjackEnv()

######################## SOLUCIÓ ###########################
sample_policy = {
    20: 0,
    21: 0
}

def play_game(policy):

    # Inicialitzem l'entorn    
    obs = env.reset()
    total_reward, done = 0, False


    print("Obs inicial: Player = {}, Dealer = {}, Usable Ace = {} ".format(obs[0],obs[1],obs[2]))

    switch_action = {
        0: "Stick",
        1: "Hit",
    }
    
    while not done:
    
        # Si la mà és 20 o 21 ens plantem sino agafem una acció aleatoria. Apliquem la policy que passem per paràmetre.
        action = policy.get(obs[0])       
        if action is None:            
            action = env.action_space.sample()

        # Executem l'acció y esperem la resposta de l'entorn
        new_obs, reward, done, info = env.step(action)
    
        # Imprimim el valor de la observació de l'estat actual
        print("Action: {} -> Obs: Player = {}, Dealer = {}, Usable Ace = {} and reward: {}".format(switch_action[action], new_obs[0],new_obs[1],new_obs[2], reward))
    
        # Actualizar variables
        obs = new_obs
        total_reward += reward
    
    print("Episodi finalitzat i la recompensa és de {} ".format(total_reward))
    env.close()
    
    return reward

def percent_of(num_a, num_b):
    return (num_a / num_b) * 100

def play_games_natural(env, sample_policy, games):
    total_reward = 0
    current_game = 0
    total_win = 0
    total_lose = 0
    total_natural = 0
    total_draw = 0
        
    while current_game < games:
        reward = play_game(sample_policy)
        if (reward == 0):
            total_draw += 1
        elif (reward == -1):
            total_lose += 1
        elif (reward == 1):
            total_win += 1
        elif (reward == 1.5):
            total_natural += 1                         
        total_reward += reward
        current_game = current_game + 1
    
    print("\n\n{} jocs executats amb una recompensa total de {} ".format(current_game, total_reward))
    print("El nombre de victories totals és de {} amb un percentatge del {}%".format(total_natural+total_win, percent_of(total_natural+total_win, current_game)))    
    print("El nombre de victories naturals és de {} amb un percentatge del {}%".format(total_natural, percent_of(total_natural, current_game)))    
    print("El nombre de victories no naturals és de {} amb un percentatge del {}%".format(total_win, percent_of(total_win, current_game)))    
    print("El nombre de derrotes és de {} amb un percentatge del {}%".format(total_lose, percent_of(total_lose, current_game)))
    print("El nombre d'empats és de {} amb un percentatge del {}%".format(total_draw, percent_of(total_draw, current_game)))
    
    
games = 100000
play_games_natural(env, sample_policy, games)