import numpy as np
import blackjack_env as bj1

# Inicialitzem l'entorn
env = bj1.BlackjackEnv()

######################## SOLUCIÓ ###########################
def optimal_policy(usable_ace, dealer, player):
    optimal_policy = {}
    optimal_policy[(True, 1)] = 18
    optimal_policy[(True, 2)] = 17
    optimal_policy[(True, 3)] = 17
    optimal_policy[(True, 4)] = 17
    optimal_policy[(True, 5)] = 17
    optimal_policy[(True, 6)] = 17
    optimal_policy[(True, 7)] = 17
    optimal_policy[(True, 8)] = 17
    optimal_policy[(True, 9)] = 18
    optimal_policy[(True, 10)] = 18
    optimal_policy[(False, 1)] = 16
    optimal_policy[(False, 2)] = 12
    optimal_policy[(False, 3)] = 12
    optimal_policy[(False, 4)] = 11
    optimal_policy[(False, 5)] = 11
    optimal_policy[(False, 6)] = 11
    optimal_policy[(False, 7)] = 16
    optimal_policy[(False, 8)] = 16
    optimal_policy[(False, 9)] = 16
    optimal_policy[(False, 10)] = 16

    max_player = optimal_policy[usable_ace, dealer]
    if player <= max_player:
        action = 1
    else:
        action = 0

    return action


def percent_of(num_a, num_b):
    return (num_a / num_b) * 100


def play_game(optimal_policy):

    obs = env.reset()
    total_reward, done = 0, False

    print("Obs inicial: Player = {}, Dealer = {}, Usable Ace = {} ".format(obs[0], obs[1], obs[2]))

    switch_action = {
        0: "Stick",
        1: "Hit",
    }

    while not done:
        # obtenim l'acció a executar a partir de la política
        action = optimal_policy(obs[2], obs[1], obs[0])
        # executem l'acció y esperem la resposta de l'entorn
        new_obs, reward, done, info = env.step(action)
        # imprimim el valor de la observació de l'estat actual
        print("Action: {} -> Obs: Player = {}, Dealer = {}, Usable Ace = {} and reward: {}".format(switch_action[action], new_obs[0], new_obs[1], new_obs[2], reward))

        # actualizar variables
        obs = new_obs
        total_reward += reward

        print("Episodi finalitzat i la recompensa és de {} ".format(total_reward))
    
    return total_reward

def play_games(env, optimal_policy, games):
    total_reward = 0
    current_game = 0
    total_win = 0
    total_lose = 0
    total_draw = 0

    while current_game < games:
        reward = play_game(optimal_policy)
        if (reward == 0):
            total_draw += 1
        elif (reward == -1):
            total_lose += 1
        elif (reward == 1):
            total_win += 1
        total_reward += reward
        current_game = current_game + 1

    print("\n\n{} jocs executats amb una recompensa total de {} ".format(current_game, total_reward))
    print("El nombre de victories totals és de {} amb un percentatge del {}%".format(total_win, percent_of(total_win, current_game)))
    print("El nombre de derrotes és de {} amb un percentatge del {}%".format(total_lose, percent_of(total_lose, current_game)))
    print("El nombre d'empats és de {} amb un percentatge del {}%".format(total_draw, percent_of(total_draw, current_game)))

games = 100000
play_games(env, optimal_policy, games)

env.close()