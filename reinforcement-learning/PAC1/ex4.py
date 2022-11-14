"""
Created on Sat Nov  5 21:21:26 2022

@author: jpuig
"""
from collections import defaultdict
import sys
import numpy as np
import blackjack_env as bj1

# Inicialitzem l'entorn
bj_env = bj1.BlackjackEnv()

switch_action = {
    0: "Stick",
    1: "Hit",
}

def calculate_epsilon(n_episode):
    epsilon = 0.2
    if (n_episode == 0): 
        epsilon = 1
    else: 
        if 1/n_episode > 0.2:
            epsilon = 1/n_episode      
    return epsilon


def epsilon_greedy_action(Q, state, n_episode):
    p = np.random.random()
    action = np.random.choice(2)
    if p >= calculate_epsilon(n_episode):
        arr = np.array(Q[state])
        action = arr.argmax()          
    return action        

def initialize_q_sarsa(Q):
    for b_usable_ace in (0, 1):
        for i_player in range(1, 22):
            for i_dealer in range(1, 22):
                state = (i_player, i_dealer, b_usable_ace)
                Q[state][0] = 0
                Q[state][1] = 0
    return Q                                   
    

def sarsa_learning(env, episodes, learning_rate, discount):
    '''
    Algoritme per a generar la política optima

    :param episodes: Nombre d'episodis a executar
    :param learning_rate: Com de ràpid convergeix a un punt
    :param discount: Com els events futurs és devaluen
    :return: x,y punts del graf
    '''

    # Inicialitzem l'estructura de dades on tindrem les funcions de valor de les accions a 0
#    Q = defaultdict(lambda: np.zeros(bj_env.action_space.n))
    Q = initialize_q_sarsa(defaultdict(lambda: np.zeros(bj_env.action_space.n)))
    
    max_td_errors = [0.0] * episodes

    for episode in range(episodes):
        # Veiem el progrés de cada un dels episodis
        if episode % 1000 == 0:
            print("\rEpisode {}/{}.".format(episode, episodes), end="")
            sys.stdout.flush()        
        
        # Inicialitzem l'entorn
        state = bj_env.reset()

        # Seleccionem la primera acció aleatoria
        action =  epsilon_greedy_action(Q, state, episode)             
                                
        max_td_error = 0.0;   
        done = False
        while not done:                                           
            # Executem l'acció i recuperem el nou estat i la recompensa
            next_state, reward, done, _ = bj_env.step(action)

            # Obtenim la següent acció per a poder calcular la funció de valor de l'acció actual
            next_action =  epsilon_greedy_action(Q, next_state, episode)             
                                               
            # TD Update on actualitzem tant la funció de valor de les accions com la política.            
            if done:                
                td_target = reward + discount * 0
            else:
                td_target = reward + discount * Q[next_state][next_action]  
                     
            td_error = td_target - Q[state][action]                            
            Q[state][action] += learning_rate * td_error
            
            # Guardem el màxim td_error de cada episodi
            if td_error > max_td_errors[episode]:                
                max_td_errors[episode] = td_error                
                            
            state = next_state  
            action = next_action     
         
                      
    return Q, max_td_error


# Printem la política
def print_policy_sarsa(q_sarsa):
    for b_usable_ace in (0, 1):
        for i_player in range(12, 22):
            for i_dealer in range(1, 11):      
                arr = np.array(q_sarsa[i_player, i_dealer, b_usable_ace])
                action = arr.argmax()                       
                print("Usable Ace: {}, Dealer: {}, Player: {}, Action: {}".format(b_usable_ace == 0, i_dealer, i_player, switch_action[action]))


# Implementem la política a partir dels valors del dealer, player i usable_ace tal i com hem vist previament.
def optimal_policy(player, dealer, usable_ace):   
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


def compara_sarsa(q_sarsa):    
    errors = 0    
    i = 0
    for b_usable_ace in (0, 1):
        for i_player in range(12, 22):
            for i_dealer in range(1, 11):                         
                print("Acció per player: {}, dealer: {}, usable_ace {}".format(i_player, i_dealer, b_usable_ace))                
                arr = np.array(q_sarsa[i_player, i_dealer, b_usable_ace])
                action_td_policy = arr.argmax()  
                action_optimal_policy = optimal_policy(i_player,i_dealer,b_usable_ace)                       
                print("------------------------------------------------------------------------------------------")
                print("La acció per player: {}, dealer: {}, usable_ace {} en la política de SARSA és: {}".format(i_player, i_dealer, b_usable_ace, switch_action[action_td_policy]))                
                print("La acció per player: {}, dealer: {}, usable_ace {} en la política óptima és: {}" .format(i_player, i_dealer, b_usable_ace, switch_action[action_optimal_policy]))
                if (action_td_policy == action_optimal_policy):
                    print("Les polítiques coincideixen!")
                else:
                    print("Les polítiques NO coincideixen")
                    errors = errors + 1 
                i = i + 1
    return errors


q, max_td_error = sarsa_learning(bj_env, episodes=1000000, learning_rate=0.001, discount=1)
errors = compara_sarsa(q)
print("Número total d'errors = {}".format(errors))      