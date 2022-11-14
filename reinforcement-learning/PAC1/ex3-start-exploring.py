# -*- coding: utf-8 -*-
"""
Created on Thu Nov  3 21:25:54 2022

@author: jpuig
"""
from collections import defaultdict
import sys
import numpy as np
import blackjack_env as bj1

# Inicialitzem l'entorn
env = bj1.BlackjackEnv()

switch_action = {
    0: "Stick",
    1: "Hit",
}

# Nota: Els estats són el mateix que les observacions.
# Generem un episodi de Blackjack sense seguir cap política i emmagatzemem totes les pases a la variable episode
def generate_episode(bj_env, policy):
    episode = []
    obs = bj_env.reset()   
    done = False
    action = bj_env.action_space.sample()   
    # Primera acció la fem aleatoria per cada episodi i la resta ja seguint la política.          
    while not done:                                  
        new_obs, reward, done, _ = bj_env.step(action)
        episode.append((obs, action, reward))
        obs = new_obs
        action = int(policy[obs[0],obs[1],obs[2]])
    return episode

######################## SOLUCIÓ ###########################
def mc_policy_first_visit(env, num_episodes, discount):
    # Inicialitzem els llistes per emmagatzemar la suma dels retorns i el nombre de retorna 
    returns_sum = defaultdict(float)
    returns_count = defaultdict(float)
    
    # Map que conté com a clau [state-action] i com a valor i com a valor el retorno promedio de tots els episodis de l'estat
    # Inicialitzem a 0
    Q = defaultdict(lambda: np.zeros(env.action_space.n))
    
    # Construïm la política inicial
    policy = defaultdict(float)


    # iterem sobre els episodis
    for i_episode in range(1, num_episodes + 1):
        # Veiem el progrés de cada un dels episodis
        if i_episode % 1000 == 0:
            print("\rEpisode {}/{}.".format(i_episode, num_episodes), end="")
            sys.stdout.flush()
        # generate an episode
        episode = generate_episode(env, policy)
            
        # obtenim arrays de observacions, accions i recompenses del episodi actual
        observations, actions, rewards = zip(*episode)
        # Implementem els descomptes. Realment amb un valor d'1 de l'exercici no caldria aquest procés ja que sempre serà 1.
        # Però amb altres valors com per exemple 0.1 va diluint el pes de descomptes conforme els pasos dels episodis.
        discounts = np.array([discount**i for i in range(len(rewards)+1)])        
        
        # Iterem les observacions i emmagatzemem tant el nombre de les ocurrencies del parell observació + acció com la mitjana de les recompenses.            
        seen = set()
        for i, obs in enumerate(observations):                 
            obs_act_pair = (obs, actions[i])       
            # Aquest control el defineix l'algortme però en el nostre cas no caldria ja que 
            # no es donaràn dos obs-action en el mateix episodi.
            if obs_act_pair not in seen:    
                seen.add(obs_act_pair)
                # Sumem totes les ocurrencies de les observacions o estats
                returns_count[obs_act_pair] += 1.0
                # Sumem totes les recompenses de l'episodi i l'emmagatzemem en el map returns_sum
                G = sum(rewards[i:]*discounts[:-(1+i)])
                returns_sum[obs_act_pair] += G            
                # Fem la mitjana de les recompenses i l'emmagatzemem amb el map [observació][acció] i així tenim els valors de les mitjanes 
                # de la mateixa observació per a cada un dels estats. L'acció de la observació serà el major d'aquests dos valors.            
                Q[obs][actions[i]] = returns_sum[obs_act_pair] / returns_count[obs_act_pair]
                # Escollim el valor màxim entre les diferents accions de Q(St,At) i l'index del valor màxim ens dona l'acció (0 o 1).           
                best_action = np.argmax(Q[obs])
                # Anem construïnt la política
                policy[obs] = best_action                                     
    return Q, policy


# Printem la política
def print_policy_MC(policy):
    for b_usable_ace in (0, 1):
        for i_player in range(12, 22):
            for i_dealer in range(1, 11):                             
                action = int(mc_policy[i_player,i_dealer,b_usable_ace])
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


def compara_mc(mc_policy):    
    switch_action = {
        0: "Stick",
        1: "Hit",
    }
    errors = 0    
    i = 0
    for b_usable_ace in (0, 1):
        for i_player in range(12, 22):
            for i_dealer in range(1, 11):            
                action_mc_policy = int(mc_policy[i_player,i_dealer,b_usable_ace])
                action_optimal_policy = optimal_policy(i_player,i_dealer,b_usable_ace)                            
                print("------------------------------------------------------------------------------------------")
                print("La acció per player: {}, dealer: {}, usable_ace {} en la política de Montecarlo és: {}".format(i_player, i_dealer, b_usable_ace, switch_action[action_mc_policy]))                
                print("La acció per player: {}, dealer: {}, usable_ace {} en la política óptima és: {}" .format(i_player, i_dealer, b_usable_ace, switch_action[action_optimal_policy]))
                if (action_mc_policy == action_optimal_policy):
                    print("Les polítiques coincideixen!")
                else:
                    print("Les polítiques NO coincideixen")
                    errors = errors + 1 
                i = i + 1
    print("Hem realitzat un total de {} iteracions per comparar les polítiques.".format(i))
    return errors


Q, mc_policy = mc_policy_first_visit(env, 5000000, discount=1)
print_policy_MC(mc_policy)                
errors = compara_mc(mc_policy)
print("Número total d'errors = {}".format(errors))      