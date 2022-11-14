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


######################## SOLUCIÓ ###########################
def build_policy(Q, epsilon, num_Actions):
    """
    Crea una política epsilon-greedy basada en una función de valor de acción Q y epsilon
    
    Args:
        Q: Un diccionario cuya correspondencia es state -> action-values.
           Cada valor es un array de numpy de longitud num_Actions (see below)
        epsilon: La probabilidad de seleccionar una acción aleatoria (float entre 0 and 1).
        num_Actions: Número de acciones del entorno. (en el caso del WIndyGridWorld es 4)
    
    Returns:
        Una función que tome como argumento la observación y devuelva como resultado
        las probabilidades de cada acción como un array de numpy de longitud num_Actions.
    """
    def policy_fn(observation):

        A = np.ones(num_Actions, dtype=float) * epsilon / num_Actions
        best_action = np.argmax(Q[observation])
        A[best_action] += (1.0 - epsilon)

        return A
    
    return policy_fn


def mc_policy_first_visit(env, num_episodes, discount=1.0, epsilon=0.1):
    """
    Control mediante métodos de Montecarlo usando políticas Epsilon-Greedy
    Encuentra una política epsilon-greedy.
    
    Args:
        env: entorno OpenAI gym.
        num_episodes: Número de episodios de la muestra.
        discount: factor de descuento.
        epsilon: La probabilidad de seleccionar una acción aleatoria (float entre 0 and 1)
    
    Returns:
        Una tupla (Q, policy).
        Q: Un diccionario cuya correspondencia es state -> action-values.
        policy: Una función que toma como argumento la observación y devuelve como resultado
                las probabilidades de cada acción
    """
    
    # Inicialitzem els llistes per emmagatzemar la suma dels retorns i el nombre de retorna 
    returns_sum = defaultdict(float)
    returns_count = defaultdict(float)
    
    # Map que conté com a clau [state-action] i com a valor i com a valor el retorno promedio de tots els episodis de l'estat
    # Inicialitzem a 0
    Q = defaultdict(lambda: np.zeros(env.action_space.n))
    
    # Construïm la política inicial
    policy = build_policy(Q, epsilon, env.action_space.n)
    
    # Iterem tots el episodis
    for i_episode in range(1, num_episodes + 1):
        
        # Printem mitjançant la llibreria sys l'episodi actual. S'actualitza de forma dinàmica.
        if i_episode % 10 == 0:
            print("\rEpisode {}/{}.".format(i_episode, num_episodes), end="")
            sys.stdout.flush()

        # 
        # Un episodi conté la terna estat - acció - recompensa (l'estat és la observació de l'entorn)
        episode = []
        state = env.reset()
        done = False
        for t in range(1000): # afegim fins a 1000 elements en aquest episodi.
            probs = policy(state)
            # agafem l'acció
            action = np.random.choice(np.arange(len(probs)), p=probs)
            # executem l'acció de l'environtment del Blackjack i recuperem la nova observació, la recompensa i si ha finalitzat.
            next_state, reward, done, _ = env.step(action)
            # emmagatzemem la terna de l'episodi.
            episode.append((state, action, reward))
            if done:
                break
            # afegim la nova observació a avaluar i emmagatzemar
            state = next_state

        # Iterem totes els claus (estat-acció) d'aquest episodi i calcularems les recompenses d'aquests parells en tots el episodis.
        sa_in_episode = set([(tuple(x[0]), x[1]) for x in episode])
        for state, action in sa_in_episode:
            sa_pair = (state, action)
            # Per optimitzar busquem la primera aparició d'aquest parell i a partir d'aqui sumem totes les recompenses.
            first_occurence_idx = next(i for i,x in enumerate(episode) if x[0] == state and x[1] == action)
            # Sumem les recompenses desde la primera aparició.
            G = sum([x[2]*(discount**i) for i,x in enumerate(episode[first_occurence_idx:])])
            # Es calcula el retorn promig
            returns_sum[sa_pair] += G
            returns_count[sa_pair] += 1.0
            Q[state][action] = returns_sum[sa_pair] / returns_count[sa_pair]
        
        # La política es va millorant en les diferents iteracions.    
    return Q, policy


# Printem la política
def print_policy_MC(policy):
    for b_usable_ace in (0, 1):
        for i_player in range(12, 22):
            for i_dealer in range(1, 11):
                arr = np.array(policy((i_player,i_dealer,b_usable_ace)))
                action = int(np.where(arr == np.amax(arr))[0])
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

def compara_mc(policy):
    
    switch_action = {
        0: "Stick",
        1: "Hit",
    }
    errors = 0    
    i = 0
    for b_usable_ace in (0, 1):
        for i_player in range(12, 22):
            for i_dealer in range(1, 11):
                arr = np.array(policy((i_player,i_dealer,b_usable_ace)))
                action_mc_policy = int(np.where(arr == np.amax(arr))[0])
                action_optimal_policy = optimal_policy(i_player,i_dealer,b_usable_ace)                            
                print("------------------------------------------------------------------------------------------")
                print("La acció per player: {}, dealer: {}, usable_ace {} en la política de Montecarlo és: {}"
                      .format(i_player, i_dealer, b_usable_ace, switch_action[action_mc_policy]))
                
                print("La acció per player: {}, dealer: {}, usable_ace {} en la política óptima és: {}"
                      .format(i_player, i_dealer, b_usable_ace, switch_action[action_optimal_policy]))
                if (action_mc_policy == action_optimal_policy):
                    print("Les polítiques coincideixen!")
                else:
                    print("Les polítiques NO coincideixen")
                    errors = errors + 1 
                i = i + 1
    print("Hem realitzat un total de {} iteracions per comparar les polítiques.".format(i))
    return errors

Q, mc_policy = mc_policy_first_visit(env, num_episodes=100, discount=1, epsilon=0.1)
print_policy_MC(mc_policy)                 
errors = compara_mc(mc_policy)
print("Número total d'errors = {}".format(errors))                