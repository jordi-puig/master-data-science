######################## SOLUCIÓ ###########################
from collections import defaultdict
import sys
import numpy as np
import blackjack_env as bj1
import math

switch_action = {
    0: "Stick",
    1: "Hit",
}

class TDAgent():
    # buckets show the discretization levels of the four features (position, velocity, angle, angular velocity)
    def __init__(self, num_episodes=5000, lr=0.8, initial_epsilon=0.9, discount=0.999):       
        self.num_episodes = num_episodes
        self.learning_rate = lr
        self.epsilon = initial_epsilon
        self.discount = discount
        self.env = bj1.BlackjackEnv()
        self.epsilon_min = 0.05
        self.lr_min = 0.1
        # create state-action Q table for the aciton values (Q)
        # this is where the value of each state-action pair is stored
        
        self.Q_table = defaultdict(lambda: np.zeros(self.env.action_space.n))

    
    # in general, starting from high exploration (high epsilon) and high learning rate 
    # we decrease the exploration probability and the learning rate as the training goes on
    
    def getEpsilon(self):
        if (self.epsilon <= self.epsilon_min):
            self.epsilon = self.epsilon_min
        else:
            self.epsilon*=0.9
            
    def getLR(self):
        if (self.learning_rate <= self.lr_min):
            self.learning_rate = self.lr_min
        else:
            self.learning_rate*=0.9
            
    # epsilon - greedy policy to choose action
    def choose_action(self, state):
        # the agent here: it chooses random action for epsilon probability otherwise it exploits
        if (np.random.random() < self.epsilon):
            return self.env.action_space.sample() 
        else:
            return np.argmax(self.Q_table[state])
    
    def SARSAupdate(self, state, action, reward, new_state, next_action):
        # updating the Q-value of the visited state-action pair
        self.Q_table[state][action] += self.learning_rate * (reward + self.discount * self.Q_table[new_state][next_action] - self.Q_table[state][action])
        
    def SARSAtrain(self):
        cum_reward = np.zeros((self.num_episodes))
        for ep in range(self.num_episodes):
            
            # Veiem el progr�s de cada un dels episodis
            if ep % 1000 == 0:
                print("\rEpisode {}/{}.".format(ep, self.num_episodes), end="")
                sys.stdout.flush()               
            
            current_state = self.env.reset()
            action = self.choose_action(current_state)
            done = False
            while not done:
                #choosing action according to our exploration-exploitation policy
                obs, reward, done, _ = self.env.step(action)
                cum_reward[ep]+=reward
                new_state = obs
                next_action = self.choose_action(new_state)
                self.SARSAupdate(current_state, action, reward, new_state, next_action)
                current_state = new_state
                action = next_action
            self.getEpsilon()
            self.getLR()
        return cum_reward, self.Q_table
        print('SARSA based training is finished!')

# Printem la política
def print_policy_sarsa(q_sarsa):
    for b_usable_ace in (0, 1):
        for i_player in range(12, 22):
            for i_dealer in range(1, 11):      
                arr = np.array(q_sarsa(i_player, i_dealer, b_usable_ace))
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
    switch_action = {
        0: "Stick",
        1: "Hit",
    }
    errors = 0    
    i = 0
    for b_usable_ace in (0, 1):
        for i_player in range(12, 22):
            for i_dealer in range(1, 11):                         
                action_td_policy = np.argmax(q_sarsa[i_player, i_dealer, b_usable_ace])
                action_optimal_policy = optimal_policy(i_player,i_dealer,b_usable_ace)                       
                print("------------------------------------------------------------------------------------------")
                print("La acció per player: {}, dealer: {}, usable_ace {} en la política de Sarsa és: {}".format(i_player, i_dealer, b_usable_ace, switch_action[action_td_policy]))                
                print("La acció per player: {}, dealer: {}, usable_ace {} en la política óptima és: {}" .format(i_player, i_dealer, b_usable_ace, switch_action[action_optimal_policy]))
                if (action_td_policy == action_optimal_policy):
                    print("Les polítiques coincideixen!")
                else:
                    print("Les polítiques NO coincideixen")
                    errors = errors + 1 
                i = i + 1
    print("Hem realitzat un total de {} iteracions per comparar les polítiques.".format(i))
    return errors


agent = TDAgent(1000000, 0.001,1,1)
cum_reward = agent.SARSAtrain()

print(agent.Q_table)
errors = compara_sarsa(agent.Q_table)
print("N�mero total d'errors = {}".format(errors))   
