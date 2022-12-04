from ex2_Buffer import experienceReplayBuffer
from ex2_DQN_CNN import DQN_CNN
from ex2_DQN_Agent import DQNAgent

import gym


env = gym.make('SpaceInvaders-v4')

lr = 0.001          #Velocitat d'aprenentatge
BATCH_SIZE = 32     #Conjunt a agafar del buffer per a la xarxa neuronal 
MEMORY_SIZE = 8000  #Màxima capacitat del buffer
GAMMA = 0.99        #Valor gamma de l'equació de Bellman
EPSILON = 1         #Valor inicial d'epsilon
EPSILON_DECAY = .99 #Decaïment d'epsilon
EPSILON_MIN = 0.01  #Valor mínim d'epsilon
BURN_IN = 100       #Nombre d'episodis inicials utilitzats per emplenar el buffer abans d'entrenar
MAX_EPISODES = 5000 #Nombre màxim d'episodis (l'agent ha d'aprendre abans d'arribar a aquest valor)
MIN_EPISODES = 250  #Nombre mínim d'episodis
DNN_UPD = 100       #Freqüència d'actualització de la xarxa neuronal 
DNN_SYNC = 5000     #Freqüència de sincronització de pesos entre la xarxa neuronal i la xarxa objectiu

REWARD_THRESHOLD = 350 #Llindar de recompensa on es considera que s'ha assolit el problema

buffer = experienceReplayBuffer(memory_size=MEMORY_SIZE, burn_in=BURN_IN)
dqn = DQN_CNN(env, learning_rate=lr)
agent = DQNAgent(env, dqn, buffer, REWARD_THRESHOLD, EPSILON, EPSILON_DECAY, BATCH_SIZE)
agent.train(gamma=GAMMA, max_episodes=MAX_EPISODES, batch_size=BATCH_SIZE, dnn_update_frequency=DNN_UPD, dnn_sync_frequency=DNN_SYNC)