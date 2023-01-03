from ex2_Buffer import experienceReplayBuffer
from ex2_DQN_NN import DQN_NN
from ex2_DQN_Agent import DQNAgent
import matplotlib.pyplot as plt
import gym


env = gym.make('LunarLander-v2')

# lr = 0.001              # (0.001) Velocitat d'aprenentatge
# BATCH_SIZE = 32         #Conjunt a agafar del buffer per a la xarxa neuronal
# MEMORY_SIZE = 8000      #Màxima capacitat del buffer
# GAMMA = 0.99            #Valor gamma de l'equació de Bellman
# EPSILON = 1             #Valor inicial d'epsilon
# EPSILON_DECAY = 0.995   #Decaïment d'epsilon
# EPSILON_MIN = 0.01      #Valor mínim d'epsilon
# BURN_IN = 1000          #Nombre d'episodis inicials utilitzats per emplenar el buffer abans d'entrenar
# MAX_EPISODES = 5000     #Nombre màxim d'episodis (l'agent ha d'aprendre abans d'arribar a aquest valor)
# MIN_EPISODES = 250      #Nombre mínim d'episodis
# DNN_UPD = 1             #Freqüència d'actualització de la xarxa neuronal
# DNN_SYNC = 200         #Freqüència de sincronització de pesos entre la xarxa neuronal i la xarxa objectiu

# REWARD_THRESHOLD = 200 #Llindar de recompensa on es considera que s'ha assolit el problema


lr = 0.0005             # (0.001) Velocitat d'aprenentatge
BATCH_SIZE = 64         #Conjunt a agafar del buffer per a la xarxa neuronal
MEMORY_SIZE = 100000    #Màxima capacitat del buffer
GAMMA = 0.99            #Valor gamma de l'equació de Bellman
EPSILON = 1             #Valor inicial d'epsilon
EPSILON_DECAY = 0.995   #Decaïment d'epsilon
EPSILON_MIN = 0.01      #Valor mínim d'epsilon
BURN_IN = 1000          #Nombre d'episodis inicials utilitzats per emplenar el buffer abans d'entrenar
MAX_EPISODES = 5000     #Nombre màxim d'episodis (l'agent ha d'aprendre abans d'arribar a aquest valor)
MIN_EPISODES = 250      #Nombre mínim d'episodis
DNN_UPD = 2             #Freqüència d'actualització de la xarxa neuronal
DNN_SYNC = 1         #Freqüència de sincronització de pesos entre la xarxa neuronal i la xarxa objectiu

REWARD_THRESHOLD = 200 #Llindar de recompensa on es considera que s'ha assolit el problema


buffer = experienceReplayBuffer(memory_size=MEMORY_SIZE, burn_in=BURN_IN)
dqn = DQN_NN(env, learning_rate=lr)
agentDQN = DQNAgent(env, dqn, buffer, REWARD_THRESHOLD, EPSILON, EPSILON_DECAY, BATCH_SIZE)
agentDQN.train(gamma=GAMMA, max_episodes=MAX_EPISODES, dnn_update_frequency=DNN_UPD, dnn_sync_frequency=DNN_SYNC)

def plot_rewards(agent):
        plt.figure(figsize=(12,8))
        plt.plot(agent.training_rewards, label='Rewards')
        plt.plot(agent.mean_training_rewards, label='Mean Rewards')
        plt.axhline(agent.reward_threshold, color='r', label="Reward threshold")
        plt.xlabel('Episodes')
        plt.ylabel('Rewards')
        plt.legend(loc="upper left")
        plt.show()

def plot_loss(agent):
        plt.figure(figsize=(12,8))
        plt.plot(agent.training_update_loss, label='Loss')
        plt.xlabel('Episodes')
        plt.ylabel('Loss')
        plt.legend(loc="upper left")
        plt.show()
        
def plot_epsilon(agent):
        plt.figure(figsize=(12,8))
        plt.plot(agent.sync_eps, label='Epsilon')
        plt.xlabel('Episodes')
        plt.ylabel('Epsilon')
        plt.legend(loc="upper left")
        plt.show()

plot_rewards(agentDQN)
plot_loss(agentDQN)
plot_epsilon(agentDQN)        

env.close()