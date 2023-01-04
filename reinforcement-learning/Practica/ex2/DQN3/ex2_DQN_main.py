import gym
from ex2_DQN_Agent import Agent
import matplotlib.pyplot as plt

BUFFER_SIZE = 100000    # Màxima capacitat del buffer
BATCH_SIZE = 64         # Conjunt a agafar del buffer per a la xarxa neuronal
GAMMA = 0.99            # Valor gamma de l'equació de Bellman
TAU = 1e-3              # Valor de tau per a soft update del target network
LEARNING_RATE = 5e-4    # Velocitat d'aprenentatge
DNN_UPD = 10            # Freqüència d'actualització de la xarxa neuronal

N_EPISODES=2000
MAX_T=1000 
EPS_START=1.0
EPS_MIN=0.01
EPS_DECAY=0.995
NBLOCK =100
MIN_EPISODES=250
REWARD_THRESHOLD = 200  # Valor de recompensa per a considerar l'entrenament com a completat

# inicialització de l'entorn de gym
env = gym.make('LunarLander-v2')
# inicialització de l'agent amb els paràmetres de l'exercici
agent = Agent(env, seed=0, learning_rate=LEARNING_RATE, gamma=GAMMA, tau=TAU, buffer_size=BUFFER_SIZE, batch_size=BATCH_SIZE, dnn_upd=DNN_UPD)
# entrenament de l'agent
scores = agent.train(N_EPISODES, MAX_T, EPS_START, EPS_MIN, EPS_DECAY, NBLOCK, MIN_EPISODES, REWARD_THRESHOLD)


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
        plt.plot(agent.mean_update_loss, label='Loss')
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

plot_rewards(agent)
plot_loss(agent)
plot_epsilon(agent)        
env.close()

# TODO: explicar Tau, MSE, soft