import gym
from ex2_DQN_Agent import Agent

BUFFER_SIZE = 100000  # Màxima capacitat del buffer
BATCH_SIZE = 64       # Conjunt a agafar del buffer per a la xarxa neuronal
GAMMA = 0.99          # Valor gamma de l'equació de Bellman
TAU = 1e-3            # Valor de tau per a soft update del target network
LEARNING_RATE = 5e-4  # Velocitat d'aprenentatge
DNN_UPD = 4           # Freqüència d'actualització de la xarxa neuronal

# inicialització de l'entorn de gym
env = gym.make('LunarLander-v2')
# inicialització de l'agent amb els paràmetres de l'exercici
agent = Agent(env, seed=0, learning_rate=LEARNING_RATE, gamma=GAMMA, tau=TAU, buffer_size=BUFFER_SIZE, batch_size=BATCH_SIZE, dnn_upd=DNN_UPD)
# run the training session
scores = agent.train()