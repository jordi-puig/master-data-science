import gym
from ex2_DQN_Agent import Agent
import matplotlib.pyplot as plt

BUFFER_SIZE = 100000    # Màxima capacitat del buffer
BATCH_SIZE = 64         # Conjunt a agafar del buffer per a la xarxa neuronal
GAMMA = 0.99            # Valor gamma de l'equació de Bellman
TAU = 1e-3              # Valor de tau per a soft update del target network
LEARNING_RATE = 5e-4    # Velocitat d'aprenentatge
DNN_UPD = 3            # Freqüència d'actualització de la xarxa neuronal

N_EPISODES=2
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

LEARNING_RATE = [1e-3, 5e-4, 1e-4]              # Velocitat d'aprenentatge
#DNN_UPD = [1, 3, 5]                             # Freqüència d'actualització de la xarxa neuronal
DNN_UPD = [3]                             # Freqüència d'actualització de la xarxa neuronal
#EPS_DECAY = [0.99, 0.995, 0.999]                # Decaiment de l'exploració
EPS_DECAY = [0.99]                # Decaiment de l'exploració
agents = []
# iteració per a buscar els millors paràmetres
for lr in LEARNING_RATE:
        for upd in DNN_UPD:
                for eps in EPS_DECAY:
                        print("Learning rate: ", lr, " DNN update: ", upd, " Epsilon decay: ", eps)
                        agent = Agent(env, seed=0, learning_rate=lr, gamma=GAMMA, tau=TAU, buffer_size=BUFFER_SIZE, batch_size=BATCH_SIZE, dnn_upd=upd)
                        scores = agent.train(N_EPISODES, MAX_T, EPS_START, EPS_MIN, eps, NBLOCK, MIN_EPISODES, REWARD_THRESHOLD)
                        agents.append(agent)
                        print("Mean reward: ", agent.mean_training_rewards[-1])
                        print("Mean loss: ", agent.mean_update_loss[-1])
                        print("Epsilon: ", agent.sync_eps[-1])
                        print("")

# tanquem l'entorn de gym
env.close()

# plot de la recompensa de tots els agents en una mateixa gràfica
for agent in agents:
        plt.plot(agent.mean_training_rewards)        
        plt.legend(['Reward'])
        
plt.show()

