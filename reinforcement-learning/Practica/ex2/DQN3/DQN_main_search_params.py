import gym
import matplotlib.pyplot as plt

from DQN_Agent import Agent

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
DNN_UPD = [1, 3, 5]              # Freqüència d'actualització de la xarxa neuronal
EPS_DECAY = [0.99, 0.995]                # Decaiment de l'exploració

agents = []
n_test = 1
# iteració per a buscar els millors paràmetres
for lr in LEARNING_RATE:
    for upd in DNN_UPD:
        for eps in EPS_DECAY:
            print("Test Number: ",n_test)
            print("Learning rate: ", lr, " DNN update: ", upd, " Epsilon decay: ", eps)
            agent = Agent(env, seed=0, learning_rate=lr, gamma=GAMMA, tau=TAU, buffer_size=BUFFER_SIZE, batch_size=BATCH_SIZE, dnn_upd=upd)
            scores = agent.train(N_EPISODES, MAX_T, EPS_START, EPS_MIN, eps, NBLOCK, MIN_EPISODES, REWARD_THRESHOLD)
            agents.append(agent)
            print("Mean reward: ", agent.mean_training_rewards[-1])
            print("Mean loss: ", agent.mean_update_loss[-1])
            print("Epsilon: ", agent.sync_eps[-1])
            print("Steps: ", agent.total_episodes)
            print("Total time: ", agent.total_time)
        n_test += 1               
                  
# tanquem l'entorn de gym
env.close()

# plot de la recompensa de tots els agents en una mateixa gràfica
for idx,agent in enumerate(agents):    
        if max(agent.mean_training_rewards) > 200:
            plt.plot(agent.mean_training_rewards)       
            plt.legend(str(idx))
        
plt.axhline(agent.reward_threshold, color='r', label="Reward threshold")
plt.xlabel('Episodes')
plt.ylabel('Rewards')        
plt.rcParams['figure.figsize'] = [18, 14]
plt.show()
