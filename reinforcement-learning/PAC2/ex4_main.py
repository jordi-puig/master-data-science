from ex4_PGReinforce import PGReinforce
from ex4_ReinforceAgent import ReinforceAgent
import gym

env = gym.make('SpaceInvaders-v4')

lr = 0.005          #Velocitat d'aprenentatge
BATCH_SIZE = 8      #Conjunt a agafar del buffer per a la xarxa neuronal 
GAMMA = 0.99        #Valor gamma de l'equació de Bellman
MAX_EPISODES = 3000 #Nombre màxim d'episodis (l'agent ha d'aprendre abans d'arribar a aquest valor)

pgNetwork = PGReinforce(env, learning_rate=lr)
agentPG = ReinforceAgent(env, pgNetwork)
agentPG.train(gamma=GAMMA, max_episodes=MAX_EPISODES, batch_size=BATCH_SIZE)