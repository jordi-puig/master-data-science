from ex2_Buffer import experienceReplayBuffer
from ex3_DDQN_Agent import DuelingDQNAgent
import gym



env = gym.make('SpaceInvaders-v4')

import torch
import numpy as np
import torch.nn as nn
import torch.autograd as autograd
import torch.optim as optim

class DuelingDQN(torch.nn.Module):
    ###################################
    ###TODO: inicialització i model ###
    def __init__(self, env, learning_rate=1e-3):
        """
        Params
        ======
        input_shape: mida de l'espai d'estats
        n_outputs: mida de l'espai d'accions
        actions: array d'accions possibles
        device: cpu o cuda
        red_cnn: definició de la xarxa convolucional
        value: definició de la xarxa neuronal value
        advantage: definició de la xarxa neuronal advantage
        """
        ###################################
        ###TODO: Inicialitzar variables####
        super(DuelingDQN, self).__init__()
        self.input_shape = (4,84,84)
        self.n_outputs = env.action_space.n
        self.actions = np.arange(env.action_space.n)
        
        if torch.cuda.is_available():
            self.device = 'cuda'
        else:
            self.device = 'cpu'


            
        ########################################
        ##TODO: Construcció de la xarxa neuronal
        self.red_cnn = nn.Sequential(
            nn.Conv2d(in_channels=4, out_channels=32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(3136, 512),
            nn.ReLU(),
            nn.Linear(512, self.n_outputs),
        )

        if torch.cuda.is_available():
            self.red_cnn.cuda()


        ### Inicialitzem l'optimitzador
        self.optimizer = optim.Adam(self.parameters(), lr=learning_rate)

    #######################################
    #####TODO: funció forward#############
    def forward(self, x):
        """  
        x: estat de l'entorn
        """   
        return self.online(x)
        

    ### Métode e-greedy
    def get_action(self, state, epsilon=0.05):
        if np.random.random() < epsilon:
            action = np.random.choice(self.actions)  # acció aleatòria
        else:
            qvals = self.get_qvals(state)  # acció a partir del càlcul del valor de Q per a aquesta acció
            action= torch.max(qvals, dim=-1)[1].item()
        return action


    def get_qvals(self, state):
        if type(state) is tuple:
            state = np.array(state)
        state_t = torch.FloatTensor(state).to(device=self.device)
        return self.forward(state_t)


    def feature_size(self):
        return self.red_cnn(autograd.Variable( torch.zeros(1, * self.input_shape)).to(device=self.device)).view(1, -1).size(1)



lr = 0.001          #Velocitat d'aprenentatge
BATCH_SIZE = 32     #Conjunt a agafar del buffer per a la xarxa neuronal 
MEMORY_SIZE = 100000 # (8000) Màxima capacitat del buffer
GAMMA = 0.99        #Valor gamma de l'equació de Bellman
EPSILON = 1         #Valor inicial d'epsilon
EPSILON_DECAY = .995 #Decaïment d'epsilon
EPSILON_MIN = 0.01  #Valor mínim d'epsilon
BURN_IN = 100       #Nombre d'episodis inicials utilitzats per emplenar el buffer abans d'entrenar
MAX_EPISODES = 1000 #Nombre màxim d'episodis (l'agent ha d'aprendre abans d'arribar a aquest valor)
MIN_EPISODES = 250  #Nombre mínim d'episodis
DNN_UPD = 2         # (100) Freqüència d'actualització de la xarxa neuronal 
DNN_SYNC = 10000    # (5000) Freqüència de sincronització de pesos entre la xarxa neuronal i la xarxa objectiu

REWARD_THRESHOLD = 350 #Llindar de recompensa on es considera que s'ha assolit el problema

buffer = experienceReplayBuffer(memory_size=MEMORY_SIZE, burn_in=BURN_IN)
ddqn = DuelingDQN(env, learning_rate=lr)
agentDDQN = DuelingDQNAgent(env, ddqn, buffer, REWARD_THRESHOLD, EPSILON, EPSILON_DECAY, BATCH_SIZE)
agentDDQN.train(gamma=GAMMA, max_episodes=MAX_EPISODES, batch_size=BATCH_SIZE, dnn_update_frequency=DNN_UPD, dnn_sync_frequency=DNN_SYNC)

