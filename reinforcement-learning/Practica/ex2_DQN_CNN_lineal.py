import gym
import torch
import numpy as np
import torch.nn as nn
import torch.autograd as autograd
import torch.optim as optim



# Crea el entorno de gym
env = gym.make('LunarLander-v2')


### Definició de la xarxa neuronal
class DQN_CNN(nn.Module):
    
    def __init__(self, env, learning_rate=1e-3):
        super(DQN_CNN, self).__init__()
        """
        Params
        ======
        n_inputs: mida de l'espai d'estats
        n_outputs: mida de l'espai d'accions
        actions: array d'acciones possibles
        device: cpu o cuda
        red_cnn: definició de la xarxa convolucional
        red_lineal: definició de la xarxa lineal
        """
        #######################################
        ### Inicialització i model ###        

        # learning rate de l'optimitzador        
        self.learning_rate = learning_rate
        # Inicialització de l'espai d'estats a 8 i d'accions a 4        
        self.n_outputs = env.action_space.n
        self.actions = np.arange(env.action_space.n)
        
        
        if torch.cuda.is_available():
            self.device = 'cuda'
        else:
            self.device = 'cpu'
        
        #######################################
        # ## Construcció de la xarxa neuronal      
        self.red_lineal = nn.Sequential(
            nn.Linear(8, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, self.n_outputs),
            nn.ReLU()
        )

        if torch.cuda.is_available():
            self.red_lineal.cuda()

        #######################################
        ## Inicialitzar l'optimitzador
        ## Recupera els paràmetres de la xarxa neuronal i inicialitza l'optimitzador amb el learning rate indicat en el constructor      
        self.optimizer = optim.Adam(self.parameters(), self.learning_rate)

    ### Mètode e-greedy per a l'acció a realitzar en un estat determinat
    def get_action(self, state, epsilon=0.05):
        if np.random.random() < epsilon:           
            action = np.random.choice(self.actions)
        else:
            qvals = self.get_qvals(state)
            action= torch.max(qvals, dim=-1)[1].item()
        return action

    ### Mètode per a calcular els qvals d'un estat determinat     
    def get_qvals(self, state):        
        if type(state) is tuple:
            state = np.array(state)
        state_t = torch.FloatTensor(state).to(device=self.device)
        return self.red_lineal(state_t)

    
    def feature_size(self):
        return self.red_cnn(autograd.Variable(torch.zeros(1, *self.input_shape)).to(device=self.device)).view(1, -1).size(1)