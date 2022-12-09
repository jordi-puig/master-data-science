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
            nn.ReLU()
        )

        if torch.cuda.is_available():
            self.red_cnn.cuda()


        self.fc_layer_inputs = self.feature_size()

        self.advantage = nn.Sequential(
            nn.Linear(self.fc_layer_inputs, 512),
            nn.ReLU(),
            nn.Linear(512, self.n_outputs)
        )

        self.value = nn.Sequential(
            nn.Linear(self.fc_layer_inputs, 512),
            nn.ReLU(),
            nn.Linear(512, 1)
        )

        ### S'ofereix l'opció de treballar amb cuda
        if self.device == 'cuda':
            self.value.cuda()
            self.advantage.cuda()

        ### Inicialitzem l'optimitzador
        self.optimizer = optim.Adam(self.parameters(), lr=learning_rate)

    #######################################
    #####TODO: funció forward#############
    def forward(self, x):
        """  
        x: estat de l'entorn
        """             
        x = self.red_cnn(x)
        x = x.view(x.size(0), -1) # x: vector de sortida de la xarxa convolucional
        advantage = self.advantage(x) # advantage: sortida de la xarxa neuronal advantage
        value = self.value(x) # value: sortida de la xarxa neuronal value
        return value + advantage - advantage.mean() # sortida de la xarxa neuronal
        

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
