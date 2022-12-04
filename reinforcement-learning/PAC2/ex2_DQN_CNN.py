import torch
import numpy as np
import torch.nn as nn
import torch.autograd as autograd
import torch.optim as optim

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
        ###TODO: Inicialització i model ###        
        self.learning_rate = learning_rate
        self.input_shape = (4,84,84)
        self.n_outputs = env.action_space.n
        self.actions = np.arange(env.action_space.n)
        
        if torch.cuda.is_available():
            self.device = 'cuda'
        else:
            self.device = 'cpu'

        #######################################
        ##TODO: Construcció de la xarxa neuronal convolucional
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

        #######################################
        ##TODO: Construcció de la xarxa neuronal lineal completament connectada
        self.fc_layer_inputs = self.feature_size()
        self.red_lineal = nn.Sequential(
            nn.Linear(self.fc_layer_inputs, 512),
            nn.ReLU(),
            nn.Linear(512, self.n_outputs)
        )


        if torch.cuda.is_available():
            self.red_lineal.cuda()

        #######################################
        ##TODO: Inicialitzar l'optimitzador
        ## Recupera els paràmetres de la xarxa neuronal i inicialitza l'optimitzador amb el learning rate indicat en el constructor      
        self.optimizer = optim.Adam(self.parameters(), self.learning_rate)

    ### Mètode e-greedy
    def get_action(self, state, epsilon=0.05):
        if np.random.random() < epsilon:            
            action = np.random.choice(self.actions)
        else:
            qvals = self.get_qvals(state)
            action= torch.max(qvals, dim=-1)[1].item()
        return action


    def get_qvals(self, state):
        if type(state) is tuple:
            state = np.array(state)
        state_t = torch.FloatTensor(state).to(device=self.device)
        cnn_out = self.red_cnn(state_t).reshape(-1,  self.fc_layer_inputs)
        return self.red_lineal(cnn_out)


    def feature_size(self):
        return self.red_cnn(autograd.Variable(torch.zeros(1, *self.input_shape)).to(device=self.device)).view(1, -1).size(1)





