import numpy as np
import torch
import torch.nn as nn
import torch.autograd as autograd
import torch.optim as optim

class PGReinforce(torch.nn.Module):

    def __init__(self, env, learning_rate=1e-3, device='cpu'):
        print("init")
        """
        Params
        ======
        n_inputs: mida de l'espai d'estats
        n_outputs: mida de l'espai d'accions
        actions: array d'accions possibles
        """
        super(PGReinforce, self).__init__()
         ###################################
        ####TODO: Inicialitzar variables####
        self.input_shape = (4,84,84)
        self.n_outputs = env.action_space.n
        self.actions = np.arange(env.action_space.n)
        self.device = 'cpu'
        self.learning_rate = learning_rate
        ######

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

        if self.device == 'cuda':
            self.red_cnn.cuda()

        self.fc_layer_inputs = self.feature_size()
        self.red_lineal = nn.Sequential(
            nn.Linear(self.fc_layer_inputs, 512, bias=True),
            nn.Tanh(),
            nn.Linear(512, self.n_outputs, bias=True),
            nn.Softmax(dim=-1)
        )

        ### S'ofereix l'opció de treballar amb cuda
        if self.device == 'cuda':
            self.red_lineal.cuda()


        #######################################
        ##TODO: Inicialitzar l'optimizador
        self.optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        #####

    # Obtenció de les probabilitats de les possibles accions
    def get_action_prob(self, state):
        if type(state) is tuple:
            state = np.array(state)        
        state_t = torch.FloatTensor(state).to(device=self.device)
        cnn_out = self.red_cnn(state_t).reshape(-1,  self.fc_layer_inputs)   
        cnn_lineal_out = self.red_lineal(cnn_out)
        # clipped per evitar NaNs.
        cnn_lineal_out_clamped = torch.clamp(cnn_lineal_out, 1e-8, 1-1e-8)
        return cnn_lineal_out_clamped

    def feature_size(self):
        return self.red_cnn(autograd.Variable( torch.zeros(1, * self.input_shape)).to(device=self.device)).view(1, -1).size(1)