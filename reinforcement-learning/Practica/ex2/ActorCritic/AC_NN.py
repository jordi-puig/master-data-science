import torch
import torch as T
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical

class ACNetwork(nn.Module):
    """ Actor-Critic Neural Network """
 
    def __init__(self, n_state, n_action, seed, n_layer=128):
        """
        Inicialització de la xarxa neuronal
        Params
        =======
            n_state (int): Dimensions de l'espai d'estats
            n_action (int): Dimensions de l'espai d'accions
            n_layer1 (int): Nombre de nodes en la primera capa oculta
            n_layer2 (int): Nombre de nodes en la segona capa oculta
            seed (int): Random seed per a inicialitzar els valors aleatoris
        """
        super(ACNetwork, self).__init__()
        self.affine = nn.Linear(8, n_layer)
        
        self.action_layer = nn.Linear(n_layer, 4)
        self.value_layer = nn.Linear(n_layer, 1)
        

    def forward(self, state):
        """
        Forward pass de la xarxa neuronal per a calcular la probabilitat de cada acció i el valor de l'estat actual.        
        """
        
        state = torch.from_numpy(state).float()
        state = F.relu(self.affine(state))
         
        state = self.value_layer(state)
        
        action_probs = F.softmax(self.action_layer(state))
        return Categorical(action_probs)